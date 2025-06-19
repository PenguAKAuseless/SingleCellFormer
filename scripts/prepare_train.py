import os
import sys
import gc
import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to NumPy if CuPy is not available

# Add the project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from pathlib import Path
from utils.utils import download_dataset, build_gene_vocab, build_celltype_tissue_disease_vocab
import anndata
import scanpy as sc
from sklearn.utils import shuffle

# List of dataset URLs to download
datasets = [
    "https://datasets.cellxgene.cziscience.com/0be2ec5c-4842-4de9-9da6-51aeaa0f1133.h5ad"
]

def set_device(use_gpu=True):
    """
    Set the device for computation (GPU or CPU).
    
    Parameters:
    - use_gpu: Boolean to indicate whether to use GPU if available
    
    Returns:
    - device: String indicating the device ('cuda' or 'cpu')
    - xp: Module to use for array operations (cupy or numpy)
    """
    if use_gpu and GPU_AVAILABLE:
        try:
            cp.cuda.Device(0).use()  # Use the first available GPU
            print("Using GPU for computations.")
            return 'cuda', cp
        except cp.cuda.runtime.CUDARuntimeError:
            print("GPU not available, falling back to CPU.")
            return 'cpu', np
    else:
        print("Using CPU for computations.")
        return 'cpu', np

def get_data_stats(adata, sample_size=10000):
    """
    Get global min and max from a sample of the data to determine binning range.
    
    Parameters:
    - adata: AnnData object
    - sample_size: Number of cells to sample for statistics
    
    Returns:
    - global_min: Minimum value in the dataset
    - global_max: Maximum value in the dataset
    """
    print("Computing global statistics for binning...")
    
    # Sample data for statistics if dataset is large
    if adata.n_obs > sample_size:
        print(f"Sampling {sample_size} cells from {adata.n_obs} total cells for statistics...")
        sample_indices = np.random.choice(adata.n_obs, size=sample_size, replace=False)
        X_sample = adata.X[sample_indices]
    else:
        X_sample = adata.X
    
    # Convert to dense if sparse
    if hasattr(X_sample, 'toarray'):
        X_sample = X_sample.toarray()
    
    global_min = np.min(X_sample)
    global_max = np.max(X_sample)
    
    print(f"Expression range (from sample): {global_min:.3f} to {global_max:.3f}")
    return global_min, global_max

def bin_expression_data_batched(adata, n_bins=51, batch_size=10000, use_gpu=True):
    """
    Bin the expression data into discrete bins using batched processing.
    
    Parameters:
    - adata: AnnData object
    - n_bins: Number of bins for expression values
    - batch_size: Number of cells to process at once
    - use_gpu: Boolean to indicate whether to use GPU if available
    
    Returns:
    - adata: AnnData object with binned expression data
    """
    print(f"Binning expression data into {n_bins} bins using batch processing...")
    print(f"Batch size: {batch_size} cells")
    
    # Set device
    device, xp = set_device(use_gpu)
    
    # Get global statistics for consistent binning across batches
    global_min, global_max = get_data_stats(adata)
    
    # Create bins based on global statistics
    if device == 'cuda':
        bins = cp.linspace(global_min, global_max, n_bins + 1)
    else:
        bins = np.linspace(global_min, global_max, n_bins + 1)
    
    # Prepare output array
    n_obs, n_vars = adata.shape
    X_binned = np.zeros((n_obs, n_vars), dtype=np.int16)
    
    # Process data in batches
    n_batches = (n_obs + batch_size - 1) // batch_size
    print(f"Processing {n_obs} cells in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_obs)
        
        print(f"Processing batch {batch_idx + 1}/{n_batches} (cells {start_idx}:{end_idx})...")
        
        # Get batch data
        X_batch = adata.X[start_idx:end_idx]
        
        # Convert to dense if sparse
        if hasattr(X_batch, 'toarray'):
            X_batch = X_batch.toarray()
        
        # Move to GPU if using CuPy
        if device == 'cuda':
            X_batch_gpu = cp.array(X_batch)
            
            # Bin the batch data
            X_batch_binned = cp.digitize(X_batch_gpu, bins) - 1  # -1 to make it 0-indexed
            X_batch_binned = cp.clip(X_batch_binned, 0, n_bins - 1)  # Ensure values are within bounds
            
            # Convert back to NumPy and store
            X_binned[start_idx:end_idx] = X_batch_binned.get().astype(np.int16)
            
            # Clear GPU memory
            del X_batch_gpu, X_batch_binned
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # Process on CPU
            X_batch_binned = np.digitize(X_batch, bins) - 1  # -1 to make it 0-indexed
            X_batch_binned = np.clip(X_batch_binned, 0, n_bins - 1)  # Ensure values are within bounds
            X_binned[start_idx:end_idx] = X_batch_binned.astype(np.int16)
        
        # Clear batch memory
        del X_batch
        gc.collect()
        
        if (batch_idx + 1) % 10 == 0:  # Progress update every 10 batches
            print(f"Completed {batch_idx + 1}/{n_batches} batches...")
    
    # Update the AnnData object
    adata.X = X_binned
    
    # Store binning information in uns
    adata.uns['binning_info'] = {
        'n_bins': n_bins,
        'global_min': float(global_min),
        'global_max': float(global_max),
        'bins': bins.get() if device == 'cuda' else bins.tolist()
    }
    
    print(f"Binned data range: {np.min(X_binned)} to {np.max(X_binned)}")
    return adata

def shuffle_anndata(adata, random_state=42):
    """
    Shuffle the AnnData object.
    
    Parameters:
    - adata: AnnData object
    - random_state: Random seed for reproducibility
    
    Returns:
    - adata: Shuffled AnnData object
    """
    print("Shuffling dataset...")
    
    # Get shuffled indices
    n_obs = adata.n_obs
    indices = shuffle(np.arange(n_obs), random_state=random_state)
    
    # Shuffle the data
    adata_shuffled = adata[indices].copy()
    
    print(f"Dataset shuffled: {adata_shuffled.n_obs} cells")
    return adata_shuffled

def prepare_data(use_gpu=True, batch_size=10000):
    """
    Downloads datasets, bins expression data, builds vocabularies, and shuffles data.
    
    Parameters:
    - use_gpu: Boolean to indicate whether to use GPU if available
    - batch_size: Number of cells to process at once during binning
    """
    # Create directories if they don't exist
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("vocab", exist_ok=True)

    # Download and process each dataset
    for idx, url in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"Processing dataset {idx+1}/{len(datasets)}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        
        # Download dataset
        temp_filename = f"dataset/temp_dataset_{idx}.h5ad"
        final_filename = f"dataset/dataset_{idx}.h5ad"
        
        print(f"Downloading dataset {idx+1}/{len(datasets)}...")
        download_dataset(url, temp_filename)
        
        # Load the downloaded dataset
        print("Loading dataset...")
        adata = sc.read_h5ad(temp_filename)
        print(f"Original dataset shape: {adata.shape}")
        print(f"Original data type: {adata.X.dtype}")
        
        # Calculate memory requirements
        memory_gb = (adata.n_obs * adata.n_vars * 4) / (1024**3)  # 4 bytes per float32
        print(f"Estimated memory requirement: {memory_gb:.2f} GB")
        
        # Build vocabularies before binning (in case we need original expression values)
        print(f"\nBuilding vocabularies for dataset {idx+1}")
        gene_vocab = build_gene_vocab(
            adata_file=temp_filename,
            vocab_file=f"vocab/gene_vocab_{idx}.json"
        )
        celltype_vocab, tissue_vocab, disease_vocab = build_celltype_tissue_disease_vocab(
            adata_file=temp_filename,
            celltype_vocab_file=f"vocab/celltype_vocab_{idx}.json",
            tissue_vocab_file=f"vocab/tissue_vocab_{idx}.json",
            disease_vocab_file=f"vocab/disease_vocab_{idx}.json"
        )
        
        # Adjust batch size based on available memory and dataset size
        if memory_gb > 50:  # If dataset is very large
            suggested_batch_size = min(batch_size, 5000)
            print(f"Large dataset detected. Using smaller batch size: {suggested_batch_size}")
        else:
            suggested_batch_size = batch_size
        
        # Bin expression data using batched processing
        adata_binned = bin_expression_data_batched(
            adata, 
            n_bins=50, 
            batch_size=suggested_batch_size,
            use_gpu=use_gpu
        )
        
        # Shuffle the dataset
        adata_shuffled = shuffle_anndata(adata_binned, random_state=42)
        
        # Save the processed dataset (compressed by default in h5ad format)
        print(f"Saving processed dataset to {final_filename}...")
        adata_shuffled.write_h5ad(final_filename, compression='gzip')
        
        # Remove temporary file
        os.remove(temp_filename)
        print(f"Removed temporary file: {temp_filename}")
        
        # Memory cleanup
        del adata, adata_binned, adata_shuffled
        gc.collect()
        
        # Clear GPU memory if using GPU
        if use_gpu and GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        
        print(f"Dataset {idx+1} processing complete!")
        print(f"Final file: {final_filename}")
        
        # Print file size comparison
        file_size_mb = os.path.getsize(final_filename) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

def create_combined_vocabularies():
    """
    Create combined vocabularies from all datasets.
    """
    print("\nCreating combined vocabularies...")
    
    all_genes = set()
    all_celltypes = set()
    all_tissues = set()
    all_diseases = set()
    
    # Collect all unique values from individual vocabularies
    for idx in range(len(datasets)):
        try:
            import json
            
            # Load individual vocabularies
            with open(f"vocab/gene_vocab_{idx}.json", 'r') as f:
                gene_vocab = json.load(f)
                all_genes.update(gene_vocab.keys())
            
            with open(f"vocab/celltype_vocab_{idx}.json", 'r') as f:
                celltype_vocab = json.load(f)
                all_celltypes.update(celltype_vocab.keys())
            
            with open(f"vocab/tissue_vocab_{idx}.json", 'r') as f:
                tissue_vocab = json.load(f)
                all_tissues.update(tissue_vocab.keys())
            
            with open(f"vocab/disease_vocab_{idx}.json", 'r') as f:
                disease_vocab = json.load(f)
                all_diseases.update(disease_vocab.keys())
                
        except FileNotFoundError:
            print(f"Warning: Could not find vocabulary files for dataset {idx}")
            continue
    
    # Create combined vocabularies
    combined_gene_vocab = {gene: idx for idx, gene in enumerate(sorted(all_genes))}
    combined_celltype_vocab = {celltype: idx for idx, celltype in enumerate(sorted(all_celltypes))}
    combined_tissue_vocab = {tissue: idx for idx, tissue in enumerate(sorted(all_tissues))}
    combined_disease_vocab = {disease: idx for idx, disease in enumerate(sorted(all_diseases))}
    
    # Save combined vocabularies
    import json
    with open("vocab/gene_vocab.json", 'w') as f:
        json.dump(combined_gene_vocab, f, indent=2)
    
    with open("vocab/celltype_vocab.json", 'w') as f:
        json.dump(combined_celltype_vocab, f, indent=2)
    
    with open("vocab/tissue_vocab.json", 'w') as f:
        json.dump(combined_tissue_vocab, f, indent=2)
    
    with open("vocab/disease_vocab.json", 'w') as f:
        json.dump(combined_disease_vocab, f, indent=2)
    
    print(f"Combined vocabularies created:")
    print(f"  - Genes: {len(combined_gene_vocab)}")
    print(f"  - Cell types: {len(combined_celltype_vocab)}")
    print(f"  - Tissues: {len(combined_tissue_vocab)}")
    print(f"  - Diseases: {len(combined_disease_vocab)}")

if __name__ == "__main__":
    # You can adjust the batch_size based on your GPU memory
    # For 12GB GPU: batch_size=5000-10000
    # For 24GB GPU: batch_size=10000-20000
    # For 48GB GPU: batch_size=20000-40000
    prepare_data(use_gpu=True, batch_size=10000)
    create_combined_vocabularies()
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("- Downloaded datasets")
    print("- Binned expression values (batched processing)")
    print("- Shuffled datasets") 
    print("- Compressed and saved processed files")
    print("- Created vocabularies")
    print("="*60)