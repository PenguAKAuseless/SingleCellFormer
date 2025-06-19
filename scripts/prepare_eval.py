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

def bin_expression_data(adata, n_bins=51, use_gpu=True):
    """
    Bin the expression data into discrete bins using GPU if available.
    
    Parameters:
    - adata: AnnData object
    - n_bins: Number of bins for expression values
    - use_gpu: Boolean to indicate whether to use GPU if available
    
    Returns:
    - adata: AnnData object with binned expression data
    """
    print(f"Binning expression data into {n_bins} bins...")
    
    # Set device
    device, xp = set_device(use_gpu)
    
    # Get the expression matrix
    X = adata.X
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Move to GPU if using CuPy
    if device == 'cuda':
        X = xp.array(X)
    
    # Calculate global min and max for consistent binning
    global_min = xp.min(X)
    global_max = xp.max(X)
    
    # Convert to float for printing
    global_min_float = float(global_min) if device == 'cuda' else global_min
    global_max_float = float(global_max) if device == 'cuda' else global_max
    print(f"Expression range: {global_min_float:.3f} to {global_max_float:.3f}")
    
    # Create bins
    bins = xp.linspace(global_min, global_max, n_bins + 1)
    
    # Bin the data
    X_binned = xp.digitize(X, bins) - 1  # -1 to make it 0-indexed
    X_binned = xp.clip(X_binned, 0, n_bins - 1)  # Ensure values are within bounds
    
    # Convert back to NumPy for AnnData compatibility
    if device == 'cuda':
        X_binned = X_binned.get()  # Transfer back to CPU
    
    # Update the AnnData object
    adata.X = X_binned.astype(np.int16)  # Use int16 to save memory
    
    # Store binning information in uns
    adata.uns['binning_info'] = {
        'n_bins': n_bins,
        'global_min': float(global_min),
        'global_max': float(global_max),
        'bins': bins.get() if device == 'cuda' else bins
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

def prepare_data(use_gpu=True):
    """
    Downloads datasets, bins expression data, builds vocabularies, and shuffles data.
    
    Parameters:
    - use_gpu: Boolean to indicate whether to use GPU if available
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
        
        # Bin expression data
        adata_binned = bin_expression_data(adata, n_bins=50, use_gpu=use_gpu)
        
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
    prepare_data(use_gpu=True)
    create_combined_vocabularies()
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("- Downloaded datasets")
    print("- Binned expression values")
    print("- Shuffled datasets") 
    print("- Compressed and saved processed files")
    print("- Created vocabularies")
    print("="*60)