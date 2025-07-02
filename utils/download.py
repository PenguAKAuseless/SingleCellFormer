import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
import json
import logging
import anndata
import torch
import aiohttp
import asyncio
import dask.dataframe as dd
from multiprocessing import Pool
from scipy.sparse import issparse
import multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def download_dataset_async(url, save_path):
    """Download a dataset asynchronously from a URL and save it locally."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
        logger.info(f"Downloaded dataset to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

async def download_all_datasets(dataset_paths, output_dir):
    """Download all datasets asynchronously."""
    tasks = []
    for idx, url in enumerate(dataset_paths):
        file_name = f"dataset_{idx}.h5ad"
        local_path = os.path.join(output_dir, file_name)
        tasks.append(download_dataset_async(url, local_path))
    return await asyncio.gather(*tasks, return_exceptions=True)

def bin_expression_batch(expression_matrix, num_bins=10, device='cuda:0'):
    """Bin gene expression values into discrete categories using PyTorch on CUDA."""
    try:
        expr_tensor = torch.tensor(expression_matrix, dtype=torch.float32, device=device)
        expr_min = torch.min(expr_tensor, dim=0)[0]
        expr_max = torch.max(expr_tensor, dim=0)[0]
        bins = torch.linspace(0, 1, steps=num_bins + 1, device=device)
        binned = torch.zeros_like(expr_tensor, dtype=torch.long)
        for i in range(expr_tensor.shape[1]):
            scaled = (expr_tensor[:, i] - expr_min[i]) / (expr_max[i] - expr_min[i] + 1e-10)
            binned[:, i] = torch.bucketize(scaled, bins, right=True)
        return binned.cpu().numpy()
    except Exception as e:
        logger.error(f"Error binning expression data: {e}")
        return expression_matrix

def create_vocabularies(df, gene_id_col, cell_type_col):
    """Create vocabularies for gene_id and cell_type for a single dataset/chunk."""
    try:
        unique_gene_ids = sorted(df[gene_id_col].unique().compute())
        gene_id_vocab = {gene_id: idx for idx, gene_id in enumerate(unique_gene_ids)}
        unique_cell_types = sorted(df[cell_type_col].unique().compute())
        cell_type_vocab = {cell_type: idx for idx, cell_type in enumerate(unique_cell_types)}
        return gene_id_vocab, cell_type_vocab
    except Exception as e:
        logger.error(f"Error creating vocabularies: {e}")
        return None, None

def merge_vocabularies(vocab_list, vocab_type):
    """Merge vocabularies from multiple datasets/chunks, ensuring unique entries."""
    merged_vocab = {}
    current_idx = 0
    for vocab in vocab_list:
        for key in sorted(vocab.keys()):  # Sort to ensure consistent ordering
            if key not in merged_vocab:
                merged_vocab[key] = current_idx
                current_idx += 1
    logger.info(f"Merged {vocab_type} vocabulary with {len(merged_vocab)} unique entries")
    return merged_vocab

def save_vocabulary(vocab, output_path):
    """Save vocabulary to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        logger.info(f"Saved vocabulary to {output_path}")
    except Exception as e:
        logger.error(f"Error saving vocabulary to {output_path}: {e}")

def preprocess_h5ad_dataset(file_path, gene_id_col='gene_id', cell_type_col='cell_type', num_bins=10, device='cuda:0', chunk_size=500):
    """Preprocess a .h5ad dataset: extract gene_id, bin gene_expr, and extract cell_type."""
    try:
        adata = anndata.read_h5ad(file_path, backed='r')
        cell_types = adata.obs[cell_type_col] if cell_type_col in adata.obs.columns else pd.Series(["unknown"] * adata.n_obs)
        gene_ids = adata.var_names
        
        # Process in smaller chunks to reduce memory usage
        binned_dfs = []
        for start in range(0, adata.n_obs, chunk_size):
            end = min(start + chunk_size, adata.n_obs)
            chunk = adata[start:end].to_memory()
            expr = chunk.X.toarray() if issparse(chunk.X) else chunk.X
            df_chunk = pd.DataFrame(expr, columns=gene_ids)
            df_chunk[cell_type_col] = cell_types[start:end].values
            
            # Bin expression data
            binned_expr = bin_expression_batch(expr, num_bins=num_bins, device=device)
            binned_df = pd.DataFrame(binned_expr, columns=[f"{gene}_binned" for gene in gene_ids])
            df_chunk = pd.concat([df_chunk, binned_df], axis=1)
            
            # Convert to long format
            df_long = pd.melt(df_chunk, id_vars=[cell_type_col], value_vars=[f"{g}_binned" for g in gene_ids],
                              var_name="gene_id", value_name="binned_expr")
            df_long["gene_id"] = df_long["gene_id"].str.replace("_binned", "")
            binned_dfs.append(df_long)
        
        preprocessed_df = pd.concat(binned_dfs, ignore_index=True)
        logger.info(f"Preprocessed h5ad dataset {file_path}")
        
        # Create vocabularies for this dataset
        gene_id_vocab, cell_type_vocab = create_vocabularies(preprocessed_df, gene_id_col, cell_type_col)
        return preprocessed_df, gene_id_vocab, cell_type_vocab
    except Exception as e:
        logger.error(f"Error preprocessing h5ad {file_path}: {e}")
        return None, None, None

def save_preprocessed_dataset(df, output_path):
    """Save the preprocessed dataset to a Parquet file."""
    try:
        df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"Saved preprocessed dataset to {output_path}")
    except Exception as e:
        logger.error(f"Error saving dataset to {output_path}: {e}")

def process_single_dataset(args):
    """Process a single dataset and return preprocessed data and vocabularies."""
    idx, path, output_dir, num_bins, device, chunk_size = args
    file_name = f"dataset_{idx}.h5ad"
    local_path = os.path.join(output_dir, file_name)
    output_path = os.path.join(output_dir, f"preprocessed_dataset_{idx}.parquet")
    vocab_dir = os.path.join(output_dir, "vocab")
    Path(vocab_dir).mkdir(parents=True, exist_ok=True)
    
    # Download if path is a URL
    if path.startswith(('http://', 'https://')):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(download_dataset_async(path, local_path))
        loop.close()
        if not result:
            return None, None, None
    
    # Preprocess the dataset and create vocabularies
    preprocessed_df, gene_id_vocab, cell_type_vocab = preprocess_h5ad_dataset(
        local_path, num_bins=num_bins, device=device, chunk_size=chunk_size
    )
    if preprocessed_df is not None:
        save_preprocessed_dataset(preprocessed_df, output_path)
        # Save intermediate vocabularies
        save_vocabulary(gene_id_vocab, os.path.join(vocab_dir, f"gene_id_vocab_{idx}.json"))
        save_vocabulary(cell_type_vocab, os.path.join(vocab_dir, f"cell_type_vocab_{idx}.json"))
    
    return preprocessed_df, gene_id_vocab, cell_type_vocab

def main(dataset_paths, output_dir="preprocessed_datasets", gene_id_vocab_path="vocab/gene_id_vocab.json", 
         cell_type_vocab_path="vocab/cell_type_vocab.json", num_bins=10, device='cuda:0', chunk_size=500):
    """Main function to download, preprocess datasets, and create/save vocabularies."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        device = 'cpu'
    else:
        logger.info(f"Using device: {device}")
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "vocab")).mkdir(parents=True, exist_ok=True)
    
    # Set the 'spawn' start method for multiprocessing to avoid CUDA issues
    mp.set_start_method('spawn', force=True)
    
    # Process datasets in parallel
    with Pool(processes=os.cpu_count()) as pool:
        args = [(idx, path, output_dir, num_bins, device, chunk_size) for idx, path in enumerate(dataset_paths)]
        results = pool.map(process_single_dataset, args)
    
    # Collect and merge vocabularies
    all_preprocessed = [r[0] for r in results if r[0] is not None]
    gene_id_vocabs = [r[1] for r in results if r[1] is not None]
    cell_type_vocabs = [r[2] for r in results if r[2] is not None]
    
    if all_preprocessed:
        # Merge vocabularies
        final_gene_id_vocab = merge_vocabularies(gene_id_vocabs, "gene_id")
        final_cell_type_vocab = merge_vocabularies(cell_type_vocabs, "cell_type")
        
        # Save final vocabularies
        save_vocabulary(final_gene_id_vocab, os.path.join(output_dir, gene_id_vocab_path))
        save_vocabulary(final_cell_type_vocab, os.path.join(output_dir, cell_type_vocab_path))
        
        # Optional: Combine all preprocessed datasets if needed
        # combined_df = dd.concat(all_preprocessed)
        # save_preprocessed_dataset(combined_df, os.path.join(output_dir, "combined_preprocessed.parquet"))
        logger.info("Preprocessing and vocabulary creation completed.")
    else:
        logger.error("No datasets were successfully preprocessed.")

if __name__ == "__main__":
    # Example list of dataset paths (replace with actual paths or URLs)
    dataset_paths = [
        "https://datasets.cellxgene.cziscience.com/ffdaa1f0-b1d1-4135-8774-9fed7bf039ba.h5ad",
        "https://datasets.cellxgene.cziscience.com/d80e6ae7-2848-48ea-a898-70c80ae349c2.h5ad",
        "https://datasets.cellxgene.cziscience.com/0483d387-4ef6-4bcc-af68-6e3127979711.h5ad",
        "https://datasets.cellxgene.cziscience.com/3cbdbdaa-098a-42db-a601-d4c6454925e5.h5ad",
        "https://datasets.cellxgene.cziscience.com/bdae7c8d-5d2c-45a2-a149-8ba7d9260926.h5ad",
        "https://datasets.cellxgene.cziscience.com/b8eeb150-0420-4b56-b8b1-b8e488783949.h5ad",
        "https://datasets.cellxgene.cziscience.com/b77b62f8-16f9-42d0-9967-5802c0cd8ee2.h5ad",
        "https://datasets.cellxgene.cziscience.com/06c971c0-569d-494a-9fd0-60ef96c2da45.h5ad",
        "https://datasets.cellxgene.cziscience.com/4e6cf682-3aa0-4c79-a6e1-8abc21a85146.h5ad",
        "https://datasets.cellxgene.cziscience.com/348af6fd-d958-4d8c-9bd1-1062f54e2cc8.h5ad",
        "https://datasets.cellxgene.cziscience.com/3091260a-9c1e-461d-91c3-ab5e309b90c9.h5ad",
        "https://datasets.cellxgene.cziscience.com/c6d0a970-5043-4e38-8509-125a80edf930.h5ad"
    ]
    
    # Run the preprocessing pipeline
    main(
        dataset_paths,
        output_dir="preprocessed",
        gene_id_vocab_path="vocab/gene_id_vocab.json",
        cell_type_vocab_path="vocab/cell_type_vocab.json",
        num_bins=10,
        device='cuda:0',
        chunk_size=500  # Reduced chunk size for better memory management
    )