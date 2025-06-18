import anndata
import json
import os
from pathlib import Path
import pandas as pd
import argparse
import torch
import requests
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.checkpoint import checkpoint

def download_dataset(url, filename):
    """
    Downloads a dataset from a given URL and saves it to the specified filename with a progress bar.

    Parameters:
    - url: str, URL of the dataset to download
    - filename: str, local path to save the downloaded file

    Returns:
    - None
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as f, tqdm(
        desc=filename, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print("Download complete!")

def shuffle_and_split_anndata(adata, train_size=0.8, random_state=None, train_file="train/train_adata.h5ad", eval_file="eval/eval_adata.h5ad"):
    """
    Shuffles an AnnData dataset, splits it into training and evaluation sets, and saves them to files.
    
    Parameters:
    - adata: AnnData object
    - train_size: float, proportion of dataset to include in the train split (default: 0.8)
    - random_state: int, random seed for reproducibility (default: None)
    - train_file: str, file path to save the training AnnData object (default: 'train/train_adata.h5ad')
    - eval_file: str, file path to save the evaluation AnnData object (default: 'eval/eval_adata.h5ad')
    
    Returns:
    - train_adata: AnnData object for training
    - eval_adata: AnnData object for evaluation
    """
    # Get the number of observations
    n_obs = adata.n_obs
    
    # Generate indices and shuffle them
    indices = np.arange(n_obs)
    train_idx, eval_idx = train_test_split(
        indices, 
        train_size=train_size, 
        random_state=random_state, 
        shuffle=True
    )
    
    # Create new AnnData objects for train and eval sets
    train_adata = adata[train_idx, :].copy()
    eval_adata = adata[eval_idx, :].copy()
    
    # Save the AnnData objects to files
    train_adata.write_h5ad(train_file, compression="gzip", compression_opts=4)
    eval_adata.write_h5ad(eval_file, compression="gzip", compression_opts=4)
    
    return train_adata, eval_adata

def load_vocabulary(vocab_path):
    """Load vocabulary from a JSON file."""
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            return json.load(f)
    return None

def build_celltype_tissue_disease_vocab(adata_file, 
                                       celltype_vocab_file="vocab/celltype_vocab.json",
                                       tissue_vocab_file="vocab/tissue_vocab.json",
                                       disease_vocab_file="vocab/disease_vocab.json"):
    """
    Builds or updates vocabularies for cell type, tissue, and disease from an AnnData file, including special tokens.
    
    Parameters:
    - adata_file: str or Path, path to the .h5ad file
    - celltype_vocab_file: str or Path, path to save/load the cell type vocabulary (default: 'vocab/celltype_vocab.json')
    - tissue_vocab_file: str or Path, path to save/load the tissue vocabulary (default: 'vocab/tissue_vocab.json')
    - disease_vocab_file: str or Path, path to save/load the disease vocabulary (default: 'vocab/disease_vocab.json')
    
    Returns:
    - celltype_vocab: dict, mapping cell types and special tokens to integer indices
    - tissue_vocab: dict, mapping tissues and special tokens to integer indices
    - disease_vocab: dict, mapping diseases and special tokens to integer indices
    """
    # Load AnnData object
    adata = anndata.read_h5ad(adata_file)
    
    # Load existing vocabularies or initialize empty ones
    celltype_vocab = load_vocabulary(celltype_vocab_file) or {}
    tissue_vocab = load_vocabulary(tissue_vocab_file) or {}
    disease_vocab = load_vocabulary(disease_vocab_file) or {}
    
    # Extract unique values from obs columns
    cell_types = adata.obs['cell_type'].dropna().unique().tolist()
    tissues = adata.obs['tissue'].dropna().unique().tolist()
    diseases = adata.obs['disease'].dropna().unique().tolist()
    
    # Update cell type vocabulary
    max_idx = max(celltype_vocab.values()) + 1 if celltype_vocab else 0
    for cell_type in cell_types:
        if str(cell_type) not in celltype_vocab:
            celltype_vocab[str(cell_type)] = max_idx
            max_idx += 1
    
    # Update tissue vocabulary
    max_idx = max(tissue_vocab.values()) + 1 if tissue_vocab else 0
    for tissue in tissues:
        if str(tissue) not in tissue_vocab:
            tissue_vocab[str(tissue)] = max_idx
            max_idx += 1
    
    # Update disease vocabulary
    max_idx = max(disease_vocab.values()) + 1 if disease_vocab else 0
    for disease in diseases:
        if str(disease) not in disease_vocab:
            disease_vocab[str(disease)] = max_idx
            max_idx += 1
    
    # Save vocabularies to JSON files
    with open(celltype_vocab_file, 'w') as f:
        json.dump(celltype_vocab, f, indent=4)
    with open(tissue_vocab_file, 'w') as f:
        json.dump(tissue_vocab, f, indent=4)
    with open(disease_vocab_file, 'w') as f:
        json.dump(disease_vocab, f, indent=4)
    
    # Print vocabulary sizes and sample entries
    print(f"Cell type vocabulary size: {len(celltype_vocab)}")
    print(f"Saved to {celltype_vocab_file}")
    print("Sample cell type entries:", list(celltype_vocab.items())[:5])
    
    print(f"\nTissue vocabulary size: {len(tissue_vocab)}")
    print(f"Saved to {tissue_vocab_file}")
    print("Sample tissue entries:", list(tissue_vocab.items())[:5])
    
    print(f"\nDisease vocabulary size: {len(disease_vocab)}")
    print(f"Saved to {disease_vocab_file}")
    print("Sample disease entries:", list(disease_vocab.items())[:5])
    
    return celltype_vocab, tissue_vocab, disease_vocab

def build_gene_vocab(adata_file, vocab_file="vocab/gene_vocab.json"):
    """
    Builds or updates a gene vocabulary dictionary from an AnnData file, including special tokens.
    
    Parameters:
    - adata_file: str or Path, path to the .h5ad file
    - vocab_file: str or Path, path to save/load the vocabulary dictionary as JSON (default: 'vocab/gene_vocab.json')
    
    Returns:
    - gene_vocab: dict, mapping gene names and special tokens to integer indices
    """
    # Load AnnData object
    adata = anndata.read_h5ad(adata_file)
    
    # Load existing vocabulary or initialize empty one
    gene_vocab = load_vocabulary(vocab_file) or {}
    
    # Extract gene names from var (assuming 'feature_name' contains gene names)
    gene_names = adata.var['feature_name'].tolist()
    
    # Update vocabulary with new genes
    max_idx = max(gene_vocab.values()) + 1 if gene_vocab else 0
    for gene in gene_names:
        if gene not in gene_vocab:
            gene_vocab[gene] = max_idx
            max_idx += 1
    
    # Save vocabulary to JSON file
    with open(vocab_file, 'w') as f:
        json.dump(gene_vocab, f, indent=4)
    
    print(f"Vocabulary size: {len(gene_vocab)}")
    print(f"Saved vocabulary to {vocab_file}")
    print("Sample vocabulary entries:", list(gene_vocab.items())[:5])
    
    return gene_vocab

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )