import torch
import numpy as np
from scipy.sparse import issparse
from torch.utils.data import Dataset
import pandas as pd

class SingleCellDataset(Dataset):
    """ 
    A PyTorch Dataset for single-cell RNA-seq data.
    """
    def __init__(self, adata, gene_vocab=None, cell_vocab=None, num_bins=51, seq_len=512):
        """
        Args:
            adata (AnnData): Annotated data matrix.
            gene_vocab (list): Dict mapping gene names to indices.
            cell_vocab (list): List of cell types for vocabulary.
            num_bins (int): Number of bins for gene expression data.
        """
        self.adata = adata
        self.num_bins = num_bins
        self.seq_len = seq_len
        
        self.gene_vocab = gene_vocab
        self.cell_vocab = cell_vocab
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        # Get gene expression data for this cell
        if issparse(self.adata.X):
            gene_expr = self.adata.X[idx].toarray().flatten()
        else:
            gene_expr = self.adata.X[idx].flatten()
        
        # Get gene names
        gene_names = self.adata.var_names
        
        # Pre-allocate arrays for better performance
        max_genes = min(len(gene_names), self.seq_len - 1)  # -1 for CLS token
        gene_ids = np.zeros(max_genes, dtype=np.int64)
        gene_expressions = np.zeros(max_genes, dtype=np.int64)
        
        # Vectorized operations where possible
        valid_idx = 0
        for i, gene in enumerate(gene_names):
            if valid_idx >= max_genes:
                break
            if self.gene_vocab is not None and gene in self.gene_vocab:
                gene_ids[valid_idx] = self.gene_vocab[gene]
                # Bin the expression value
                binned_expr = min(int(gene_expr[i] * self.num_bins), self.num_bins - 1)
                gene_expressions[valid_idx] = binned_expr
                valid_idx += 1
        
        # Trim to actual used length
        gene_ids = gene_ids[:valid_idx]
        gene_expressions = gene_expressions[:valid_idx]
        
        # Add cls token at the beginning
        gene_ids = np.concatenate([[0], gene_ids])  # Assuming 0 is the CLS token
        gene_expressions = np.concatenate([[0], gene_expressions])  # Assuming 0 is the CLS token for expression
        
        # Pad to seq_len using numpy (more efficient than python loops)
        if len(gene_ids) < self.seq_len:
            pad_length = self.seq_len - len(gene_ids)
            gene_ids = np.pad(gene_ids, (0, pad_length), mode='constant', constant_values=0)
            gene_expressions = np.pad(gene_expressions, (0, pad_length), mode='constant', constant_values=0)
        else:
            # Truncate to seq_len
            gene_ids = gene_ids[:self.seq_len]
            gene_expressions = gene_expressions[:self.seq_len]
        
        # Get cell type if available
        cell_type = 0  # Default cell type
        if 'cell_type' in self.adata.obs.columns:
            cell_type_name = self.adata.obs['cell_type'].iloc[idx]
            if self.cell_vocab is not None and cell_type_name in self.cell_vocab:
                cell_type = self.cell_vocab[cell_type_name]
        
        return {
            'gene_ids': torch.from_numpy(gene_ids).long(),
            'gene_expr': torch.from_numpy(gene_expressions).long(),
            'cell_type': torch.tensor(cell_type, dtype=torch.long)
        }