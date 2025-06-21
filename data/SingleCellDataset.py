import torch
import numpy as np
from scipy.sparse import issparse
from torch.utils.data import Dataset
import pandas as pd


class SingleCellDataset(Dataset):
    """
    PyTorch Dataset for single-cell RNA sequencing data.
    
    This dataset handles single-cell gene expression data stored in AnnData format,
    preparing it for machine learning tasks including masked language modeling (MLM)
    and expression prediction. It supports both sparse and dense expression matrices
    and handles cell-level metadata such as cell type, disease, and tissue annotations.
    
    Args:
        adata (AnnData): Annotated data object containing gene expression matrix
                        and cell metadata. Expected to have:
                        - adata.X: gene expression matrix (cells x genes)
                        - adata.var_names: gene names/identifiers
                        - adata.obs: cell metadata DataFrame
        gene_vocab (dict): Mapping from gene names to vocabulary indices
                          Used to convert gene names to integer tokens
        cell_type_vocab (dict, optional): Mapping from cell type names to indices
        disease_vocab (dict, optional): Mapping from disease names to indices  
        tissue_vocab (dict, optional): Mapping from tissue names to indices
        num_bins (int, default=51): Number of bins for discretizing expression values
                                   Used for masked language modeling tasks
        seq_len (int, default=512): Maximum sequence length for genes per cell
                                   Longer sequences are truncated, shorter ones padded
    """
    
    def __init__(self, adata, gene_vocab, cell_type_vocab=None, disease_vocab=None, 
                 tissue_vocab=None, num_bins=51, seq_len=512):
        self.adata = adata
        self.gene_vocab = gene_vocab
        self.cell_type_vocab = cell_type_vocab
        self.disease_vocab = disease_vocab
        self.tissue_vocab = tissue_vocab
        self.num_bins = num_bins
        self.seq_len = seq_len
        
        # Pre-compute gene IDs mapping for all genes in the dataset
        self.gene_ids = [self.gene_vocab.get(gene, 0) for gene in self.adata.var_names]
        
    def _bin_expression(self, expr_values):
        """
        Convert continuous expression values to discrete bins for MLM training.
        
        This method applies log1p transformation to handle the skewed nature of
        gene expression data, then discretizes values into bins. This is commonly
        used for masked language modeling where discrete tokens are needed.
        
        Args:
            expr_values (np.ndarray): Raw expression values for a single cell
            
        Returns:
            np.ndarray: Binned expression values as integers in range [0, num_bins-1]
                       where 0 typically represents no/low expression
        """
        # Handle invalid values (inf, -inf, nan) by replacing with 0
        expr_values = np.nan_to_num(expr_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure all values are non-negative for log1p
        expr_values = np.maximum(expr_values, 0.0)
        
        # Apply log1p transformation to handle zero values and reduce skewness
        # log1p(x) = log(1 + x) is numerically stable for small values
        log_expr = np.log1p(expr_values)
        
        # Handle any remaining invalid values after log1p
        log_expr = np.nan_to_num(log_expr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize and bin the values based on the maximum value in this cell
        max_val = np.max(log_expr)
        if max_val > 0:
            # Scale to [0, num_bins-1] range and floor to get integer bins
            binned = np.floor(log_expr * (self.num_bins - 1) / max_val).astype(int)
        else:
            # Handle edge case where all expression values are zero
            binned = np.zeros_like(log_expr, dtype=int)
            
        # Ensure all values are within valid bin range (defensive programming)
        binned = np.clip(binned, 0, self.num_bins - 1)
        return binned
        
    def __len__(self):
        """Return the number of cells in the dataset."""
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        """
        Get a single cell's data for training.
        
        This method processes a single cell to create multiple representations
        of the gene expression data suitable for different ML tasks:
        - Discrete binned values for masked language modeling
        - Continuous values for regression tasks
        - Metadata for conditional generation/classification
        
        Args:
            idx (int): Index of the cell to retrieve
            
        Returns:
            dict: Dictionary containing:
                - gene_ids: Gene vocabulary indices (torch.long)
                - gene_expr: Binned expression values for MLM (torch.long) 
                - expr_cont: Continuous expression values for regression (torch.float)
                - cell_type: Cell type index (torch.long)
                - disease: Disease condition index (torch.long)
                - tissue: Tissue type index (torch.long)
        """
        # Get gene expression data for this cell (row idx)
        X = self.adata.X[idx]
        
        # Handle both sparse and dense matrices
        if issparse(X):
            # Convert sparse matrix to dense numpy array
            X = X.toarray().flatten()
        else:
            # Ensure we have a 1D array
            X = X.flatten() if X.ndim > 1 else X
            
        # Ensure X has the correct length (same as number of genes)
        expected_length = len(self.adata.var_names)
        if len(X) != expected_length:
            raise ValueError(f"Expression data length {len(X)} doesn't match number of genes {expected_length}")
        
        # Handle invalid values in expression data
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert continuous expression to discrete bins for MLM training
        binned_expr = self._bin_expression(X)
        
        # Create fixed-length tensors by padding/truncating to seq_len
        # Use pre-computed gene IDs
        gene_ids = np.array(self.gene_ids, dtype=np.int64)
        
        # Pad or truncate all sequences to seq_len
        if len(gene_ids) >= self.seq_len:
            # Truncate if longer than seq_len
            gene_ids = gene_ids[:self.seq_len]
            binned_expr = binned_expr[:self.seq_len]
            X = X[:self.seq_len]
        else:
            # Pad if shorter than seq_len
            pad_length = self.seq_len - len(gene_ids)
            gene_ids = np.pad(gene_ids, (0, pad_length), mode='constant', constant_values=0)
            binned_expr = np.pad(binned_expr, (0, pad_length), mode='constant', constant_values=0)
            X = np.pad(X, (0, pad_length), mode='constant', constant_values=0.0)
        
        # Extract cell metadata - handle missing columns gracefully
        obs = self.adata.obs.iloc[idx]
        
        # Get cell type index, defaulting to 0 (unknown) if not available
        cell_type = 0
        if self.cell_type_vocab and 'cell_type' in obs:
            cell_type_val = obs['cell_type']
            if pd.notna(cell_type_val):  # Check for NaN values
                cell_type = self.cell_type_vocab.get(str(cell_type_val), 0)
            
        # Get disease condition index, defaulting to 0 (unknown/healthy) if not available
        disease = 0 
        if self.disease_vocab and 'disease' in obs:
            disease_val = obs['disease']
            if pd.notna(disease_val):  # Check for NaN values
                disease = self.disease_vocab.get(str(disease_val), 0)
            
        # Get tissue type index, defaulting to 0 (unknown) if not available
        tissue = 0
        if self.tissue_vocab and 'tissue' in obs:
            tissue_val = obs['tissue']
            if pd.notna(tissue_val):  # Check for NaN values
                tissue = self.tissue_vocab.get(str(tissue_val), 0)
        
        # Return all data as PyTorch tensors with appropriate dtypes and fixed shapes
        return {
            'gene_ids': torch.tensor(gene_ids, dtype=torch.long),          # Shape: (seq_len,)
            'gene_expr': torch.tensor(binned_expr, dtype=torch.long),      # Shape: (seq_len,)
            'expr_cont': torch.tensor(X, dtype=torch.float),               # Shape: (seq_len,)
            'cell_type': torch.tensor(cell_type, dtype=torch.long),        # Shape: scalar
            'disease': torch.tensor(disease, dtype=torch.long),            # Shape: scalar
            'tissue': torch.tensor(tissue, dtype=torch.long)               # Shape: scalar
        }