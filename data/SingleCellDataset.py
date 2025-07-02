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
            gene_vocab (list): List of gene names for vocabulary.
            cell_vocab (list): List of cell types for vocabulary.
            num_bins (int): Number of bins for gene expression data.
        """
        self.adata = adata
        self.num_bins = num_bins
        self.seq_len = seq_len
        
        self.gene_vocab = gene_vocab
        self.cell_vocab = cell_vocab