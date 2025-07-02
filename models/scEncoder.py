import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.checkpoint import checkpoint

class scEncoder(nn.Module):
    def __init__(self, gene_vocab_size, cell_vocab_size, num_bins=51, seq_len=512, d_model=128,
                    num_heads=8, num_layers=6, dropout=0.1, device='cuda', gradient_checkpointing=True):
        super(scEncoder, self).__init__()
        self.seq_len = seq_len
        self.num_bins = num_bins
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.gradient_checkpointing = gradient_checkpointing
        self.gene_id_embedding = nn.Embedding(gene_vocab_size, d_model, device=device)
        self.gene_expr_embedding = nn.Linear(num_bins, d_model, device=device)
        self.cell_embedding = nn.Embedding(cell_vocab_size, d_model, device=device)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, 
            dim_feedforward=4 * d_model, dropout=dropout,
            device=device, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlm_gene_id = nn.Linear(d_model, gene_vocab_size, device=device)
        self.mlm_gene_expr = nn.Linear(d_model, num_bins, device=device)

        self.contrastive_head = nn.Linear(d_model, d_model, device=device)
    
    def forward(self, gene_id, gene_expr, cell_type):
        """
        Forward pass of the encoder.
        
        Args:
            gene_id (torch.Tensor): Tensor of gene IDs.
            gene_expr (torch.Tensor): Tensor of gene expression data.
            cell_type (torch.Tensor): Tensor of cell types.
        
        Returns:
            tuple: Outputs for MLM and contrastive learning.
        """
        if gene_id.device != self.device:
            gene_id = gene_id.to(self.device)
        if gene_expr.device != self.device:
            gene_expr = gene_expr.to(self.device)
        if cell_type.device != self.device:
            cell_type = cell_type.to(self.device)
        
        gene_id_emb = self.gene_id_embedding(gene_id)
        gene_expr_emb = self.gene_expr_embedding(gene_expr)
        cell_emb = self.cell_embedding(cell_type)

        # Combine embeddings
        gene_emb = gene_id_emb + gene_expr_emb # Shape [batch_size, num_genes, d_model]
        emb = torch.cat((cell_emb.unsqueeze(1), gene_emb), dim=1)  # Shape [batch_size, num_genes + 1, d_model]

        # Truncate sequences to self.seq_len and convert to list of tensors
        batch_tensors = [emb[idx][:self.seq_len] for idx in range(emb.size(0))]  # Shape: [(seq_i, d_model), ...]

        # Pad sequences to self.seq_len
        padded_emb = rnn_utils.pad_sequence(
            batch_tensors, batch_first=True, padding_value=0.0
        )  # Shape: (batch, seq_len, d_model)

        # Vectorized padding mask (batch, seq_len)
        seq_lengths = torch.tensor([tensor.size(0) for tensor in batch_tensors], device=emb.device)
        attn_mask = torch.arange(self.seq_len, device=emb.device).unsqueeze(0) < seq_lengths.unsqueeze(1)  # Shape: (batch, seq_len)

        if self.gradient_checkpointing:
            hidden_states = checkpoint(
                self.transformer_encoder,
                padded_emb,
                src_key_padding_mask=~attn_mask
            )
            
        else:
            hidden_states = self.transformer_encoder(
                padded_emb, 
                src_key_padding_mask=~attn_mask  # Invert mask for padding positions
            )

        # MLM outputs
        mlm_gene_id_out = self.mlm_gene_id(hidden_states[:, 1:])
        mlm_gene_expr_out = self.mlm_gene_expr(hidden_states[:, 1:])

        # Contrastive learning output
        contrastive_out = self.contrastive_head(hidden_states[:, 0])

        return mlm_gene_id_out, mlm_gene_expr_out, contrastive_out, attn_mask
