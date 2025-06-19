import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint
import warnings

class scDecoder(nn.Module):
    def __init__(
            self, gene_vocab_size, 
            cell_type_vocab_size=None, disease_vocab_size=None, tissue_vocab_size=None, 
            batch_first=True, num_bins=51, seq_len=512, d_model=512, nhead=8,
            num_layers=12, hidden_dim=2048, dropout=0.1, mask_prob=0.15,
            gradient_checkpointing=False
    ):
        """
        Initializes the scDecoder model for autoregressive gene expression generation.

        Args:
            gene_vocab_size (int): Size of the gene vocabulary.
            cell_type_vocab_size (int, optional): Size of the cell type vocabulary. Default is None.
            disease_vocab_size (int, optional): Size of the disease vocabulary. Default is None.
            tissue_vocab_size (int, optional): Size of the tissue vocabulary. Default is None.
            batch_first (bool): If True, the input and output tensors are provided as (batch, seq, feature). Default is True.
            num_bins (int): Number of bins for gene expression values. Default is 51.
            seq_len (int): Length of the input sequence. Default is 512.
            d_model (int): Dimension of the model. Default is 512.
            nhead (int): Number of attention heads. Default is 8.
            num_layers (int): Number of transformer decoder layers. Default is 12.
            hidden_dim (int): Dimension of the feedforward network in the transformer. Default is 2048.
            dropout (float): Dropout rate. Default is 0.1.
            mask_prob (float): Probability of masking genes for autoregressive learning. Default is 0.15.
            gradient_checkpointing (bool): Enable gradient checkpointing to trade computation
                for memory during training. Useful for large models. Default: False.
        """
        super(scDecoder, self).__init__()
        
        # Store model configuration parameters
        self.gene_vocab_size = gene_vocab_size
        self.cell_type_vocab_size = cell_type_vocab_size
        self.disease_vocab_size = disease_vocab_size
        self.tissue_vocab_size = tissue_vocab_size
        self.batch_first = batch_first
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.mask_prob = mask_prob
        self.gradient_checkpointing = gradient_checkpointing

        # ==================== EMBEDDING LAYERS ====================
        # Gene ID embeddings: maps gene indices to dense vectors
        self.gene_embedding = nn.Embedding(gene_vocab_size, d_model)
        
        # Gene expression value embeddings: maps binned expression values to dense vectors
        self.value_embedding = nn.Embedding(num_bins, d_model)
        
        # Optional metadata embeddings (only created if vocab sizes are provided)
        self.cell_embedding = nn.Embedding(cell_type_vocab_size, d_model) if cell_type_vocab_size else None
        self.disease_embedding = nn.Embedding(disease_vocab_size, d_model) if disease_vocab_size else None
        self.tissue_embedding = nn.Embedding(tissue_vocab_size, d_model) if tissue_vocab_size else None
        
        # Special tokens for masking and padding
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pad_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Dropout layer for regularization
        self.dropout_layer = nn.Dropout(dropout)

        # ==================== TRANSFORMER DECODER ARCHITECTURE ====================
        # Define transformer decoder layer with specified architecture
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,              # Input/output dimension
            nhead=nhead,                  # Number of attention heads
            dim_feedforward=hidden_dim,   # Feedforward network dimension
            dropout=dropout,              # Dropout rate
            batch_first=batch_first       # Batch dimension ordering
        )

        # Stack multiple decoder layers
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

        # ==================== OUTPUT PREDICTION HEADS ====================
        # Gene expression prediction head (main task)
        self.gene_output = nn.Linear(d_model, gene_vocab_size)
        self.value_output = nn.Linear(d_model, num_bins)
        
        # Optional metadata prediction heads
        self.cell_output = nn.Linear(d_model, cell_type_vocab_size) if cell_type_vocab_size else None
        self.disease_output = nn.Linear(d_model, disease_vocab_size) if disease_vocab_size else None
        self.tissue_output = nn.Linear(d_model, tissue_vocab_size) if tissue_vocab_size else None

        # ==================== NORMALIZATION ====================
        # Layer normalization for output stabilization
        self.norm = nn.LayerNorm(d_model)

        # Initialize all model weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights using Xavier uniform initialization.
        """
        # Initialize all parameters with dimension > 1 using Xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize special tokens
        nn.init.normal_(self.mask_token, mean=0, std=0.02)
        nn.init.zeros_(self.pad_token)

    def create_random_mask(self, gene_ids, gene_expr):
        """
        Create random mask for autoregressive learning.
        
        Args:
            gene_ids (torch.Tensor): Gene ID tensor of shape (batch_size, seq_len)
            gene_expr (torch.Tensor): Gene expression tensor of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Boolean mask of shape (batch_size, seq_len), True for masked positions
        """
        batch_size, seq_len = gene_ids.size()
        device = gene_ids.device
        
        # Create mask for non-zero gene expressions (don't mask padding or unexpressed genes)
        valid_positions = (gene_expr > 0)
        
        # Create random mask with specified probability
        random_mask = torch.rand(batch_size, seq_len, device=device) < self.mask_prob
        
        # Only mask valid positions (expressed genes)
        mask = random_mask & valid_positions
        
        return mask

    def apply_masking(self, gene_emb, value_emb, mask):
        """
        Apply masking to embeddings by replacing masked positions with mask token.
        
        Args:
            gene_emb (torch.Tensor): Gene embeddings of shape (batch_size, seq_len, d_model)
            value_emb (torch.Tensor): Value embeddings of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Masked combined embeddings
        """
        batch_size, seq_len = mask.size()
        
        # Combine gene and value embeddings
        combined_emb = gene_emb + value_emb
        
        # Expand mask token to match batch and sequence dimensions
        mask_emb = self.mask_token.expand(batch_size, seq_len, -1)
        
        # Apply masking: use mask token where mask is True, original embedding otherwise
        masked_emb = torch.where(mask.unsqueeze(-1), mask_emb, combined_emb)
        
        return masked_emb

    def create_metadata_embeddings(self, batch_size, seq_len, cell_type=None, disease=None, tissue=None):
        """
        Create metadata embeddings and broadcast to sequence length for element-wise addition.
        
        Args:
            batch_size (int): Batch size
            seq_len (int): Sequence length
            cell_type (torch.Tensor, optional): Cell type IDs of shape (batch_size,)
            disease (torch.Tensor, optional): Disease IDs of shape (batch_size,)
            tissue (torch.Tensor, optional): Tissue IDs of shape (batch_size,)
            
        Returns:
            torch.Tensor: Combined metadata embeddings of shape (batch_size, seq_len, d_model)
        """
        # Initialize with zeros
        device = cell_type.device if cell_type is not None else 'cpu'
        combined_metadata_emb = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Add cell type embedding if available
        if self.cell_embedding is not None and cell_type is not None:
            cell_emb = self.cell_embedding(cell_type)  # (batch_size, d_model)
            # Broadcast to sequence length and add
            combined_metadata_emb += cell_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Add disease embedding if available
        if self.disease_embedding is not None and disease is not None:
            disease_emb = self.disease_embedding(disease)  # (batch_size, d_model)
            # Broadcast to sequence length and add
            combined_metadata_emb += disease_emb.unsqueeze(1).expand(-1, seq_len, -1)
            
        # Add tissue embedding if available
        if self.tissue_embedding is not None and tissue is not None:
            tissue_emb = self.tissue_embedding(tissue)  # (batch_size, d_model)
            # Broadcast to sequence length and add
            combined_metadata_emb += tissue_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        return combined_metadata_emb

    def create_causal_mask(self, seq_len, device):
        """
        Create causal (lower triangular) mask for autoregressive generation.
        
        Args:
            seq_len (int): Sequence length
            device (torch.device): Device to create mask on
            
        Returns:
            torch.Tensor: Causal mask of shape (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def _apply_gradient_checkpointing(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        Apply gradient checkpointing to transformer decoder layers.
        
        Args:
            tgt (torch.Tensor): Target sequence tensor
            memory (torch.Tensor): Memory sequence tensor
            tgt_mask (torch.Tensor): Target mask
            tgt_key_padding_mask (torch.Tensor): Target key padding mask
            memory_key_padding_mask (torch.Tensor): Memory key padding mask
            
        Returns:
            torch.Tensor: Output tensor after applying all decoder layers with checkpointing
        """
        def create_custom_forward(layer):
            """Create a custom forward function for gradient checkpointing."""
            def custom_forward(tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
                return layer(tgt, memory, tgt_mask=tgt_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
            return custom_forward
        
        # Apply each decoder layer with gradient checkpointing
        output = tgt
        for layer in self.transformer_decoder.layers:
            output = checkpoint(
                create_custom_forward(layer),
                output, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask
            )
        
        # Apply final norm if it exists
        if self.transformer_decoder.norm is not None:
            output = self.transformer_decoder.norm(output)
            
        return output

    def forward(self, x, cell_type=None, disease=None, tissue=None, create_mask=True):
        """
        Forward pass of the scDecoder model.

        Args:
            x (dict): Input data dictionary containing:
                - 'gene_ids': Tensor of shape (batch_size, seq_len) with gene IDs
                - 'gene_expr': Tensor of shape (batch_size, seq_len) with binned gene expression values
            cell_type (torch.Tensor, optional): Cell type tensor of shape (batch_size,). Default is None.
            disease (torch.Tensor, optional): Disease tensor of shape (batch_size,). Default is None.
            tissue (torch.Tensor, optional): Tissue tensor of shape (batch_size,). Default is None.
            create_mask (bool): Whether to create random mask for training. Default is True.

        Returns:
            dict: A dictionary containing:
                - 'gene_output': Gene ID predictions of shape (batch_size, seq_len, gene_vocab_size)
                - 'value_output': Gene expression predictions of shape (batch_size, seq_len, num_bins)
                - 'cell_output': Cell type predictions of shape (batch_size, cell_type_vocab_size) or None
                - 'disease_output': Disease predictions of shape (batch_size, disease_vocab_size) or None
                - 'tissue_output': Tissue predictions of shape (batch_size, tissue_vocab_size) or None
                - 'mask': Random mask used for training of shape (batch_size, seq_len) or None
                - 'hidden_states': Final hidden states of shape (batch_size, seq_len, d_model)
        """
        # Extract input tensors
        gene_ids = x['gene_ids']
        gene_expr = x['gene_expr']
        batch_size, seq_len = gene_ids.size()
        device = gene_ids.device

        # ==================== INPUT VALIDATION ====================
        if gene_ids.size() != gene_expr.size():
            raise ValueError("gene_ids and gene_expr must have the same shape")
        
        # ==================== EMBEDDING COMPUTATION ====================
        # Compute gene and value embeddings
        gene_emb = self.gene_embedding(gene_ids)
        value_emb = self.value_embedding(gene_expr)
        
        # ==================== RANDOM MASKING ====================
        mask = None
        if create_mask and self.training:
            # Create random mask for training
            mask = self.create_random_mask(gene_ids, gene_expr)
            # Apply masking to embeddings
            gene_expr_emb = self.apply_masking(gene_emb, value_emb, mask)
        else:
            # No masking during inference
            gene_expr_emb = gene_emb + value_emb

        # ==================== METADATA EMBEDDINGS ====================
        # Create metadata embeddings that will be summed with gene expression embeddings
        metadata_emb = self.create_metadata_embeddings(batch_size, seq_len, cell_type, disease, tissue)
        
        # ==================== SEQUENCE CONSTRUCTION ====================
        # Sum metadata embeddings with gene expression embeddings
        sequence_emb = gene_expr_emb + metadata_emb

        # Handle sequence length variations through padding/truncation
        if seq_len < self.seq_len:
            # Pad sequence to fixed length
            pad_size = self.seq_len - seq_len
            pad_emb = self.pad_token.expand(batch_size, pad_size, -1)
            sequence_emb = torch.cat([sequence_emb, pad_emb], dim=1)
            total_seq_len = self.seq_len
        elif seq_len > self.seq_len:
            # Truncate sequence if it exceeds maximum length
            warnings.warn("Input sequence length exceeds the model's expected sequence length. Truncating to fit.")
            sequence_emb = sequence_emb[:, :self.seq_len, :]
            total_seq_len = self.seq_len
        else:
            total_seq_len = seq_len

        # Apply dropout for regularization
        sequence_emb = self.dropout_layer(sequence_emb)

        # ==================== ATTENTION MASKS ====================
        # Create causal mask for autoregressive decoding
        causal_mask = self.create_causal_mask(total_seq_len, device)
        
        # Create padding mask to ignore padded positions
        padding_mask = torch.zeros(batch_size, total_seq_len, device=device, dtype=torch.bool)
        if total_seq_len > seq_len:
            padding_mask[:, seq_len:] = True

        # ==================== TRANSFORMER FORWARD PASS ====================
        # Apply transformer decoder with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            # This trades computation for memory by recomputing activations during backward pass
            hidden_states = self._apply_gradient_checkpointing(
                tgt=sequence_emb,
                memory=sequence_emb,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask
            )
        else:
            # Standard forward pass through transformer decoder
            # For decoder, we need memory (encoder output) - using same sequence as both input and memory
            # This is common in autoregressive models like GPT
            hidden_states = self.transformer_decoder(
                tgt=sequence_emb,
                memory=sequence_emb,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask
            )

        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)

        # ==================== OUTPUT PREDICTIONS ====================
        # Extract gene sequence part (up to original sequence length)
        gene_hidden = hidden_states[:, :seq_len, :]
        
        # Generate predictions from output heads
        outputs = {
            'gene_output': self.gene_output(gene_hidden),
            'value_output': self.value_output(gene_hidden),
            'mask': mask,
            'hidden_states': hidden_states
        }
        
        # Add metadata predictions using global average pooling of the sequence
        if any([cell_type is not None, disease is not None, tissue is not None]):
            # Use mean pooling of the entire sequence for metadata classification
            pooled_hidden = torch.mean(gene_hidden, dim=1)  # (batch_size, d_model)
            
            if self.cell_output is not None:
                outputs['cell_output'] = self.cell_output(pooled_hidden)
            if self.disease_output is not None:
                outputs['disease_output'] = self.disease_output(pooled_hidden)
            if self.tissue_output is not None:
                outputs['tissue_output'] = self.tissue_output(pooled_hidden)
        else:
            outputs['cell_output'] = None
            outputs['disease_output'] = None
            outputs['tissue_output'] = None

        return outputs