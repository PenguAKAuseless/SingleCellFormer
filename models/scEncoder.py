import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
import warnings
import random

class scEncoder(nn.Module):
    """
    Improved Single-cell RNA-seq Transformer Encoder Model
    
    Key improvements:
    1. Removed CLS token - metadata embeddings are directly summed with gene embeddings
    2. Implements proper gene masking for autoregressive learning
    3. Only accepts binned gene expression data for consistency
    4. Returns masking information for loss calculation
    5. Uses embedding summation instead of concatenation for efficiency
    """
    
    def __init__(
            self, gene_vocab_size, 
            cell_type_vocab_size=None, disease_vocab_size=None, tissue_vocab_size=None,
            batch_first=True, num_bins=51, seq_len=512,
            d_model=512, nhead=8, num_layers=12, hidden_dim=2048, dropout=0.1,
            gradient_checkpointing=False, mask_prob=0.15, mask_token_id=0
        ):
        """
        Initialize the improved scEncoder model.

        Args:   
            gene_vocab_size (int): Total number of unique genes in the vocabulary
            cell_type_vocab_size (int, optional): Number of unique cell types
            disease_vocab_size (int, optional): Number of unique diseases
            tissue_vocab_size (int, optional): Number of unique tissues
            batch_first (bool): Whether input tensors have batch dimension first
            num_bins (int): Number of discretization bins for gene expression values
            seq_len (int): Maximum sequence length
            d_model (int): Dimensionality of model embeddings and hidden states
            nhead (int): Number of attention heads in each transformer layer
            num_layers (int): Number of transformer encoder layers to stack
            hidden_dim (int): Dimensionality of feedforward network inside transformer
            dropout (float): Dropout probability applied throughout the model
            gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency
            mask_prob (float): Probability of masking each gene for MLM training
            mask_token_id (int): Special token ID used for masking genes
        """
        super(scEncoder, self).__init__()
        
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
        self.gradient_checkpointing = gradient_checkpointing
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

        # ==================== EMBEDDING LAYERS ====================
        # Gene ID embeddings
        self.gene_embedding = nn.Embedding(gene_vocab_size, d_model)
        
        # Gene expression value embeddings (binned values only)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        
        # Positional embeddings for sequence positions
        self.position_embedding = nn.Embedding(seq_len, d_model)
        
        # Metadata embeddings (only created if vocab sizes are provided)
        # These will be broadcast and summed with gene embeddings
        self.cell_embedding = nn.Embedding(cell_type_vocab_size, d_model) if cell_type_vocab_size else None
        self.disease_embedding = nn.Embedding(disease_vocab_size, d_model) if disease_vocab_size else None
        self.tissue_embedding = nn.Embedding(tissue_vocab_size, d_model) if tissue_vocab_size else None

        # Special mask token embedding for masked positions
        self.mask_token_embedding = nn.Parameter(torch.randn(1, d_model))

        # ==================== TRANSFORMER ARCHITECTURE ====================
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=batch_first,
            activation='gelu'
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )        

        # ==================== OUTPUT PREDICTION HEADS ====================
        # Token-level prediction heads (for each position in sequence)
        self.gene_expr_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_bins)
        )
        
        self.gene_id_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, gene_vocab_size)
        )
        
        # Global prediction heads (for entire sequence metadata)
        # These use a separate pooling mechanism
        if cell_type_vocab_size:
            self.cell_type_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim // 4),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 4),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, cell_type_vocab_size)
            )
        else:
            self.cell_type_head = None
            
        if disease_vocab_size:
            self.disease_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim // 4),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 4),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, disease_vocab_size)
            )
        else:
            self.disease_head = None
            
        if tissue_vocab_size:
            self.tissue_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim // 4),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 4),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, tissue_vocab_size)
            )
        else:
            self.tissue_head = None

        # ==================== NORMALIZATION AND REGULARIZATION ====================
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize all model weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize mask token with small random values
        nn.init.normal_(self.mask_token_embedding, mean=0, std=0.02)

    def create_gene_mask(self, batch_size, seq_len, device):
        """
        Create random gene mask for MLM training.
        
        Args:
            batch_size (int): Batch size
            seq_len (int): Sequence length
            device: Device to create mask on
            
        Returns:
            torch.Tensor: Boolean mask of shape (batch_size, seq_len)
                         True for positions to be masked
        """
        mask = torch.rand(batch_size, seq_len, device=device) < self.mask_prob
        
        # Ensure at least one position is masked per sample
        for i in range(batch_size):
            if not mask[i].any():
                # Randomly select one position to mask
                pos = random.randint(0, seq_len - 1)
                mask[i, pos] = True
                
        return mask

    def forward(self, x, create_mask=True):
        """
        Forward pass through the improved scEncoder model.
        
        Args:
            x (dict): Input data dictionary containing:
                - 'gene_ids': Tensor of shape (batch_size, seq_len) with gene IDs
                - 'gene_expr': Tensor of shape (batch_size, seq_len) with BINNED gene expression values
                - 'cell_type' (optional): Tensor of shape (batch_size,) with cell type IDs
                - 'disease' (optional): Tensor of shape (batch_size,) with disease IDs
                - 'tissue' (optional): Tensor of shape (batch_size,) with tissue IDs
            create_mask (bool): Whether to create random mask for MLM training
            
        Returns:
            dict: Dictionary containing:
                - 'gene_expr': Gene expression predictions (batch_size, seq_len, num_bins)
                - 'gene_ids': Gene identity predictions (batch_size, seq_len, gene_vocab_size)
                - 'cell_type': Cell type predictions (batch_size, cell_type_vocab_size) or None
                - 'disease': Disease predictions (batch_size, disease_vocab_size) or None
                - 'tissue': Tissue predictions (batch_size, tissue_vocab_size) or None
                - 'mask': Boolean mask used for MLM (batch_size, seq_len) or None
                - 'hidden_states': Final hidden states (batch_size, seq_len, d_model)
        """
        batch_size, gene_seq_len = x['gene_expr'].size()
        device = x['gene_expr'].device

        # ==================== INPUT VALIDATION ====================
        self._validate_input(x, batch_size, gene_seq_len)

        # Ensure sequence length doesn't exceed model capacity
        if gene_seq_len > self.seq_len:
            warnings.warn(f"Input sequence length {gene_seq_len} exceeds model capacity {self.seq_len}. Truncating.")
            gene_seq_len = self.seq_len
            for key in ['gene_ids', 'gene_expr']:
                x[key] = x[key][:, :gene_seq_len]

        # ==================== GENE MASKING ====================
        gene_mask = None
        if create_mask and self.training:
            gene_mask = self.create_gene_mask(batch_size, gene_seq_len, device)
            
            # Apply masking - replace masked positions with mask token
            gene_ids_input = x['gene_ids'].clone()
            gene_expr_input = x['gene_expr'].clone()
            
            # Set masked positions to mask token ID
            gene_ids_input[gene_mask] = self.mask_token_id
            gene_expr_input[gene_mask] = self.mask_token_id
        else:
            gene_ids_input = x['gene_ids']
            gene_expr_input = x['gene_expr']

        # ==================== EMBEDDING COMPUTATION ====================
        # Compute base embeddings
        gene_emb = self.gene_embedding(gene_ids_input)  # (batch, seq_len, d_model)
        value_emb = self.value_embedding(gene_expr_input)  # (batch, seq_len, d_model)
        
        # Add positional embeddings
        positions = torch.arange(gene_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # (batch, seq_len, d_model)
        
        # Sum gene, value, and positional embeddings
        sequence_emb = gene_emb + value_emb + pos_emb
        
        # Handle masked positions with special mask token embedding
        if gene_mask is not None:
            mask_emb = self.mask_token_embedding.expand(batch_size, gene_seq_len, -1)
            sequence_emb = torch.where(gene_mask.unsqueeze(-1), mask_emb, sequence_emb)

        # ==================== METADATA EMBEDDING SUMMATION ====================
        # Sum metadata embeddings directly into the sequence embeddings
        # This approach broadcasts metadata across all positions
        
        if self.cell_embedding and 'cell_type' in x:
            cell_emb = self.cell_embedding(x['cell_type'])  # (batch, d_model)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_seq_len, -1)  # (batch, seq_len, d_model)
            sequence_emb = sequence_emb + cell_emb
            
        if self.disease_embedding and 'disease' in x:
            disease_emb = self.disease_embedding(x['disease'])  # (batch, d_model)
            disease_emb = disease_emb.unsqueeze(1).expand(-1, gene_seq_len, -1)  # (batch, seq_len, d_model)
            sequence_emb = sequence_emb + disease_emb
            
        if self.tissue_embedding and 'tissue' in x:
            tissue_emb = self.tissue_embedding(x['tissue'])  # (batch, d_model)
            tissue_emb = tissue_emb.unsqueeze(1).expand(-1, gene_seq_len, -1)  # (batch, seq_len, d_model)
            sequence_emb = sequence_emb + tissue_emb

        # ==================== SEQUENCE PROCESSING ====================
        # Apply input normalization and dropout
        sequence_emb = self.input_norm(sequence_emb)
        sequence_emb = self.dropout_layer(sequence_emb)
        
        # Handle padding if needed
        if gene_seq_len < self.seq_len:
            pad_size = self.seq_len - gene_seq_len
            sequence_emb = F.pad(sequence_emb, (0, 0, 0, pad_size), value=0)
            
        # Create attention mask
        attention_mask = torch.ones(batch_size, self.seq_len, device=device, dtype=torch.bool)
        if gene_seq_len < self.seq_len:
            attention_mask[:, gene_seq_len:] = False

        # ==================== TRANSFORMER FORWARD PASS ====================
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            hidden_states = sequence_emb
            for layer in self.transformer_encoder.layers:
                hidden_states = checkpoint(
                    create_custom_forward(layer), 
                    hidden_states, 
                    None, 
                    ~attention_mask  # Invert mask for padding positions
                )
        else:
            hidden_states = self.transformer_encoder(
                sequence_emb, 
                src_key_padding_mask=~attention_mask  # Invert mask for padding positions
            )

        # Extract only the actual sequence length (remove padding)
        hidden_states = hidden_states[:, :gene_seq_len, :] # type: ignore

        # ==================== OUTPUT GENERATION ====================
        # Token-level predictions
        gene_expr_pred = self.gene_expr_head(hidden_states)  # (batch, seq_len, num_bins)
        gene_ids_pred = self.gene_id_head(hidden_states)     # (batch, seq_len, gene_vocab_size)
        
        # Global predictions using mean pooling
        # Create pooling mask to ignore padded positions
        pooling_mask = torch.ones(batch_size, gene_seq_len, device=device)
        if gene_mask is not None:
            # Optionally exclude masked positions from pooling
            pooling_mask = pooling_mask & (~gene_mask)
        
        # Compute weighted average
        pooled_repr = (hidden_states * pooling_mask.unsqueeze(-1)).sum(dim=1) / pooling_mask.sum(dim=1, keepdim=True)
        
        # Generate metadata predictions
        cell_type_pred = self.cell_type_head(pooled_repr) if self.cell_type_head else None
        disease_pred = self.disease_head(pooled_repr) if self.disease_head else None
        tissue_pred = self.tissue_head(pooled_repr) if self.tissue_head else None
        
        outputs = {
            'gene_expr': gene_expr_pred,
            'gene_ids': gene_ids_pred,
            'cell_type': cell_type_pred,
            'disease': disease_pred,
            'tissue': tissue_pred,
            'mask': gene_mask,
            'hidden_states': hidden_states
        }
        
        return outputs

    def _validate_input(self, x, batch_size, seq_len):
        """Validate input tensor shapes and contents."""
        # Check required keys
        if 'gene_ids' not in x or 'gene_expr' not in x:
            raise ValueError("Input must contain 'gene_ids' and 'gene_expr'.")
        
        # Check batch size consistency
        if x['gene_ids'].size(0) != batch_size or x['gene_expr'].size(0) != batch_size:
            raise ValueError("Batch size mismatch in input tensors.")
        
        # Check sequence length consistency
        if x['gene_ids'].size(1) != seq_len or x['gene_expr'].size(1) != seq_len:
            raise ValueError("Sequence length mismatch in input tensors.")
        
        # Check that gene expression values are properly binned
        if torch.any(x['gene_expr'] >= self.num_bins) or torch.any(x['gene_expr'] < 0):
            raise ValueError(f"Gene expression values must be binned in range [0, {self.num_bins-1}]")
        
        # Check optional metadata
        if self.cell_embedding and 'cell_type' in x:
            if x['cell_type'].size(0) != batch_size:
                raise ValueError("Batch size mismatch in 'cell_type'.")
        if self.disease_embedding and 'disease' in x:
            if x['disease'].size(0) != batch_size:
                raise ValueError("Batch size mismatch in 'disease'.")
        if self.tissue_embedding and 'tissue' in x:
            if x['tissue'].size(0) != batch_size:
                raise ValueError("Batch size mismatch in 'tissue'.")

    def get_attention_weights(self, x, layer_idx=-1):
        """
        Extract attention weights for visualization.
        
        Args:
            x (dict): Input data dictionary
            layer_idx (int): Which transformer layer to extract weights from (-1 for last layer)
            
        Returns:
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # This would require modifying the transformer to return attention weights
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError("Attention weight extraction requires custom transformer implementation")
    
    def get_embeddings(self, x):
        """
        Get the final hidden states without prediction heads.
        
        Args:
            x (dict): Input data dictionary
            
        Returns:
            torch.Tensor: Hidden states of shape (batch_size, seq_len, d_model)
        """
        with torch.no_grad():
            outputs = self.forward(x, create_mask=False)
            return outputs['hidden_states']