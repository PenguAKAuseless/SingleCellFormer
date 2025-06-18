import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
import warnings

class scEncoder(nn.Module):
    """
    Single-cell RNA-seq Transformer Encoder Model
    
    A transformer-based model designed for single-cell RNA sequencing data analysis.
    This model can handle gene expression data along with optional metadata such as
    cell type, disease, and tissue information. It supports both binned and continuous
    gene expression values and includes various prediction heads for different tasks.
    
    The model uses a [CLS] token for global representation and supports gradient
    checkpointing for memory-efficient training on large datasets.
    """
    
    def __init__(
            self, gene_vocab_size, 
            cell_type_vocab_size=None, disease_vocab_size=None, tissue_vocab_size=None,
            batch_first=True, num_bins=51, seq_len=512,
            d_model=512, nhead=8, num_layers=12, hidden_dim=2048, dropout=0.1,
            gradient_checkpointing=False,
        ):
        """
        Initialize the scEncoder model with specified architecture parameters.

        Args:   
            gene_vocab_size (int): Total number of unique genes in the vocabulary
            cell_type_vocab_size (int, optional): Number of unique cell types. 
                If None, cell type prediction head is disabled. Default: None
            disease_vocab_size (int, optional): Number of unique diseases. 
                If None, disease prediction head is disabled. Default: None
            tissue_vocab_size (int, optional): Number of unique tissues. 
                If None, tissue prediction head is disabled. Default: None
            batch_first (bool): Whether input tensors have batch dimension first.
                Format: (batch, seq, feature) if True, (seq, batch, feature) if False. Default: True
            num_bins (int): Number of discretization bins for gene expression values.
                Used when converting continuous values to discrete tokens. Default: 51
            seq_len (int): Maximum sequence length the model can handle.
                Sequences are padded/truncated to this length. Default: 512
            d_model (int): Dimensionality of model embeddings and hidden states. Default: 512
            nhead (int): Number of attention heads in each transformer layer. Default: 8
            num_layers (int): Number of transformer encoder layers to stack. Default: 12
            hidden_dim (int): Dimensionality of feedforward network inside transformer. Default: 2048
            dropout (float): Dropout probability applied throughout the model. Default: 0.1
            gradient_checkpointing (bool): Enable gradient checkpointing to trade computation
                for memory during training. Useful for large models. Default: False
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

        # ==================== EMBEDDING LAYERS ====================
        # Gene ID embeddings: maps gene indices to dense vectors
        self.gene_embedding = nn.Embedding(gene_vocab_size, d_model)
        
        # Gene expression value embeddings: maps binned expression values to dense vectors
        self.value_embedding = nn.Embedding(num_bins, d_model)
        
        # Optional metadata embeddings (only created if vocab sizes are provided)
        self.cell_embedding = nn.Embedding(cell_type_vocab_size, d_model) if cell_type_vocab_size else None
        self.disease_embedding = nn.Embedding(disease_vocab_size, d_model) if disease_vocab_size else None
        self.tissue_embedding = nn.Embedding(tissue_vocab_size, d_model) if tissue_vocab_size else None

        # ==================== SPECIAL TOKENS ====================
        # [CLS] token: learnable parameter for global sequence representation
        # Used for classification tasks and capturing sequence-level information
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ==================== TRANSFORMER ARCHITECTURE ====================
        # Define transformer encoder layer with specified architecture
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,              # Input/output dimension
            nhead=nhead,                  # Number of attention heads
            dim_feedforward=hidden_dim,   # Feedforward network dimension
            dropout=dropout,              # Dropout rate
            batch_first=batch_first,      # Batch dimension ordering
            activation='gelu'             # Activation function (GELU works well for transformers)
        )

        # Stack multiple encoder layers with layer normalization
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),   # Final layer normalization
        )        

        # ==================== OUTPUT PREDICTION HEADS ====================
        # Masked Language Modeling head: predicts binned expression values
        self.mlm_output = nn.Linear(d_model, num_bins)
        
        # Continuous value regression head: predicts actual expression values
        self.continuous_output = nn.Linear(d_model, 1)
        
        # Gene prediction head: predicts gene identities (for tasks like gene imputation)
        self.gene_output = nn.Linear(d_model, gene_vocab_size)
        
        # Optional metadata prediction heads (only created if corresponding embeddings exist)
        self.cell_output = nn.Linear(d_model, cell_type_vocab_size) if cell_type_vocab_size else None
        self.disease_output = nn.Linear(d_model, disease_vocab_size) if disease_vocab_size else None
        self.tissue_output = nn.Linear(d_model, tissue_vocab_size) if tissue_vocab_size else None

        # ==================== NORMALIZATION AND REGULARIZATION ====================
        # Layer normalization for output stabilization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout layer for regularization
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize all model weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights using Xavier uniform initialization.
        
        This initialization helps with gradient flow and training stability.
        The [CLS] token is initialized with small random values following
        common practices in transformer models.
        """
        # Initialize all parameters with dimension > 1 using Xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize [CLS] token with small random values
        nn.init.normal_(self.cls_token, mean=0, std=0.02)

    def forward(self, x, binned_value=True):
        """
        Forward pass through the scEncoder model.
        
        This method processes single-cell RNA-seq data through the transformer
        architecture and returns predictions from multiple task-specific heads.
        
        Args:
            x (dict): Input data dictionary containing:
                - 'gene_ids': Tensor of shape (batch_size, seq_len) with gene IDs
                - 'gene_expr': Tensor of shape (batch_size, seq_len) with gene expression values
                - 'cell_type' (optional): Tensor of shape (batch_size,) with cell type IDs
                - 'disease' (optional): Tensor of shape (batch_size,) with disease IDs
                - 'tissue' (optional): Tensor of shape (batch_size,) with tissue IDs
            binned_value (bool): Whether gene expression values are already binned.
                If False, continuous values will be log-transformed and binned. Default: True
            
        Returns:
            dict: Dictionary containing predictions from all heads:
                - 'gene_expr': Binned expression value predictions (batch_size, seq_len, num_bins)
                - 'value': Continuous expression value predictions (batch_size, seq_len, 1)
                - 'gene_ids': Gene identity predictions (batch_size, seq_len, gene_vocab_size)
                - 'cell_type': Cell type predictions (batch_size, cell_type_vocab_size) or None
                - 'disease': Disease predictions (batch_size, disease_vocab_size) or None
                - 'tissue': Tissue predictions (batch_size, tissue_vocab_size) or None
        """
        # Extract batch and sequence dimensions
        batch_size, seq_len = x['gene_expr'].size()
        device = x['gene_expr'].device

        # ==================== INPUT VALIDATION ====================
        def validate_input(x, has_cell_type=False, has_disease=False, has_tissue=False):
            """
            Validate input tensor shapes and contents.
            
            Ensures all required keys are present and tensor dimensions are consistent.
            This helps catch data preprocessing errors early in the pipeline.
            """
            # Check required keys
            if 'gene_ids' not in x or 'gene_expr' not in x:
                raise ValueError("Input must contain 'gene_ids' and 'gene_expr'.")
            
            # Check batch size consistency
            if x['gene_ids'].size(0) != batch_size or x['gene_expr'].size(0) != batch_size:
                raise ValueError("Batch size mismatch in input tensors.")
            
            # Check sequence length consistency
            if x['gene_ids'].size(1) != seq_len or x['gene_expr'].size(1) != seq_len:
                raise ValueError("Sequence length mismatch in input tensors.")
            
            # Check optional metadata keys and dimensions
            if has_cell_type and 'cell_type' not in x:
                raise ValueError("Input must contain 'cell_type' when has_cell_type is True.")
            if has_disease and 'disease' not in x:
                raise ValueError("Input must contain 'disease' when has_disease is True.")
            if has_tissue and 'tissue' not in x:
                raise ValueError("Input must contain 'tissue' when has_tissue is True.")
            
            # Check metadata tensor dimensions
            if has_cell_type and x['cell_type'].size(0) != batch_size:
                raise ValueError("Batch size mismatch in 'cell_type'.")
            if has_disease and x['disease'].size(0) != batch_size:
                raise ValueError("Batch size mismatch in 'disease'.")
            if has_tissue and x['tissue'].size(0) != batch_size:
                raise ValueError("Batch size mismatch in 'tissue'.")
            
        # Perform validation based on available embedding layers
        validate_input(x, 
                       has_cell_type=self.cell_embedding is not None, 
                       has_disease=self.disease_embedding is not None, 
                       has_tissue=self.tissue_embedding is not None)

        # ==================== EMBEDDING COMPUTATION ====================
        # Compute gene ID embeddings
        gene_emb = self.gene_embedding(x['gene_ids'])
        
        # Handle gene expression value embeddings
        if binned_value:
            # Values are already discretized into bins
            value_emb = self.value_embedding(x['gene_expr'])
        else:
            # Convert continuous values to binned values
            # Apply log1p transformation to handle the wide dynamic range of gene expression
            value_emb = torch.log1p(x['gene_expr'])
            
            # Normalize and bin the values per sample to handle different expression scales
            max_val = torch.max(value_emb, dim=-1, keepdim=True)[0]  # Per-sample maximum
            value_emb = torch.floor(value_emb * (self.num_bins - 1) / (max_val + 1e-6)).long()

            # Convert binned values to embeddings
            value_emb = self.value_embedding(value_emb.long())

        # Compute optional metadata embeddings
        cell_emb = self.cell_embedding(x['cell_type']) if self.cell_embedding else None
        disease_emb = self.disease_embedding(x['disease']) if self.disease_embedding else None
        tissue_emb = self.tissue_embedding(x['tissue']) if self.tissue_embedding else None
        
        # ==================== EMBEDDING COMBINATION ====================
        # Combine gene and value embeddings additively
        # This allows the model to learn relationships between gene identity and expression level
        emb = gene_emb + value_emb
        
        # ==================== [CLS] TOKEN PREPARATION ====================
        # Expand [CLS] token to match batch size
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        # Add metadata information to [CLS] token
        # This encodes sample-level information that can be used for classification tasks
        if cell_emb is not None:
            cls_token = cls_token + cell_emb.unsqueeze(1)  # Add sequence dimension
        if disease_emb is not None:
            cls_token = cls_token + disease_emb.unsqueeze(1)
        if tissue_emb is not None:
            cls_token = cls_token + tissue_emb.unsqueeze(1)
        
        # ==================== SEQUENCE CONSTRUCTION ====================
        # Concatenate [CLS] token with gene embeddings
        # Final sequence: [CLS] + gene_1 + gene_2 + ... + gene_n
        x = torch.cat((cls_token, emb), dim=1)
        seq_len += 1  # Account for [CLS] token

        # Handle sequence length variations through padding/truncation
        if seq_len < self.seq_len:
            # Pad sequence to fixed length with zeros
            pad_size = self.seq_len - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_size), value=0)
        elif x.size(1) > self.seq_len:
            # Truncate sequence if it exceeds maximum length
            warnings.warn("Input sequence length exceeds the model's expected sequence length. Truncating to fit.")
            x = x[:, :self.seq_len, :]
            
        # Apply dropout for regularization
        x = self.dropout_layer(x)

        # ==================== ATTENTION MASK CREATION ====================
        # Create attention mask to ignore padded positions
        # 1 = attend to this position, 0 = ignore this position
        attention_mask = torch.ones(batch_size, self.seq_len, device=device)
        if seq_len < self.seq_len:
            attention_mask[:, seq_len:] = 0

        # ==================== TRANSFORMER FORWARD PASS ====================
        if self.gradient_checkpointing:
            # Use gradient checkpointing to save memory during training
            # This trades computation for memory by recomputing activations during backward pass
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # Apply checkpointing to each transformer layer
            for layer in self.transformer_encoder.layers:
                x = checkpoint(create_custom_forward(layer), x, None, (attention_mask == 0))
        else:
            # Standard forward pass through all transformer layers
            x = self.transformer_encoder(x, src_key_padding_mask=(attention_mask == 0))

        # Apply final layer normalization
        x = self.norm(x)

        # ==================== OUTPUT HEAD PREDICTIONS ====================
        # Generate predictions from all available output heads
        # Skip [CLS] token (position 0) for token-level predictions
        outputs = {
            # Token-level predictions
            'gene_expr': self.mlm_output(x),      # Binned expression predictions
            'value': self.continuous_output(x),   # Continuous value predictions  
            'gene_ids': self.gene_output(x),      # Gene identity predictions
            
            # Sequence-level predictions (use [CLS] token)
            'cell_type': self.cell_output(x[:, 0, :]) if self.cell_output else None,
            'disease': self.disease_output(x[:, 0, :]) if self.disease_output else None,
            'tissue': self.tissue_output(x[:, 0, :]) if self.tissue_output else None
        }
        
        return outputs