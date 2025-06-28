
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch_size, n_heads, seq_len, d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class scSetFormer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, training=True):
        # Source encoding
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        src_embedded = self.dropout(src_embedded)
        
        # Create source mask if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        
        # Pass through transformer blocks
        encoder_output = src_embedded
        attention_weights = []
        
        for transformer_block in self.transformer_blocks:
            encoder_output, attn_weights = transformer_block(encoder_output, src_mask)
            attention_weights.append(attn_weights)
        
        encoder_output = self.norm(encoder_output)
        
        # If target is provided (training mode), compute output logits
        if tgt is not None:
            # Target embedding
            tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
            tgt_embedded = self.dropout(tgt_embedded)
            
            # Create target mask if not provided
            if tgt_mask is None:
                tgt_padding_mask = self.create_padding_mask(tgt)
                tgt_look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
                tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)
            
            # Decoder (simplified - using same blocks)
            decoder_output = tgt_embedded
            for transformer_block in self.transformer_blocks:
                decoder_output, _ = transformer_block(decoder_output, tgt_mask)
            
            decoder_output = self.norm(decoder_output)
            logits = self.output_projection(decoder_output)
            
            return logits, attention_weights
        
        # Inference mode - return encoded representation
        return encoder_output, attention_weights
    
    def generate(self, src, max_len=50, start_token=1, end_token=2, temperature=1.0):
        """Generate sequence using the transformer"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        with torch.no_grad():
            encoder_output, _ = self.forward(src)
            
            # Initialize target with start token
            tgt = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
            
            for _ in range(max_len):
                # Get logits for current sequence
                logits, _ = self.forward(src, tgt)
                
                # Get next token probabilities
                next_token_logits = logits[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, 1)
                
                # Append to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check if all sequences have generated end token
                if (next_token == end_token).all():
                    break
            
            return tgt