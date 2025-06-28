import torch
import torch.nn as nn
import torch.nn.functional as F

class ScDecoder(nn.Module):
    def __init__(self, vocab_size, num_cell_types, num_bins=51, d_model=512, nhead=8, num_layers=12, dropout=0.1):
        super(ScDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_cell_types = num_cell_types
        self.num_bins = num_bins
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        self.cell_type_embedding = nn.Embedding(num_cell_types, d_model)

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, num_bins)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gene_embedding.weight)
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.xavier_uniform_(self.cell_type_embedding.weight)

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, gene_ids, binned_expr_value, cell_type, non_zero_mask=None, return_attn=False):
        batch_size, seq_len = binned_expr_value.size()
        device = binned_expr_value.device

        # Embedding inputs
        gene_emb = self.gene_embedding(gene_ids)
        value_emb = self.value_embedding(binned_expr_value)
        cell_type_emb = self.cell_type_embedding(cell_type).unsqueeze(1)

        # Concatenate embeddings
        emb = gene_emb + value_emb + cell_type_emb
    
        if non_zero_mask is not None:
            emb = emb * non_zero_mask.unsqueeze(-1)

        if return_attn:
            attn_out = []
            def hook(module, input, output):
                attn_out.append(output[1])  # Capture attention weights
            handles = [layer.register_forward_hook(hook) for layer in self.transformer_decoder.layers]
            output = self.transformer_decoder(emb, memory=None)
            for handle in handles:
                handle.remove()
            fc_out = self.fc_out(output)
            return fc_out, output, attn_out
        else:
            output = self.transformer_decoder(emb, memory=None)
            fc_out = self.fc_out(output)
            return fc_out, output