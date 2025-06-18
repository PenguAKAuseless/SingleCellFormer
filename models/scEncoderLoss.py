import torch
import torch.nn as nn
import torch.nn.functional as F

def scencoder_loss(outputs, targets, mask, model, args, device):
    """
    Compute the combined loss for the scEncoder model.
    
    This function implements a multi-task loss for single-cell RNA sequencing data modeling,
    combining several loss components:
    1. Masked Language Modeling (MLM) loss for binned gene expression prediction
    2. Continuous value regression loss for exact expression values
    3. Gene ID classification loss
    4. Cell type classification loss
    5. Disease classification loss (optional)
    6. Tissue classification loss (optional)
    7. Contrastive loss for embedding quality (optional)
    8. L2 regularization (optional)

    Args:
        outputs (dict): Model outputs containing:
            - 'gene_expr': Predicted binned gene expression (batch_size, seq_len, num_bins)
            - 'value': Predicted continuous gene expression values (batch_size, seq_len, 1)
            - 'gene_ids': Predicted gene IDs (batch_size, seq_len, gene_vocab_size)
            - 'cell_type': Predicted cell type IDs (batch_size, cell_type_vocab_size) or None
            - 'disease': Predicted disease IDs (batch_size, disease_vocab_size) or None
            - 'tissue': Predicted tissue IDs (batch_size, tissue_vocab_size) or None
        targets (dict): Target tensors containing:
            - 'gene_ids': Ground truth gene IDs (batch_size, seq_len)
            - 'gene_expr': Ground truth binned gene expression (batch_size, seq_len)
            - 'expr_cont': Ground truth continuous gene expression (batch_size, seq_len)
            - 'cell_type': Ground truth cell type IDs (batch_size) or None
            - 'disease': Ground truth disease IDs (batch_size) or None
            - 'tissue': Ground truth tissue IDs (batch_size) or None
        mask (torch.Tensor): Boolean mask for MLM (batch_size, seq_len), True for masked positions
        model (nn.Module): The scEncoder model instance
        args (argparse.Namespace): Training arguments with weights for loss components
        device (torch.device): Device to perform computations on

    Returns:
        dict: Dictionary containing total loss and individual loss components
    """

    # Helper function to safely extract scalar value
    def safe_item(value):
        return value.item() if isinstance(value, torch.Tensor) else value
    
    # ============================
    # Initialize Loss Tracking
    # ============================
    total_loss = 0.0
    loss_dict = {
        'mlm_loss': 0.0,          # Masked language modeling loss
        'cont_loss': 0.0,         # Continuous value regression loss
        'gene_loss': 0.0,         # Gene ID classification loss
        'cell_loss': 0.0,         # Cell type classification loss
        'disease_loss': 0.0,      # Disease classification loss
        'tissue_loss': 0.0,       # Tissue classification loss
        'contrastive_loss': 0.0,  # Contrastive embedding loss
        'l2_loss': 0.0            # L2 regularization loss
    }

    # ============================
    # 1. Masked Language Modeling (MLM) Loss
    # ============================
    # This loss trains the model to predict binned gene expression values
    # at masked positions, similar to BERT's MLM objective
    if outputs['gene_expr'] is not None and targets['gene_expr'] is not None:
        # Create effective mask: only consider masked positions with non-zero expression
        # This prevents the model from learning to predict zeros for unexpressed genes
        effective_mask = mask & (targets['gene_expr'] > 0)
        
        if effective_mask.sum() > 0:  # Only compute loss if there are valid masked positions
            # Reshape predictions and targets for cross-entropy loss
            mlm_logits = outputs['gene_expr'].reshape(-1, outputs['gene_expr'].size(-1))
            mlm_targets = targets['gene_expr'].reshape(-1)
            
            # Compute cross-entropy loss only for effective masked positions
            mlm_loss = F.cross_entropy(
                mlm_logits[effective_mask.reshape(-1)],
                mlm_targets[effective_mask.reshape(-1)],
                reduction='mean'
            )
            
            # Add weighted MLM loss to total
            total_loss += args.mlm_weight * mlm_loss
            loss_dict['mlm_loss'] = safe_item(mlm_loss)

    # ============================
    # 2. Continuous Value Regression Loss
    # ============================
    # This loss trains the model to predict exact continuous gene expression values
    # in addition to the binned predictions
    if outputs['value'] is not None and targets['expr_cont'] is not None:
        # Create mask for non-zero expressions to avoid learning from dropout zeros
        non_zero_mask = (targets['expr_cont'] > 0).float()
        
        # Compute MSE loss for continuous values
        cont_loss = F.mse_loss(
            outputs['value'].squeeze(-1),  # Remove last dimension if present
            targets['expr_cont'],
            reduction='none'  # Don't reduce yet, we need to apply the mask
        )
        
        # Apply mask and compute mean only over non-zero positions
        cont_loss = (cont_loss * non_zero_mask).sum() / (non_zero_mask.sum() + 1e-8)
        
        # Add weighted continuous loss to total
        total_loss += args.cont_weight * cont_loss
        loss_dict['cont_loss'] = safe_item(cont_loss)

    # ============================
    # 3. Gene ID Classification Loss
    # ============================
    # This loss trains the model to correctly identify which genes are present
    # at each position in the sequence
    if outputs['gene_ids'] is not None and targets['gene_ids'] is not None:
        gene_loss = F.cross_entropy(
            outputs['gene_ids'].reshape(-1, outputs['gene_ids'].size(-1)),
            targets['gene_ids'].reshape(-1),
            reduction='mean'
        )
        
        # Add weighted gene ID loss to total
        total_loss += args.gene_weight * gene_loss
        loss_dict['gene_loss'] = safe_item(gene_loss)

    # ============================
    # 4. Cell Type Classification Loss
    # ============================
    # This loss trains the model to predict the cell type from gene expression patterns
    if outputs['cell_type'] is not None and targets['cell_type'] is not None:
        cell_loss = F.cross_entropy(
            outputs['cell_type'],
            targets['cell_type'],
            reduction='mean'
        )
        
        # Add weighted cell type loss to total
        total_loss += args.cell_weight * cell_loss
        loss_dict['cell_loss'] = safe_item(cell_loss)

    # ============================
    # 5. Disease Classification Loss (Optional)
    # ============================
    # This loss trains the model to predict disease state from gene expression
    if outputs['disease'] is not None and targets['disease'] is not None:
        disease_loss = F.cross_entropy(
            outputs['disease'],
            targets['disease'],
            reduction='mean'
        )
        
        # Add weighted disease loss to total
        total_loss += args.disease_weight * disease_loss
        loss_dict['disease_loss'] = safe_item(disease_loss)

    # ============================
    # 6. Tissue Classification Loss (Optional)
    # ============================
    # This loss trains the model to predict tissue type from gene expression
    if outputs['tissue'] is not None and targets['tissue'] is not None:
        tissue_loss = F.cross_entropy(
            outputs['tissue'],
            targets['tissue'],
            reduction='mean'
        )
        
        # Add weighted tissue loss to total
        total_loss += args.tissue_weight * tissue_loss
        loss_dict['tissue_loss'] = safe_item(tissue_loss)

    # ============================
    # 7. Contrastive Loss for Embedding Quality (InfoNCE-style)
    # ============================
    # This loss encourages similar cell types to have similar embeddings
    # and different cell types to have dissimilar embeddings using InfoNCE
    if args.contrastive_weight > 0:
        # Extract or compute CLS token embeddings (representation of entire cell)
        cls_embeddings = outputs.get('cls_embedding', None)
        
        if cls_embeddings is None:
            # If CLS embedding not provided, compute it from the model
            cls_embeddings = model.norm(model.transformer_encoder(
                model.dropout_layer(
                    torch.cat(
                        (model.cls_token.expand(targets['gene_ids'].size(0), -1, -1),
                        model.gene_embedding(targets['gene_ids']) + model.value_embedding(targets['gene_expr'])),
                        dim=1
                    )
                )
            ))[:, 0, :]  # Extract CLS token (first position)
        
        # Compute InfoNCE loss using cell type labels as similarity ground truth
        if targets['cell_type'] is not None:
            batch_size = cls_embeddings.size(0)
            temperature = getattr(args, 'temperature', 0.07)  # Temperature parameter for scaling logits
            
            # Compute similarity matrix (cosine similarity)
            cls_embeddings = F.normalize(cls_embeddings, dim=-1)  # Normalize embeddings for cosine similarity
            similarity_matrix = torch.matmul(cls_embeddings, cls_embeddings.t()) / temperature
            
            # Create positive pair mask: 1 for same cell type, 0 otherwise
            positive_mask = (targets['cell_type'].unsqueeze(1) == targets['cell_type'].unsqueeze(0)).float()
            positive_mask = positive_mask - torch.eye(batch_size, device=positive_mask.device)  # Remove self-pairs
            
            # InfoNCE loss: log-sum-exp over negative samples for each anchor
            contrastive_loss = 0.0
            for i in range(batch_size):
                pos_sim = similarity_matrix[i][positive_mask[i].bool()]  # Similarities for positive pairs
                all_sim = similarity_matrix[i]  # All similarities for anchor i
                
                if pos_sim.numel() > 0:  # Ensure there are positive pairs
                    # Compute log-sum-exp of similarities, excluding self
                    log_sum_exp = torch.logsumexp(all_sim, dim=0)
                    pos_sum = torch.sum(pos_sim)
                    contrastive_loss += - (pos_sum / pos_sim.numel() - log_sum_exp)
            
            # Average loss over batch
            contrastive_loss = contrastive_loss / batch_size if batch_size > 0 else contrastive_loss
            
            # Add weighted contrastive loss to total
            total_loss += args.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = safe_item(contrastive_loss)

    # ============================
    # 8. L2 Regularization (Optional)
    # ============================
    # This loss prevents overfitting by penalizing large parameter values
    if args.l2_weight > 0:
        l2_loss = 0.0
        
        # Sum squared L2 norms of all trainable parameters
        for param in model.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, p=2) ** 2
        
        # Add weighted L2 regularization to total
        total_loss += args.l2_weight * l2_loss
        loss_dict['l2_loss'] = safe_item(l2_loss)

    # ============================
    # Return Results
    # ============================
    loss_dict['total_loss'] = total_loss
    return loss_dict