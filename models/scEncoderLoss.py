import torch
import torch.nn as nn
import torch.nn.functional as F

def scencoder_loss(outputs, targets, mask, model, args, device):
    """
    Compute the combined loss for the improved scEncoder model.
    
    This function implements a multi-task loss specifically designed for the improved
    scEncoder architecture without CLS token and with proper gene masking.
    
    Key improvements:
    1. Uses the provided mask for MLM loss calculation
    2. Removes CLS-token specific loss components
    3. Focuses on gene-level and sequence-level predictions
    4. Simplified and consistent loss computation

    Args:
        outputs (dict): Model outputs containing:
            - 'gene_expr': Predicted binned gene expression (batch_size, seq_len, num_bins)
            - 'gene_ids': Predicted gene IDs (batch_size, seq_len, gene_vocab_size)
            - 'cell_type': Predicted cell type IDs (batch_size, cell_type_vocab_size) or None
            - 'disease': Predicted disease IDs (batch_size, disease_vocab_size) or None
            - 'tissue': Predicted tissue IDs (batch_size, tissue_vocab_size) or None
            - 'mask': Boolean mask used for MLM (batch_size, seq_len) or None
        targets (dict): Target tensors containing:
            - 'gene_ids': Ground truth gene IDs (batch_size, seq_len)
            - 'gene_expr': Ground truth binned gene expression (batch_size, seq_len)
            - 'cell_type': Ground truth cell type IDs (batch_size) or None
            - 'disease': Ground truth disease IDs (batch_size) or None
            - 'tissue': Ground truth tissue IDs (batch_size) or None
        mask (torch.Tensor): Boolean mask for MLM positions (batch_size, seq_len)
        model (nn.Module): The improved scEncoder model instance
        args (argparse.Namespace): Training arguments with weights for loss components
        device (torch.device): Device to perform computations on

    Returns:
        dict: Dictionary containing total loss and individual loss components
    """

    def safe_item(value):
        """Safely extract scalar value from tensor or return value as-is."""
        return value.item() if isinstance(value, torch.Tensor) else value
    
    # ============================
    # Initialize Loss Tracking
    # ============================
    total_loss = 0.0
    loss_dict = {
        'mlm_loss': 0.0,              # Combined MLM loss for gene expression prediction
        'cont_loss': 0.0,             # Continuous expression loss (not used in this version)
        'gene_loss': 0.0,             # Gene ID prediction loss
        'cell_loss': 0.0,             # Cell type classification loss
        'disease_loss': 0.0,          # Disease classification loss
        'tissue_loss': 0.0,           # Tissue classification loss
        'contrastive_loss': 0.0,      # Contrastive loss (placeholder)
        'l2_loss': 0.0                # L2 regularization loss
    }

    # ============================
    # 1. Masked Language Modeling Loss (Gene Expression)
    # ============================
    if outputs['gene_expr'] is not None and 'gene_expr' in targets and mask is not None:
        # Only compute loss for masked positions with non-zero expression
        effective_mask = mask & (targets['gene_expr'] > 0)
        
        if effective_mask.sum() > 0:
            # Reshape for cross-entropy computation
            gene_expr_logits = outputs['gene_expr'].reshape(-1, outputs['gene_expr'].size(-1))
            gene_expr_targets = targets['gene_expr'].reshape(-1)
            
            # Compute cross-entropy loss only for effectively masked positions
            mlm_loss = F.cross_entropy(
                gene_expr_logits[effective_mask.reshape(-1)],
                gene_expr_targets[effective_mask.reshape(-1)],
                reduction='mean'
            )
            
            # Apply weighting
            weight = getattr(args, 'mlm_weight', 1.0)
            total_loss += weight * mlm_loss
            loss_dict['mlm_loss'] = safe_item(mlm_loss)

    # ============================
    # 2. Gene ID Prediction Loss
    # ============================
    if outputs['gene_ids'] is not None and 'gene_ids' in targets and mask is not None:
        if mask.sum() > 0:
            # Reshape for cross-entropy computation
            gene_ids_logits = outputs['gene_ids'].reshape(-1, outputs['gene_ids'].size(-1))
            gene_ids_targets = targets['gene_ids'].reshape(-1)
            
            # Compute cross-entropy loss only for masked positions
            gene_loss = F.cross_entropy(
                gene_ids_logits[mask.reshape(-1)],
                gene_ids_targets[mask.reshape(-1)],
                reduction='mean'
            )
            
            # Apply weighting
            weight = getattr(args, 'gene_weight', 1.0)
            total_loss += weight * gene_loss
            loss_dict['gene_loss'] = safe_item(gene_loss)

    # ============================
    # 3. Cell Type Classification Loss
    # ============================
    if outputs['cell_type'] is not None and 'cell_type' in targets:
        cell_loss = F.cross_entropy(
            outputs['cell_type'],
            targets['cell_type'],
            reduction='mean'
        )
        
        weight = getattr(args, 'cell_weight', 1.0)
        total_loss += weight * cell_loss
        loss_dict['cell_loss'] = safe_item(cell_loss)

    # ============================
    # 4. Disease Classification Loss
    # ============================
    if outputs['disease'] is not None and 'disease' in targets:
        disease_loss = F.cross_entropy(
            outputs['disease'],
            targets['disease'],
            reduction='mean'
        )
        
        weight = getattr(args, 'disease_weight', 1.0)
        total_loss += weight * disease_loss
        loss_dict['disease_loss'] = safe_item(disease_loss)

    # ============================
    # 5. Tissue Classification Loss
    # ============================
    if outputs['tissue'] is not None and 'tissue' in targets:
        tissue_loss = F.cross_entropy(
            outputs['tissue'],
            targets['tissue'],
            reduction='mean'
        )
        
        weight = getattr(args, 'tissue_weight', 1.0)
        total_loss += weight * tissue_loss
        loss_dict['tissue_loss'] = safe_item(tissue_loss)

    # ============================
    # 6. Contrastive Loss (Placeholder)
    # ============================
    # Implement contrastive learning if needed
    contrastive_weight = getattr(args, 'contrastive_weight', 0.0)
    if contrastive_weight > 0:
        # This is a placeholder - implement actual contrastive loss if needed
        # For now, we'll set it to 0
        loss_dict['contrastive_loss'] = 0.0

    # ============================
    # 7. L2 Regularization
    # ============================
    l2_weight = getattr(args, 'l2_weight', 0.0)
    if l2_weight > 0:
        l2_loss = 0.0
        for param in model.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, p=2) ** 2
        
        total_loss += l2_weight * l2_loss
        loss_dict['l2_loss'] = safe_item(l2_loss)

    # ============================
    # Return Results
    # ============================
    loss_dict['total_loss'] = safe_item(total_loss)
    return loss_dict


def compute_mlm_accuracy(outputs, targets, mask):
    """
    Compute accuracy metrics for masked language modeling.
    
    Args:
        outputs (dict): Model outputs
        targets (dict): Ground truth targets
        mask (torch.Tensor): Boolean mask for MLM positions
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    metrics = {}
    
    if mask is not None and mask.sum() > 0:
        # Gene expression accuracy
        if outputs['gene_expr'] is not None and 'gene_expr' in targets:
            gene_expr_pred = torch.argmax(outputs['gene_expr'], dim=-1)
            correct_expr = (gene_expr_pred[mask] == targets['gene_expr'][mask]).float()
            metrics['mlm_gene_expr_accuracy'] = correct_expr.mean().item()
        
        # Gene ID accuracy
        if outputs['gene_ids'] is not None and 'gene_ids' in targets:
            gene_ids_pred = torch.argmax(outputs['gene_ids'], dim=-1)
            correct_ids = (gene_ids_pred[mask] == targets['gene_ids'][mask]).float()
            metrics['mlm_gene_ids_accuracy'] = correct_ids.mean().item()
    
    # Metadata classification accuracies
    if outputs['cell_type'] is not None and 'cell_type' in targets:
        cell_pred = torch.argmax(outputs['cell_type'], dim=-1)
        metrics['cell_type_accuracy'] = (cell_pred == targets['cell_type']).float().mean().item()
    
    if outputs['disease'] is not None and 'disease' in targets:
        disease_pred = torch.argmax(outputs['disease'], dim=-1)
        metrics['disease_accuracy'] = (disease_pred == targets['disease']).float().mean().item()
    
    if outputs['tissue'] is not None and 'tissue' in targets:
        tissue_pred = torch.argmax(outputs['tissue'], dim=-1)
        metrics['tissue_accuracy'] = (tissue_pred == targets['tissue']).float().mean().item()
    
    return metrics