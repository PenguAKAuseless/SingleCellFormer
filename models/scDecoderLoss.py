import torch
import torch.nn as nn
import torch.nn.functional as F

def scdecoder_loss(outputs, targets, model, args, device):
    """
    Compute the combined loss for the scDecoder model with random masking.
    
    This function implements a multi-task loss for autoregressive single-cell RNA 
    sequencing data modeling, with emphasis on masked gene expression prediction.
    
    Loss components:
    1. Masked Gene Expression Prediction loss (primary task)
    2. Masked Gene ID Prediction loss 
    3. Cell type classification loss (optional)
    4. Disease classification loss (optional)
    5. Tissue classification loss (optional)
    6. L2 regularization (optional)

    Args:
        outputs (dict): Model outputs containing:
            - 'gene_output': Predicted gene IDs (batch_size, seq_len, gene_vocab_size)
            - 'value_output': Predicted binned gene expression (batch_size, seq_len, num_bins)
            - 'cell_output': Predicted cell type IDs (batch_size, cell_type_vocab_size) or None
            - 'disease_output': Predicted disease IDs (batch_size, disease_vocab_size) or None
            - 'tissue_output': Predicted tissue IDs (batch_size, tissue_vocab_size) or None
            - 'mask': Random mask used during forward pass (batch_size, seq_len) or None
        targets (dict): Target tensors containing:
            - 'gene_ids': Ground truth gene IDs (batch_size, seq_len)
            - 'gene_expr': Ground truth binned gene expression (batch_size, seq_len)
            - 'cell_type': Ground truth cell type IDs (batch_size) or None
            - 'disease': Ground truth disease IDs (batch_size) or None
            - 'tissue': Ground truth tissue IDs (batch_size) or None
        model (nn.Module): The scDecoder model instance
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
        'masked_expr_loss': 0.0,      # Masked gene expression prediction loss
        'masked_gene_loss': 0.0,      # Masked gene ID prediction loss
        'cell_loss': 0.0,             # Cell type classification loss
        'disease_loss': 0.0,          # Disease classification loss
        'tissue_loss': 0.0,           # Tissue classification loss
        'l2_loss': 0.0                # L2 regularization loss
    }

    # Extract mask from outputs
    mask = outputs.get('mask', None)
    
    # ============================
    # 1. Masked Gene Expression Prediction Loss
    # ============================
    # This is the primary loss for autoregressive learning
    if outputs['value_output'] is not None and targets['gene_expr'] is not None:
        if mask is not None:
            # Only compute loss for masked positions
            effective_mask = mask & (targets['gene_expr'] > 0)  # Don't learn from unexpressed genes
            
            if effective_mask.sum() > 0:  # Only compute loss if there are valid masked positions
                # Reshape predictions and targets for cross-entropy loss
                expr_logits = outputs['value_output'].reshape(-1, outputs['value_output'].size(-1))
                expr_targets = targets['gene_expr'].reshape(-1)
                
                # Compute cross-entropy loss only for masked positions
                masked_expr_loss = F.cross_entropy(
                    expr_logits[effective_mask.reshape(-1)],
                    expr_targets[effective_mask.reshape(-1)],
                    reduction='mean'
                )
                
                # Add weighted expression loss to total
                total_loss += args.expr_weight * masked_expr_loss
                loss_dict['masked_expr_loss'] = safe_item(masked_expr_loss)
        else:
            # If no mask provided, compute loss for all non-zero positions
            non_zero_mask = (targets['gene_expr'] > 0)
            if non_zero_mask.sum() > 0:
                expr_logits = outputs['value_output'].reshape(-1, outputs['value_output'].size(-1))
                expr_targets = targets['gene_expr'].reshape(-1)
                
                masked_expr_loss = F.cross_entropy(
                    expr_logits[non_zero_mask.reshape(-1)],
                    expr_targets[non_zero_mask.reshape(-1)],
                    reduction='mean'
                )
                
                total_loss += args.expr_weight * masked_expr_loss
                loss_dict['masked_expr_loss'] = safe_item(masked_expr_loss)

    # ============================
    # 2. Masked Gene ID Prediction Loss
    # ============================
    # Secondary task to help model understand gene identity
    if outputs['gene_output'] is not None and targets['gene_ids'] is not None:
        if mask is not None:
            # Only compute loss for masked positions
            if mask.sum() > 0:
                gene_logits = outputs['gene_output'].reshape(-1, outputs['gene_output'].size(-1))
                gene_targets = targets['gene_ids'].reshape(-1)
                
                masked_gene_loss = F.cross_entropy(
                    gene_logits[mask.reshape(-1)],
                    gene_targets[mask.reshape(-1)],
                    reduction='mean'
                )
                
                # Add weighted gene ID loss to total
                total_loss += args.gene_weight * masked_gene_loss
                loss_dict['masked_gene_loss'] = safe_item(masked_gene_loss)
        else:
            # If no mask provided, compute loss for all positions
            gene_loss = F.cross_entropy(
                outputs['gene_output'].reshape(-1, outputs['gene_output'].size(-1)),
                targets['gene_ids'].reshape(-1),
                reduction='mean'
            )
            
            total_loss += args.gene_weight * gene_loss
            loss_dict['masked_gene_loss'] = safe_item(gene_loss)

    # ============================
    # 3. Cell Type Classification Loss
    # ============================
    if outputs['cell_output'] is not None and targets['cell_type'] is not None:
        cell_loss = F.cross_entropy(
            outputs['cell_output'],
            targets['cell_type'],
            reduction='mean'
        )
        
        # Add weighted cell type loss to total
        total_loss += args.cell_weight * cell_loss
        loss_dict['cell_loss'] = safe_item(cell_loss)

    # ============================
    # 4. Disease Classification Loss (Optional)
    # ============================
    if outputs['disease_output'] is not None and targets['disease'] is not None:
        disease_loss = F.cross_entropy(
            outputs['disease_output'],
            targets['disease'],
            reduction='mean'
        )
        
        # Add weighted disease loss to total
        total_loss += args.disease_weight * disease_loss
        loss_dict['disease_loss'] = safe_item(disease_loss)

    # ============================
    # 5. Tissue Classification Loss (Optional)
    # ============================
    if outputs['tissue_output'] is not None and targets['tissue'] is not None:
        tissue_loss = F.cross_entropy(
            outputs['tissue_output'],
            targets['tissue'],
            reduction='mean'
        )
        
        # Add weighted tissue loss to total
        total_loss += args.tissue_weight * tissue_loss
        loss_dict['tissue_loss'] = safe_item(tissue_loss)

    # ============================
    # 6. L2 Regularization (Optional)
    # ============================
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


def compute_accuracy_metrics(outputs, targets, mask=None):
    """
    Compute accuracy metrics for the scDecoder model.
    
    Args:
        outputs (dict): Model outputs
        targets (dict): Target tensors
        mask (torch.Tensor, optional): Mask used during training
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    metrics = {}
    
    # Gene expression accuracy (for masked positions if available)
    if outputs['value_output'] is not None and targets['gene_expr'] is not None:
        pred_expr = torch.argmax(outputs['value_output'], dim=-1)
        
        if mask is not None:
            # Accuracy only for masked positions
            effective_mask = mask & (targets['gene_expr'] > 0)
            if effective_mask.sum() > 0:
                correct = (pred_expr == targets['gene_expr']) & effective_mask
                metrics['masked_expr_accuracy'] = (correct.sum().float() / effective_mask.sum().float()).item()
        else:
            # Overall accuracy for non-zero expressions
            non_zero_mask = targets['gene_expr'] > 0
            if non_zero_mask.sum() > 0:
                correct = (pred_expr == targets['gene_expr']) & non_zero_mask
                metrics['expr_accuracy'] = (correct.sum().float() / non_zero_mask.sum().float()).item()
    
    # Gene ID accuracy (for masked positions if available)
    if outputs['gene_output'] is not None and targets['gene_ids'] is not None:
        pred_genes = torch.argmax(outputs['gene_output'], dim=-1)
        
        if mask is not None and mask.sum() > 0:
            correct = (pred_genes == targets['gene_ids']) & mask
            metrics['masked_gene_accuracy'] = (correct.sum().float() / mask.sum().float()).item()
        else:
            correct = pred_genes == targets['gene_ids']
            metrics['gene_accuracy'] = correct.float().mean().item()
    
    # Cell type accuracy
    if outputs['cell_output'] is not None and targets['cell_type'] is not None:
        pred_cell = torch.argmax(outputs['cell_output'], dim=-1)
        correct = pred_cell == targets['cell_type']
        metrics['cell_accuracy'] = correct.float().mean().item()
    
    # Disease accuracy
    if outputs['disease_output'] is not None and targets['disease'] is not None:
        pred_disease = torch.argmax(outputs['disease_output'], dim=-1)
        correct = pred_disease == targets['disease']
        metrics['disease_accuracy'] = correct.float().mean().item()
    
    # Tissue accuracy
    if outputs['tissue_output'] is not None and targets['tissue'] is not None:
        pred_tissue = torch.argmax(outputs['tissue_output'], dim=-1)
        correct = pred_tissue == targets['tissue']
        metrics['tissue_accuracy'] = correct.float().mean().item()
    
    return metrics