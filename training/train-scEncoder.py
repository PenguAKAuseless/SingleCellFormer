import sys
import os
import glob

# Add the project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import argparse
import json
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from models.scEncoder import scEncoder
from data.SingleCellDataset import SingleCellDataset
from utils.utils import setup_logging, load_vocabulary
from models.scEncoderLoss import scEncoderLoss
from info_nce import InfoNCE
import numpy as np
import anndata
from scipy.sparse import issparse

def train(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'encoder_train_{timestamp}.log')
    setup_logging(log_file)
    
    logging.info("Loading AnnData objects from directory")
    data_files = glob.glob(os.path.join(args.data_dir, "*.h5ad"))
    if not data_files:
        logging.error("No .h5ad files found in the specified directory")
        return
    
    # Load vocabularies
    gene_vocab = load_vocabulary(args.gene_vocab_path)
    cell_type_vocab = load_vocabulary(args.cell_type_vocab_path)
    
    if gene_vocab is None:
        logging.error("Gene vocabulary is required")
        return
    
    # Initialize datasets
    datasets = []
    for data_file in data_files:
        logging.info(f"Processing file: {data_file}")
        adata = anndata.read_h5ad(data_file)
        dataset = SingleCellDataset(
            adata, 
            gene_vocab, 
            cell_type_vocab, 
            num_bins=args.num_bins, 
            seq_len=args.seq_len
        )
        datasets.append(dataset)
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    logging.info(f"Combined {len(datasets)} datasets with total {len(combined_dataset)} samples")
    
    # Initialize model
    model = scEncoder(
        gene_vocab_size=len(gene_vocab),
        cell_vocab_size=len(cell_type_vocab) if cell_type_vocab else 1,
        num_bins=args.num_bins,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device,
        gradient_checkpointing=args.gradient_checkpointing
    ).to(args.device)

    # Define loss functions
    mlm_loss_fn = nn.CrossEntropyLoss()
    contrastive_loss_fn = InfoNCE()
    loss_fn = scEncoderLoss(mlm_loss_fn, contrastive_loss_fn)
    
    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
        logging.info(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # Setup optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler() if torch.cuda.is_available() and args.device.startswith('cuda') else None
    
    # Dynamic batch size adjustment
    batch_size = args.batch_size
    min_batch_size = 1
    accumulation_steps = args.accumulation_steps
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        dataloader = DataLoader(
            combined_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True if args.device.startswith('cuda') else False,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else 2
        )
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, data in enumerate(dataloader):
            try:
                # Prepare inputs (non_blocking for async GPU transfer)
                gene_ids = data['gene_ids'].to(args.device, non_blocking=True)
                gene_expr = data['gene_expr'].to(args.device, non_blocking=True)
                cell_type = data['cell_type'].to(args.device, non_blocking=True)
                
                # Create mask for MLM (directly on GPU)
                mask = torch.rand(gene_ids.shape, device=args.device) < args.mlm_prob
                mask = mask & (gene_expr > 0)  # Only mask non-zero expressions
                
                # Ensure CLS token (position 0) is never masked
                mask[:, 0] = False
                
                # Forward pass with mixed precision
                with autocast(enabled=torch.cuda.is_available() and args.device.startswith('cuda')):
                    mlm_gene_id_out, mlm_gene_expr_out, contrastive_out, attn_mask = model(
                        gene_ids, gene_expr, cell_type
                    )
                    
                    # Calculate MLM losses (mask automatically excludes CLS position)
                    mlm_gene_id_loss = loss_fn.calculate_mlm_loss(
                        mlm_gene_id_out[mask], gene_ids[mask]
                    )
                    mlm_gene_expr_loss = loss_fn.calculate_mlm_loss(
                        mlm_gene_expr_out[mask], gene_expr[mask].long()
                    )
                    mlm_loss = args.mlm_weight * (mlm_gene_id_loss + mlm_gene_expr_loss)
                    
                    # Calculate contrastive loss
                    cell_emb = model.cell_embedding(cell_type)
                    contrastive_loss = loss_fn.calculate_contrastive_loss(
                        contrastive_out, cell_emb
                    )
                    contrastive_loss = args.contrastive_weight * contrastive_loss
                    
                    # Total loss
                    loss = mlm_loss + contrastive_loss
                
                # Backward pass
                if torch.cuda.is_available() and args.device.startswith('cuda') and scaler:
                    scaler.scale(loss / accumulation_steps).backward()
                else:
                    (loss / accumulation_steps).backward()
                
                # Perform optimization step after accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    if torch.cuda.is_available() and args.device.startswith('cuda') and scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if batch_idx % args.log_interval == 0:
                    logging.info(
                        f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, "
                        f"Batch Size: {batch_size}, Total Loss: {loss.item():.4f}, "
                        f"MLM Gene ID Loss: {mlm_gene_id_loss.item():.4f}, "
                        f"MLM Gene Expr Loss: {mlm_gene_expr_loss.item():.4f}, "
                        f"Contrastive Loss: {contrastive_loss.item():.4f}"
                    )
                
                # Clear unused memory periodically
                if args.device.startswith('cuda') and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch_size > min_batch_size:
                    logging.warning(f"Out of memory error with batch size {batch_size}. Reducing to {batch_size // 2}.")
                    batch_size = max(min_batch_size, batch_size // 2)
                    dataloader = DataLoader(
                        combined_dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=args.num_workers,
                        pin_memory=True if args.device.startswith('cuda') else False,
                        persistent_workers=True if args.num_workers > 0 else False,
                        prefetch_factor=2 if args.num_workers > 0 else 2
                    )
                    optimizer.zero_grad()
                    if args.device.startswith('cuda'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    logging.error(f"Error during training: {e}")
                    raise e
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'encoder_checkpoint_epoch_{epoch+1}_{timestamp}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {total_loss/len(dataloader):.4f}, Final Batch Size: {batch_size}")
    
    # Save final model if path is provided
    if args.final_model_path:
        os.makedirs(os.path.dirname(args.final_model_path) or '.', exist_ok=True)
        torch.save(model.state_dict(), args.final_model_path)
        logging.info(f"Saved final model to {args.final_model_path}")
    else:
        logging.info("Final model saving skipped as no path was provided")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train scEncoder model')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing AnnData files (.h5ad)')
    parser.add_argument('--gene-vocab-path', type=str, required=True, help='Path to gene vocabulary JSON')
    parser.add_argument('--cell-type-vocab-path', type=str, default=None, help='Path to cell type vocabulary JSON')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--batch-size', type=int, default=32, help='Initial batch size')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-bins', type=int, default=51, help='Number of bins for gene expression')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to model checkpoint to resume training')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--final-model-path', type=str, default=None, help='Path to save the final model')
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N batches')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--mlm-prob', type=float, default=0.15, help='Probability of masking tokens for MLM')
    parser.add_argument('--mlm-weight', type=float, default=1.0, help='Weight for MLM loss')
    parser.add_argument('--contrastive-weight', type=float, default=0.1, help='Weight for contrastive loss')
    
    args = parser.parse_args()
    train(args)