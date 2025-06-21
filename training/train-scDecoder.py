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
from models.scDecoder import scDecoder
from data.SingleCellDataset import SingleCellDataset
from utils.utils import setup_logging, load_vocabulary
from models.scDecoderLoss import scdecoder_loss
import numpy as np
import anndata
from scipy.sparse import issparse

def train(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'train_{timestamp}.log')
    setup_logging(log_file)
    
    logging.info("Loading AnnData objects from directory")
    data_files = glob.glob(os.path.join(args.data_dir, "*.h5ad"))
    if not data_files:
        logging.error("No .h5ad files found in the specified directory")
        return
    
    # Load vocabularies
    gene_vocab = load_vocabulary(args.gene_vocab_path)
    cell_type_vocab = load_vocabulary(args.cell_type_vocab_path)
    disease_vocab = load_vocabulary(args.disease_vocab_path)
    tissue_vocab = load_vocabulary(args.tissue_vocab_path)
    
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
            disease_vocab, 
            tissue_vocab, 
            num_bins=args.num_bins, 
            seq_len=args.seq_len
        )
        datasets.append(dataset)
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    logging.info(f"Combined {len(datasets)} datasets with total {len(combined_dataset)} samples")
    
    # Initialize model with gradient checkpointing support
    model = scDecoder(
        gene_vocab_size=len(gene_vocab),
        cell_type_vocab_size=len(cell_type_vocab) if cell_type_vocab else None,
        disease_vocab_size=len(disease_vocab) if disease_vocab else None,
        tissue_vocab_size=len(tissue_vocab) if tissue_vocab else None,
        num_bins=args.num_bins,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        mask_prob=args.mask_prob,
        gradient_checkpointing=args.gradient_checkpointing  # Added gradient checkpointing parameter
    ).to(args.device)
    
    # Log gradient checkpointing status
    if args.gradient_checkpointing:
        logging.info("Gradient checkpointing enabled - trading computation for memory")
    else:
        logging.info("Gradient checkpointing disabled - using standard forward pass")
    
    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path))
        logging.info(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # Setup optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler() if args.device == 'cuda' else None
    
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
            pin_memory=True if args.device == 'cuda' else False
        )
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, data in enumerate(dataloader):
            try:
                # Prepare input dictionary
                inputs = {
                    'gene_ids': data['gene_ids'].to(args.device, non_blocking=True),
                    'gene_expr': data['gene_expr'].to(args.device, non_blocking=True),
                    'cell_type': data['cell_type'].to(args.device, non_blocking=True),
                    'disease': data['disease'].to(args.device, non_blocking=True),
                    'tissue': data['tissue'].to(args.device, non_blocking=True)
                }
                
                # Forward pass with mixed precision
                with autocast(enabled=args.device == 'cuda'):
                    outputs = model(
                        inputs,
                        cell_type=inputs['cell_type'],
                        disease=inputs['disease'],
                        tissue=inputs['tissue'],
                        create_mask=True
                    )
                    loss_dict = scdecoder_loss(
                        outputs=outputs,
                        targets=inputs,
                        model=model,
                        args=args,
                        device=args.device
                    )
                    loss = loss_dict['total_loss']
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Backward pass
                if args.device == 'cuda' and scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Perform optimization step after accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    if args.device == 'cuda' and scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                
                if batch_idx % args.log_interval == 0:
                    # Enhanced logging to include gradient checkpointing status
                    gc_status = "GC" if args.gradient_checkpointing else "NoGC"
                    logging.info(
                        f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Batch Size: {batch_size}, {gc_status}, "
                        f"Total Loss: {loss.item() * accumulation_steps:.4f}, "
                        f"Masked Expr: {loss_dict['masked_expr_loss']:.4f}, "
                        f"Masked Gene: {loss_dict['masked_gene_loss']:.4f}, "
                        f"Cell: {loss_dict['cell_loss']:.4f}, "
                        f"Disease: {loss_dict['disease_loss']:.4f}, "
                        f"Tissue: {loss_dict['tissue_loss']:.4f}, "
                        f"L2: {loss_dict['l2_loss']:.4f}"
                    )
                
                # Clear unused memory
                if args.device == 'cuda':
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
                        pin_memory=True if args.device == 'cuda' else False
                    )
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                else:
                    logging.error(f"Error during training: {e}")
                    raise e
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}_{timestamp}.pth')
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
    parser = argparse.ArgumentParser(description='Train scDecoder model')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing AnnData files (.h5ad)')
    parser.add_argument('--gene-vocab-path', type=str, required=True, help='Path to gene vocabulary JSON')
    parser.add_argument('--cell-type-vocab-path', type=str, default=None, help='Path to cell type vocabulary JSON')
    parser.add_argument('--disease-vocab-path', type=str, default=None, help='Path to disease vocabulary JSON')
    parser.add_argument('--tissue-vocab-path', type=str, default=None, help='Path to tissue vocabulary JSON')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--batch-size', type=int, default=32, help='Initial batch size')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-bins', type=int, default=51, help='Number of bins for gene expression')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--hidden-dim', type=int, default=2048, help='Hidden dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--mask-prob', type=float, default=0.15, help='Probability of masking tokens for training')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to model checkpoint to resume training')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--final-model-path', type=str, default=None, help='Path to save the final model')
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N batches')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')  # Added gradient checkpointing argument
    parser.add_argument('--expr-weight', type=float, default=1.0, help='Weight for masked expression loss')
    parser.add_argument('--gene-weight', type=float, default=1.0, help='Weight for masked gene ID loss')
    parser.add_argument('--cell-weight', type=float, default=1.0, help='Weight for cell type classification loss')
    parser.add_argument('--disease-weight', type=float, default=1.0, help='Weight for disease classification loss')
    parser.add_argument('--tissue-weight', type=float, default=1.0, help='Weight for tissue classification loss')
    parser.add_argument('--l2-weight', type=float, default=0.0, help='Weight for L2 regularization')
    
    args = parser.parse_args()
    train(args)