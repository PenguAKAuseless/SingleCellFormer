#!/bin/bash

# Configuration
DATA_DIR="train"
GENE_VOCAB="vocab/gene_vocab.json"
CELL_TYPE_VOCAB="vocab/celltype_vocab.json" # Optional
DISEASE_VOCAB="vocab/disease_vocab.json"    # Optional
TISSUE_VOCAB="vocab/tissue_vocab.json"      # Optional
OUTPUT_DIR="output"
LOG_DIR="logs"
CHECKPOINT_PATH=""               # Optional: path to existing checkpoint
FINAL_MODEL_PATH="final/final_model.pth"    # Set to "" to skip final model saving
BATCH_SIZE=32
ACCUMULATION_STEPS=4
EPOCHS=10
LR=0.0001
NUM_BINS=51
SEQ_LEN=512
D_MODEL=512
NHEAD=8
NUM_LAYERS=12
HIDDEN_DIM=2048
DROPOUT=0.1
CHECKPOINT_INTERVAL=5
LOG_INTERVAL=100
NUM_WORKERS=4
DEVICE="cuda:0"                              # Use "cpu" if CUDA is not available
MLM_PROB=0.15
MLM_WEIGHT=1.0
CONT_WEIGHT=1.0
GENE_WEIGHT=1.0
CELL_WEIGHT=1.0
DISEASE_WEIGHT=1.0
TISSUE_WEIGHT=1.0
CONTRASTIVE_WEIGHT=0.0
CONTRASTIVE_MARGIN=0.5
L2_WEIGHT=0.0
GRADIENT_CHECKPOINTING="--gradient-checkpointing"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Timestamp for unique log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build the command
CMD=(
  python training/train-scEncoder.py
  --data-dir "$DATA_DIR"
  --gene-vocab-path "$GENE_VOCAB"
  --cell-type-vocab-path "$CELL_TYPE_VOCAB"
  --disease-vocab-path "$DISEASE_VOCAB"
  --tissue-vocab-path "$TISSUE_VOCAB"
  --output-dir "$OUTPUT_DIR"
  --log-dir "$LOG_DIR"
  --batch-size "$BATCH_SIZE"
  --accumulation-steps "$ACCUMULATION_STEPS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --num-bins "$NUM_BINS"
  --seq-len "$SEQ_LEN"
  --d-model "$D_MODEL"
  --nhead "$NHEAD"
  --num-layers "$NUM_LAYERS"
  --hidden-dim "$HIDDEN_DIM"
  --dropout "$DROPOUT"
  --checkpoint-path "$CHECKPOINT_PATH"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --mlm-prob "$MLM_PROB"
  --mlm-weight "$MLM_WEIGHT"
  --cont-weight "$CONT_WEIGHT"
  --gene-weight "$GENE_WEIGHT"
  --cell-weight "$CELL_WEIGHT"
  --disease-weight "$DISEASE_WEIGHT"
  --tissue-weight "$TISSUE_WEIGHT"
  --contrastive-weight "$CONTRASTIVE_WEIGHT"
  --contrastive-margin "$CONTRASTIVE_MARGIN"
  --l2-weight "$L2_WEIGHT"
  "$GRADIENT_CHECKPOINTING"
)

# Add final model path if provided
if [ -n "$FINAL_MODEL_PATH" ]; then
  CMD+=(--final-model-path "$FINAL_MODEL_PATH")
fi

# Run training with nohup
nohup "${CMD[@]}" > "$LOG_DIR/train_out_$TIMESTAMP.log" 2> "$LOG_DIR/train_err_$TIMESTAMP.err" &

# Save PID
echo $! > "$LOG_DIR/train_pid_$TIMESTAMP.pid"
echo "Training started with PID $(cat "$LOG_DIR/train_pid_$TIMESTAMP.pid")"
echo "Logs: $LOG_DIR/train_out_$TIMESTAMP.log"
echo "Errors: $LOG_DIR/train_err_$TIMESTAMP.err"
echo "PID file: $LOG_DIR/train_pid_$TIMESTAMP.pid"