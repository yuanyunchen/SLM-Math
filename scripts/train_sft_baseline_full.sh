#!/bin/bash

################################################################################
# Full SFT Training Script
# Trains Qwen2.5-Math-1.5B with full fine-tuning (no LoRA)
# Configuration matches the notebook settings

################################################################################
# Training Settings

# Model to train
MODEL="pretrained_models/Qwen2.5-Math-1.5B"

# Training data path
DATA_PATH="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"

# Training mode (sft = full fine-tuning)
MODE="sft"

# Number of training epochs
NUM_EPOCHS=5

# Batch size (single GPU)
BATCH_SIZE=16

# Gradient accumulation steps
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
# = 16 * 4 = 64
GRADIENT_ACCUMULATION_STEPS=4

# Save checkpoint every N epochs
SAVE_EVERY_N_EPOCHS=1

# GPU to use (GPU 0)
export CUDA_VISIBLE_DEVICES=0
GPUS="0"

# Enable Weights & Biases logging
USE_WANDB="--use_wandb"  # Add --use_wandb flag if you want to enable wandb

################################################################################
# Run training

echo "=========================================="
echo "Starting Full SFT Training"
echo "=========================================="
echo "Model: $MODEL"
echo "Data: $DATA_PATH"
echo "Mode: $MODE"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Effective batch size: $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Save checkpoint every: $SAVE_EVERY_N_EPOCHS epochs"
echo "=========================================="

python models/train_sft_baseline.py \
    --mode "$MODE" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    $USE_WANDB

echo "=========================================="
echo "Training completed!"
echo "Check checkpoints/ directory for saved models"
echo "Check logs/ directory for:"
echo "  - training.log: Full training logs"
echo "  - training_metrics.csv: Training metrics in CSV format"
echo "  - metrics_summary.txt: Summary of training metrics"
echo "=========================================="

