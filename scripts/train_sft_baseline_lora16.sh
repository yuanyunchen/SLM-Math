#!/bin/bash

################################################################################
# LoRA Training Script (Rank 16)
# Trains Qwen2.5-Math-1.5B with LoRA rank 16
# Configuration matches the notebook settings

################################################################################
# Training Settings

# Model to train
MODEL="pretrained_models/Qwen2.5-Math-1.5B"

# Training data path
DATA_PATH="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"

# Training mode (lora = LoRA fine-tuning)
MODE="lora"

# LoRA rank
LORA_RANK=16

# Number of training epochs
NUM_EPOCHS=5

# Batch size (single GPU, LoRA can use larger batch)
BATCH_SIZE=32

# Gradient accumulation steps
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
# = 16 * 4 = 64
GRADIENT_ACCUMULATION_STEPS=4

# Save checkpoint every N epochs
SAVE_EVERY_N_EPOCHS=1

# GPU to use (GPU 1)
export CUDA_VISIBLE_DEVICES=1
GPUS="1"

# Enable Weights & Biases logging
USE_WANDB=""  # Add --use_wandb flag if you want to enable wandb

################################################################################
# Run training

echo "=========================================="
echo "Starting LoRA Training (Rank $LORA_RANK)"
echo "=========================================="
echo "Model: $MODEL"
echo "Data: $DATA_PATH"
echo "Mode: $MODE"
echo "LoRA Rank: $LORA_RANK"
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
    --lora_rank "$LORA_RANK" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    $USE_WANDB

echo "=========================================="
echo "Training completed!"
echo "Check checkpoints/ directory for saved LoRA adapters"
echo "Check logs/ directory for:"
echo "  - training.log: Full training logs"
echo "  - training_metrics.csv: Training metrics in CSV format"
echo "  - metrics_summary.txt: Summary of training metrics"
echo "=========================================="

