#!/bin/bash

################################################################################
# Plain SFT/LoRA Training (no chat template)
# Uses models/train_sft_plain.py
################################################################################

set -e

################################################################################
# Training Settings

# Round name (used in checkpoint/log directory names)
ROUND_NAME="plain_sft_full"

# Model to train
MODEL="pretrained_models/Qwen2.5-Math-1.5B"

# Training data path
DATA_PATH="data/reasoning_code_v2/run_1209_1011/reasoning_code_v2_gsm8k_train_math_train_1209_1011_code_box_eq_gt.jsonl"

# Training mode: "sft" (full fine-tuning) or "lora"
MODE="sft"

# Number of epochs
NUM_EPOCHS=3

# Batch size (per GPU)
BATCH_SIZE=16

# Gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS=4

# Learning rate (default inside script: 1e-5 for sft, 1e-4 for lora)
LEARNING_RATE=5e-5

# Save checkpoint every N epochs
SAVE_EVERY_N_EPOCHS=999

# Max sequence length
MAX_SEQ_LENGTH=2048

# GPU selection
export CUDA_VISIBLE_DEVICES=2
GPUS="2"

# Enable gradient checkpointing
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# Enable evaluation during training
ENABLE_EVAL="--enable_eval"
EVAL_EVERY_N_EPOCHS=1
EVAL_SAMPLES=20

# Weights & Biases
USE_WANDB="--use_wandb"   # leave empty to disable
WANDB_PROJECT="plain_sft_runs"
WANDB_RUN_NAME=""         # leave empty for auto naming

################################################################################
# Run training

echo "=========================================="
echo "Starting Plain SFT Training (no chat template)"
echo "=========================================="
echo "Round Name: $ROUND_NAME"
echo "Model: $MODEL"
echo "Data: $DATA_PATH"
echo "Mode: $MODE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Eval every: $EVAL_EVERY_N_EPOCHS epoch(s) | Samples: $EVAL_SAMPLES"
echo "=========================================="

# Build W&B args
WANDB_ARGS=""
if [ -n "$USE_WANDB" ]; then
    WANDB_ARGS="$USE_WANDB --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

python models/train_sft_plain.py \
    --mode "$MODE" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$ROUND_NAME" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    $GRADIENT_CHECKPOINTING \
    $ENABLE_EVAL \
    --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
    --eval_samples "$EVAL_SAMPLES" \
    $WANDB_ARGS

echo "=========================================="
echo "Training completed!"
echo "Checkpoints: checkpoints/${ROUND_NAME}_*/"
echo "Logs:        logs/${ROUND_NAME}_*/"
echo "=========================================="

