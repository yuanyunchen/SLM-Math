#!/bin/bash

################################################################################
# GRPO RL Training Script (Baseline)
# Group Relative Policy Optimization for Mathematical Reasoning
# Configuration based on exp_gpu1_rl_baseline.sh settings
#
# Usage:
#   bash scripts/train_rl_baseline.sh
#
# Features:
# - Binary reward verifier (correct/wrong)
# - LoRA for parameter-efficient training
# - Multiple generations per prompt for comparison
################################################################################

set -e  # Exit on error

################################################################################
# Training Settings

# Model path (can be base model or SFT checkpoint)
# Option 1: Start from base model
MODEL_PATH="pretrained_models/Qwen2.5-Math-1.5B"

# Option 2: Start from SFT checkpoint (comment out Option 1 and uncomment this)
# MODEL_PATH="checkpoints/sft_20251124_131423/final_model"

# Configuration file
CONFIG_FILE="configs/rl_grpo_config.yaml"

# Output directory
OUTPUT_DIR="results/rl_grpo_baseline_$(date +%Y%m%d_%H%M%S)"

# GPU to use
export CUDA_VISIBLE_DEVICES=0

# W&B settings
WANDB_PROJECT="slm_math_experiments_1124"
WANDB_RUN_NAME=""  # Leave empty for auto-generated name, or set custom name
USE_WANDB="--use_wandb"  # Comment out to disable W&B logging

################################################################################
# Training Hyperparameters

# Number of epochs
NUM_EPOCHS=1

# Max samples (set to -1 for all data)
MAX_SAMPLES=-1

# Batch size and gradient accumulation
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4

# Learning rate
LEARNING_RATE=5e-6

# Generation settings
NUM_RETURN_SEQUENCES=2
TEMPERATURE=0.7

# KL divergence coefficient
# Set to 0 to disable KL penalty (saves ~3x speed and ~3GB VRAM)
KL_COEF=0

# Logging and evaluation frequency
LOGGING_STEPS=10
EVAL_STEPS=100
SAVE_STEPS=300
EVAL_SAMPLES=200  # Samples per dataset for evaluation

################################################################################
# Pre-flight Checks

echo "=========================================="
echo "GRPO RL Training - Baseline"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Max samples: $MAX_SAMPLES"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Learning rate: $LEARNING_RATE"
echo "Num return sequences: $NUM_RETURN_SEQUENCES"
echo "Temperature: $TEMPERATURE"
if [ "$KL_COEF" = "0" ]; then
    echo "KL Penalty: Disabled (faster training)"
else
    echo "KL Penalty: Enabled (kl_coef=$KL_COEF)"
fi
echo "Logging steps: $LOGGING_STEPS"
echo "Eval steps: $EVAL_STEPS"
echo "Save steps: $SAVE_STEPS"
echo "Eval samples: $EVAL_SAMPLES per dataset"
echo "=========================================="

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found at $MODEL_PATH"
    echo "Please update MODEL_PATH in this script to point to your SFT checkpoint"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure CUDA is available."
fi

echo ""
echo "Starting training..."
echo ""

################################################################################
# Run Training

# Build wandb arguments
WANDB_ARGS=""
if [ -n "$USE_WANDB" ]; then
    WANDB_ARGS="$USE_WANDB --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

python models/train_rl_grpo.py \
    --sft_checkpoint "$MODEL_PATH" \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --kl_coef "$KL_COEF" \
    --gpus "${CUDA_VISIBLE_DEVICES}"

################################################################################
# Post-training Summary

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files created:"
echo "  - checkpoints: $OUTPUT_DIR/checkpoint-*/"
echo "  - training log: $OUTPUT_DIR/training.log"
echo "  - training state: $OUTPUT_DIR/*/training_state.json"
echo "  - evaluation summary: $OUTPUT_DIR/*/eval_summary.txt"
echo ""
echo "To resume training or use the model:"
echo "  - Load checkpoint from: $OUTPUT_DIR/checkpoint-*/"
echo "  - View logs: cat $OUTPUT_DIR/training.log"
echo "  - View evaluation: cat $OUTPUT_DIR/*/eval_summary.txt"
echo ""
echo "=========================================="
