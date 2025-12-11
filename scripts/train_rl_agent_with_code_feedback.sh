#!/bin/bash

################################################################################
# GRPO RL Training Script for Agent with Code Feedback
# Group Relative Policy Optimization for Agent with Code Execution Feedback
#
# Usage:
#   bash scripts/train_rl_agent_with_code_feedback.sh
#
# Features:
# - Two-step generation: initial reasoning+code → code execution → final answer
# - Binary reward verifier (correct/wrong) based on final answer
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
OUTPUT_DIR="results/rl_agent_code_feedback_$(date +%Y%m%d_%H%M%S)"

# GPU to use
export CUDA_VISIBLE_DEVICES=0

# W&B settings
WANDB_PROJECT="slm_math_rl_code_feedback"
WANDB_RUN_NAME=""  # Leave empty for auto-generated name, or set custom name
USE_WANDB="--use_wandb"  # Comment out to disable W&B logging

################################################################################
# Training Hyperparameters

# Number of epochs
NUM_EPOCHS=1

# Max samples (set to -1 for all data)
MAX_SAMPLES=-1

# Batch size and gradient accumulation
# Note: Code feedback workflow uses more memory due to two-step generation
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8

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
echo "GRPO RL Training - Agent with Code Feedback"
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
    echo "Please update MODEL_PATH in this script to point to your model checkpoint"
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

python models/train_rl_agent_with_code_feedback.py \
    --config "$CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --temperature "$TEMPERATURE" \
    --kl_coef "$KL_COEF" \
    --logging_steps "$LOGGING_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_samples "$EVAL_SAMPLES" \
    $WANDB_ARGS

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
echo "To evaluate the trained agent:"
echo "  python -m evaluation.eval_agent --model $MODEL_PATH --checkpoint $OUTPUT_DIR/checkpoint-*/ --agent agent_with_code_feedback"
echo ""
echo "=========================================="

