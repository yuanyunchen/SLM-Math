#!/bin/bash

################################################################################
# Multi-Agent GRPO RL Training Script
# Solver + Verifier Co-Evolution
#
# This script trains two models jointly:
# - Solver: Generates code-based solutions to math problems
# - Verifier: Evaluates solver's code and decides CORRECT/INCORRECT/UNCLEAR
#
# Usage:
#   bash scripts/train_rl_multi_agent.sh
#
################################################################################

set -e  # Exit on error

################################################################################
# Model Paths
################################################################################

# Solver model - use SFT checkpoint trained on code solutions
# Option 1: Base model
SOLVER_MODEL_PATH="pretrained_models/Qwen2.5-Math-1.5B"

# Option 2: SFT checkpoint (uncomment and modify)
# SOLVER_MODEL_PATH="checkpoints/solver_sft_YYYYMMDD/final_model"

# Verifier model - use SFT checkpoint trained on code verification
# Option 1: Base model
VERIFIER_MODEL_PATH="pretrained_models/Qwen2.5-Math-1.5B"

# Option 2: SFT checkpoint (uncomment and modify)
# VERIFIER_MODEL_PATH="checkpoints/verifier_sft_YYYYMMDD/final_model"

################################################################################
# Configuration
################################################################################

# Configuration file
CONFIG_FILE="configs/rl_multi_agent_config.yaml"

# Output directory
OUTPUT_DIR="results/rl_multi_agent_$(date +%Y%m%d_%H%M%S)"

# GPU to use
export CUDA_VISIBLE_DEVICES=0

# W&B settings
WANDB_PROJECT="slm_math_multi_agent"
WANDB_RUN_NAME=""  # Leave empty for auto-generated name
USE_WANDB="--use_wandb"  # Comment out to disable W&B logging

################################################################################
# Training Hyperparameters
################################################################################

# Number of epochs
NUM_EPOCHS=1

# Max samples (set to -1 for all data)
MAX_SAMPLES=-1

# Datasets to use
DATASETS="gsm8k,math500"

# Batch size and gradient accumulation
# Note: Using smaller batch size due to two models in memory
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8

# Learning rates
SOLVER_LEARNING_RATE=5e-6
VERIFIER_LEARNING_RATE=1e-5

# Solver generation settings
SOLVER_NUM_RETURN_SEQUENCES=2
SOLVER_TEMPERATURE=0.7

# Logging and evaluation frequency
LOGGING_STEPS=10
EVAL_STEPS=100
SAVE_STEPS=200
EVAL_SAMPLES=100  # Samples per dataset for evaluation

# Training mode
# Set these to control which models to train
TRAIN_SOLVER="true"   # Set to "false" to freeze solver
TRAIN_VERIFIER="true" # Set to "false" to freeze verifier

################################################################################
# Pre-flight Checks
################################################################################

echo "=========================================="
echo "Multi-Agent GRPO RL Training"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo ""
echo "Models:"
echo "  Solver: $SOLVER_MODEL_PATH"
echo "  Verifier: $VERIFIER_MODEL_PATH"
echo ""
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Training Settings:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Max samples: $MAX_SAMPLES"
echo "  Datasets: $DATASETS"
echo "  Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "  Solver LR: $SOLVER_LEARNING_RATE"
echo "  Verifier LR: $VERIFIER_LEARNING_RATE"
echo "  Solver generations: $SOLVER_NUM_RETURN_SEQUENCES"
echo "  Temperature: $SOLVER_TEMPERATURE"
echo ""
echo "Training Mode:"
echo "  Train Solver: $TRAIN_SOLVER"
echo "  Train Verifier: $TRAIN_VERIFIER"
echo ""
echo "Logging:"
echo "  Logging steps: $LOGGING_STEPS"
echo "  Eval steps: $EVAL_STEPS"
echo "  Save steps: $SAVE_STEPS"
echo "  Eval samples: $EVAL_SAMPLES per dataset"
echo "=========================================="

# Check if solver model exists
if [ ! -d "$SOLVER_MODEL_PATH" ]; then
    echo "Error: Solver model not found at $SOLVER_MODEL_PATH"
    echo "Please update SOLVER_MODEL_PATH in this script"
    exit 1
fi

# Check if verifier model exists
if [ ! -d "$VERIFIER_MODEL_PATH" ]; then
    echo "Error: Verifier model not found at $VERIFIER_MODEL_PATH"
    echo "Please update VERIFIER_MODEL_PATH in this script"
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
################################################################################

# Build wandb arguments
WANDB_ARGS=""
if [ -n "$USE_WANDB" ]; then
    WANDB_ARGS="$USE_WANDB --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

# Build training mode arguments
TRAIN_MODE_ARGS=""
if [ "$TRAIN_SOLVER" = "false" ]; then
    TRAIN_MODE_ARGS="$TRAIN_MODE_ARGS --no_train_solver"
fi
if [ "$TRAIN_VERIFIER" = "false" ]; then
    TRAIN_MODE_ARGS="$TRAIN_MODE_ARGS --no_train_verifier"
fi

python models/train_rl_multi_agent.py \
    --config "$CONFIG_FILE" \
    --solver_model_path "$SOLVER_MODEL_PATH" \
    --verifier_model_path "$VERIFIER_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --datasets "$DATASETS" \
    --max_samples "$MAX_SAMPLES" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --solver_learning_rate "$SOLVER_LEARNING_RATE" \
    --verifier_learning_rate "$VERIFIER_LEARNING_RATE" \
    --solver_num_return_sequences "$SOLVER_NUM_RETURN_SEQUENCES" \
    --solver_temperature "$SOLVER_TEMPERATURE" \
    --logging_steps "$LOGGING_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_samples "$EVAL_SAMPLES" \
    $TRAIN_MODE_ARGS \
    $WANDB_ARGS

################################################################################
# Post-training Summary
################################################################################

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files created:"
echo "  - Solver checkpoints: $OUTPUT_DIR/*/solver/"
echo "  - Verifier checkpoints: $OUTPUT_DIR/*/verifier/"
echo "  - Training log: $OUTPUT_DIR/training.log"
echo "  - Training state: $OUTPUT_DIR/*/training_state.json"
echo "  - Evaluation summary: $OUTPUT_DIR/*/eval_summary.txt"
echo ""
echo "To use the trained models:"
echo "  - Load solver from: $OUTPUT_DIR/epoch_N/solver/"
echo "  - Load verifier from: $OUTPUT_DIR/epoch_N/verifier/"
echo ""
echo "To view logs:"
echo "  - Training log: cat $OUTPUT_DIR/training.log"
echo "  - Evaluation: cat $OUTPUT_DIR/*/eval_summary.txt"
echo ""
echo "=========================================="

