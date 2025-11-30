#!/bin/bash

################################################################################
# Experiment: RL Baseline Training (GRPO)
# GPU: 1
# Description: GRPO RL training on base model
#              Log every 10 steps, eval every 100 steps, save every 300 steps
################################################################################

set -e  # Exit on error

################################################################################
# Configuration

# GPU
export CUDA_VISIBLE_DEVICES=1

# Experiment naming
WANDB_PROJECT="slm_math_experiments_1124"
WANDB_RUN_NAME="gpu1_rl_grpo_from_base"

# Model and data
MODEL_PATH="pretrained_models/Qwen2.5-Math-1.5B"
CONFIG_FILE="configs/rl_grpo_config.yaml"

# Output directory with timestamp
OUTPUT_DIR="results/exp_rl_baseline_$(date +%Y%m%d_%H%M%S)"

# Training config overrides
NUM_EPOCHS=1
LOGGING_STEPS=10
EVAL_STEPS=100
SAVE_STEPS=300
MAX_SAMPLES=-1  # Use all data

# RL-specific parameters (can be customized)
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-6
NUM_RETURN_SEQUENCES=2
TEMPERATURE=0.7
# KL coefficient: Set to 0 to disable (saves ~3x speed and ~3GB VRAM)
KL_COEF=0
EVAL_SAMPLES=200  # Samples per dataset for evaluation

################################################################################
# Pre-flight Checks

echo "=========================================="
echo "Experiment: RL Baseline Training (GRPO)"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
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
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

################################################################################
# Run Training

python models/train_rl_base.py \
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
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME"

echo "=========================================="
echo "GPU 1: RL Baseline Training Completed!"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

