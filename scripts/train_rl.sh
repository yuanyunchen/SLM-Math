#!/bin/bash

################################################################################
# Rule-based RL Training Script
# Reinforcement Learning with Outcome-based Rewards
#
# Features:
# - PPO (Proximal Policy Optimization)
# - Rule-based reward system (correctness, format, reasoning quality)
# - Support for LoRA and QLoRA
################################################################################

set -e  # Exit on error

################################################################################
# Training Settings

# Model to train
MODEL="Qwen2.5-Math-1.5B"

# Run name
RUN_NAME="qwen25math_1.5b_rl_ppo"

# RL algorithm: "ppo", "grpo", "reinforce"
RL_ALGORITHM="ppo"

# Training data
DATASET="gsm8k"
DATA_FILE="data/cot_generated/cot_x_ai_grok_4_1_fast_reasoning_gsm8k_train_7473_1120_2035/cot_data.json"

# Training duration
NUM_EPOCHS=3
BATCH_SIZE=8

# Configuration file
CONFIG_FILE="configs/rl_config.yaml"

################################################################################
# Advanced Settings

# Reward weights (optional overrides)
# CORRECTNESS_WEIGHT=1.0
# FORMAT_WEIGHT=0.1
# REASONING_WEIGHT=0.2

# PPO parameters
# CLIP_RANGE=0.2
# KL_COEF=0.1

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                  Rule-based RL Training - PPO                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if model exists
MODEL_PATH="pretrained_models/$MODEL"
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Model found: $MODEL_PATH"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "✗ Error: Data file not found: $DATA_FILE"
    exit 1
fi
echo "✓ Data file found: $DATA_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file: $CONFIG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL"
echo "  Algorithm: $RL_ALGORITHM"
echo "  Run name: $RUN_NAME"
echo "  Dataset: $DATASET"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Confirm before starting
read -p "Start RL training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

################################################################################
# Run Training

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Starting RL Training...                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Build command
CMD="python -m models.train_RL --config $CONFIG_FILE"

# Run training
echo "Command: $CMD"
echo ""
$CMD

################################################################################
# Training Complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         RL Training Complete!                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: results/rl_checkpoints/${RUN_NAME}_*/"
echo ""
echo "Next steps:"
echo "  1. View training logs:"
echo "     tensorboard --logdir=logs/rl"
echo ""
echo "  2. Evaluate trained model:"
echo "     python -m evaluation.eval_pipeline --model results/rl_checkpoints/${RUN_NAME}_*/final_model"
echo ""


