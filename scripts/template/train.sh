#!/bin/bash

################################################################################
# Training Script Template
# 通用训练脚本模板 - 可复制此文件创建新的训练任务
################################################################################

set -e  # Exit on error

################################################################################
# Configuration Variables (需要修改)

# Model settings
MODEL="Qwen2.5-Math-1.5B"
MODEL_PATH="pretrained_models/${MODEL}"

# Training type: "sft", "rl", "distill", "qlora"
TRAINING_TYPE="sft"

# Run name (用于标识本次训练)
RUN_NAME="my_training_run"

# Dataset settings
DATASET="gsm8k"
DATA_FILE="data/cot_generated/cot_x_ai_grok_4_1_fast_reasoning_gsm8k_train_7473_1120_2035/cot_data.json"

# Training parameters
NUM_EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE="2.0e-4"

# Configuration file
CONFIG_FILE="configs/${TRAINING_TYPE}_config.yaml"

################################################################################
# Advanced Settings (可选修改)

# Output directory
OUTPUT_DIR="results/${TRAINING_TYPE}_checkpoints"

# LoRA settings
USE_LORA=true
LORA_R=16
LORA_ALPHA=32

# Hardware
USE_BF16=true
GRADIENT_CHECKPOINTING=true

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                       Training Script                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check model
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Model: $MODEL_PATH"

# Check data
if [ ! -f "$DATA_FILE" ]; then
    echo "✗ Error: Data file not found: $DATA_FILE"
    exit 1
fi
echo "✓ Data: $DATA_FILE"

# Check config
if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Config: $CONFIG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Training type: $TRAINING_TYPE"
echo "  Model: $MODEL"
echo "  Run name: $RUN_NAME"
echo "  Dataset: $DATASET"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# Run Training

echo "Starting training..."
echo ""

# Build command based on training type
case "$TRAINING_TYPE" in
    "sft"|"qlora")
        CMD="python -m models.train_SFT --config $CONFIG_FILE"
        ;;
    "rl")
        CMD="python -m models.train_RL --config $CONFIG_FILE"
        ;;
    "distill")
        CMD="python -m models.train_distill --config $CONFIG_FILE"
        ;;
    *)
        echo "✗ Error: Unknown training type: $TRAINING_TYPE"
        exit 1
        ;;
esac

# Execute
echo "Command: $CMD"
echo ""
$CMD

################################################################################
# Complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Training Complete!                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/${RUN_NAME}_*/"
echo ""

