#!/bin/bash

################################################################################
# Evaluation Script Template
# 通用评估脚本模板 - 可复制此文件创建新的评估任务
################################################################################

set -e  # Exit on error

################################################################################
# Configuration Variables (需要修改)

# Model to evaluate
MODEL="Qwen3-0.6B"

# Test round name
ROUND_NAME="my_evaluation"

# Dataset: "gsm8k", "math", "math500", "competition_math"
DATASET="gsm8k"

# Number of test cases (0 = full dataset)
COUNT=100

# Evaluation mode: "standard"
MODE="standard"

# Detailed output
DETAILED="true"

# Resume from existing results directory (leave empty to start fresh)
# Example: RESUME_DIR="results/round1_model_gsm8k_1000_0101"
RESUME_DIR=""

# Save interval: save intermediate results every N samples (default: 10)
SAVE_INTERVAL=10

################################################################################
# Advanced Settings (可选)

# Output directory
OUTPUT_DIR="results"

# Generation parameters
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7


MODEL_PATH="pretrained_models/${MODEL}"

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                       Evaluation Script                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check model
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Model: $MODEL_PATH"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Evaluation Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL"
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT"
echo "  Mode: $MODE"
if [ -n "$RESUME_DIR" ]; then
    echo "  Resume: $RESUME_DIR"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# Run Evaluation

echo "Starting evaluation..."
echo ""

# Build command
CMD="python -m evaluation.eval \
    --model \"$MODEL\" \
    --round \"$ROUND_NAME\" \
    --dataset \"$DATASET\" \
    --count \"$COUNT\" \
    --mode \"$MODE\" \
    --detailed \"$DETAILED\" \
    --save_interval \"$SAVE_INTERVAL\""

# Add resume parameter if specified
if [ -n "$RESUME_DIR" ]; then
    CMD="$CMD --resume \"$RESUME_DIR\""
fi

# Execute command
eval $CMD

################################################################################
# Complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Evaluation Complete!                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

