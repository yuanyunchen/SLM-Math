#!/bin/bash

################################################################################
# GPU 0: Qwen2.5-Math-1.5B Base Evaluation
# Datasets: GSM8K (500 samples) + MATH500
################################################################################

set -e  # Exit on error

# Set GPU
export CUDA_VISIBLE_DEVICES=2

################################################################################
# Configuration Variables

# Model to evaluate
MODEL="Qwen2.5-Math-1.5B"

# Test round name
ROUND_NAME="gpu0_qwen25_base"

# Evaluation mode: "standard"
MODE="standard"

# Detailed output
DETAILED="false"

# Resume from existing results directory (leave empty to start fresh)
RESUME_DIR=""

# Save interval: save intermediate results every N samples (default: 10)
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          GPU 0: Qwen2.5-Math-1.5B Base Evaluation                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

MODEL_PATH="pretrained_models/${MODEL}"

# Check model
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Model: $MODEL_PATH"
echo "✓ GPU: 0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""

# ################################################################################
# # Run Evaluation - GSM8K (500 samples)

# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "Dataset 1: GSM8K (500 samples)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  Model: $MODEL"
# echo "  Round: ${ROUND_NAME}_gsm8k"
# echo "  Dataset: gsm8k"
# echo "  Count: 500"
# echo "  Mode: $MODE"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo ""

# CMD="python -m evaluation.eval \
#     --model \"$MODEL\" \
#     --round \"${ROUND_NAME}_gsm8k\" \
#     --dataset \"gsm8k\" \
#     --count 500 \
#     --mode \"$MODE\" \
#     --detailed \"$DETAILED\" \
#     --save_interval \"$SAVE_INTERVAL\""

# if [ -n "$RESUME_DIR" ]; then
#     CMD="$CMD --resume \"$RESUME_DIR\""
# fi

# eval $CMD

# echo ""
# echo "✓ GSM8K evaluation complete!"
# echo ""

################################################################################
# Run Evaluation - MATH500

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Dataset 2: MATH500 (full dataset)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL"
echo "  Round: ${ROUND_NAME}_math500"
echo "  Dataset: math500"
echo "  Count: 0 (full dataset)"
echo "  Mode: $MODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CMD="python -m evaluation.eval \
    --model \"$MODEL\" \
    --round \"${ROUND_NAME}_math500\" \
    --dataset \"math500\" \
    --count 0 \
    --mode \"$MODE\" \
    --detailed \"$DETAILED\" \
    --save_interval \"$SAVE_INTERVAL\""

if [ -n "$RESUME_DIR" ]; then
    CMD="$CMD --resume \"$RESUME_DIR\""
fi

eval $CMD

################################################################################
# Complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         All Evaluations Complete!                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  - results/${ROUND_NAME}_gsm8k_*"
echo "  - results/${ROUND_NAME}_math500_*"
echo ""

