#!/bin/bash

################################################################################
# GPU 2: Qwen2.5-Math-1.5B with vLLM Backend Evaluation
# Datasets: GSM8K (500 samples) + MATH500
# Backend: vLLM for high-throughput inference
################################################################################

set -e  # Exit on error

# Set GPU
export CUDA_VISIBLE_DEVICES=2

################################################################################
# Configuration Variables

# Model to evaluate
MODEL="Qwen2.5-Math-1.5B"

# Test round name
ROUND_NAME="gpu2_qwen25_vllm"

# Evaluation mode: "standard"
MODE="standard"

# Detailed output
DETAILED="false"

# Inference backend: "vllm" for high-speed inference
INFERENCE_BACKEND="vllm"

# Batch size: vLLM handles batching internally, but you can tune this
# Recommended: 16-32 for optimal GPU utilization
BATCH_SIZE=16

# Resume from existing results directory (leave empty to start fresh)
RESUME_DIR=""

# Save interval: save intermediate results every N samples (default: 10)
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║        GPU 2: Qwen2.5-Math-1.5B with vLLM Backend Evaluation               ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

MODEL_PATH="pretrained_models/${MODEL}"

# Check model
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Model: $MODEL_PATH"
echo "✓ GPU: 2 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "✓ Backend: $INFERENCE_BACKEND"
echo "✓ Batch Size: $BATCH_SIZE"
echo ""

# Check vLLM availability
echo "Checking vLLM availability..."
if ! python models/check_vllm.py > /dev/null 2>&1; then
    echo "✗ Error: vLLM is not available"
    echo ""
    echo "vLLM is required for this script. Install it with:"
    echo "  pip install vllm"
    echo ""
    echo "Note: vLLM requires:"
    echo "  - CUDA 11.8 or higher"
    echo "  - Compatible NVIDIA GPU"
    echo "  - Linux operating system"
    echo ""
    exit 1
fi
echo "✓ vLLM is available"
echo ""

################################################################################
# Run Evaluation - GSM8K (500 samples)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Dataset 1: GSM8K (500 samples)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL"
echo "  Round: ${ROUND_NAME}_gsm8k"
echo "  Dataset: gsm8k"
echo "  Count: 500"
echo "  Mode: $MODE"
echo "  Backend: $INFERENCE_BACKEND"
echo "  Batch Size: $BATCH_SIZE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CMD="python -m evaluation.eval \
    --model \"$MODEL\" \
    --round \"${ROUND_NAME}_gsm8k\" \
    --dataset \"gsm8k\" \
    --count 500 \
    --mode \"$MODE\" \
    --detailed \"$DETAILED\" \
    --inference_backend \"$INFERENCE_BACKEND\" \
    --batch_size \"$BATCH_SIZE\" \
    --save_interval \"$SAVE_INTERVAL\""

if [ -n "$RESUME_DIR" ]; then
    CMD="$CMD --resume \"$RESUME_DIR\""
fi

eval $CMD

echo ""
echo "✓ GSM8K evaluation complete!"
echo ""

# ################################################################################
# # Run Evaluation - MATH500

# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "Dataset 2: MATH500 (full dataset)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  Model: $MODEL"
# echo "  Round: ${ROUND_NAME}_math500"
# echo "  Dataset: math500"
# echo "  Count: 0 (full dataset)"
# echo "  Mode: $MODE"
# echo "  Backend: $INFERENCE_BACKEND"
# echo "  Batch Size: $BATCH_SIZE"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo ""

# CMD="python -m evaluation.eval \
#     --model \"$MODEL\" \
#     --round \"${ROUND_NAME}_math500\" \
#     --dataset \"math500\" \
#     --count 0 \
#     --mode \"$MODE\" \
#     --detailed \"$DETAILED\" \
#     --inference_backend \"$INFERENCE_BACKEND\" \
#     --batch_size \"$BATCH_SIZE\" \
#     --save_interval \"$SAVE_INTERVAL\""

# if [ -n "$RESUME_DIR" ]; then
#     CMD="$CMD --resume \"$RESUME_DIR\""
# fi

# eval $CMD

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
echo "Performance Notes:"
echo "  - vLLM backend provides 2-5x speedup over transformers"
echo "  - Batch size $BATCH_SIZE used for optimal GPU utilization"
echo "  - Consider increasing batch size for larger GPUs (32-64)"
echo ""

