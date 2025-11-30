#!/bin/bash

################################################################################
# Evaluation Script for LoRA Checkpoint
# Example: Evaluate LoRA fine-tuned model on GSM8K
################################################################################

set -e

################################################################################
# Configuration Variables


export CUDA_VISIBLE_DEVICES=1

# Base model name (must match the model used during training)
MODEL="Qwen2.5-Math-1.5B"

# LoRA checkpoint path
# Use the latest checkpoint or best performing checkpoint
CHECKPOINT="checkpoints/lora_r16_1124/final_model"
# CHECKPOINT="checkpoints/lora_r16_20251124_130958/checkpoint-297"

# Test round name
ROUND_NAME="eval_lora_r16_final_model"

# Dataset: "gsm8k", "math", "math500"
DATASET="gsm8k"

# Number of test cases (0 = full dataset)
COUNT=500

# Evaluation mode: "standard"
MODE="standard"

# Detailed output
DETAILED="false"

# Save interval: save intermediate results every N samples
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║            LoRA Checkpoint Evaluation Script                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check base model
MODEL_PATH="pretrained_models/${MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Base model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Base Model: $MODEL_PATH"

# Check LoRA checkpoint
if [ ! -d "$CHECKPOINT" ]; then
    echo "✗ Error: LoRA checkpoint not found at $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
    echo "✗ Error: Not a valid LoRA checkpoint (missing adapter_config.json)"
    exit 1
fi
echo "✓ LoRA Checkpoint: $CHECKPOINT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Evaluation Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Base Model: $MODEL"
echo "  LoRA Checkpoint: $CHECKPOINT"
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT"
echo "  Mode: $MODE"
echo "  Detailed Output: $DETAILED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# Run Evaluation

echo "Starting evaluation with LoRA checkpoint..."
echo ""

python -m evaluation.eval \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --mode "$MODE" \
    --detailed "$DETAILED" \
    --save_interval "$SAVE_INTERVAL" 

################################################################################
# Complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Evaluation Complete!                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

