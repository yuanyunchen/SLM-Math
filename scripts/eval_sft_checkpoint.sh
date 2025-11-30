#!/bin/bash

################################################################################
# Evaluation Script for SFT Checkpoint
# Example: Evaluate SFT fine-tuned model on GSM8K
################################################################################

set -e


export CUDA_VISIBLE_DEVICES=0

################################################################################
# Configuration Variables

# Model name (can be any name for reference, not used for loading)
MODEL="Qwen2.5-Math-1.5B"

# SFT checkpoint path
# Use the latest checkpoint or best performing checkpoint
CHECKPOINT="checkpoints/full_sft_1124/checkpoint-1485"

# Test round name
ROUND_NAME="eval_sft_final_model"

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
echo "║             SFT Checkpoint Evaluation Script                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check SFT checkpoint
if [ ! -d "$CHECKPOINT" ]; then
    echo "✗ Error: SFT checkpoint not found at $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CHECKPOINT/config.json" ]; then
    echo "✗ Error: Not a valid SFT checkpoint (missing config.json)"
    exit 1
fi
echo "✓ SFT Checkpoint: $CHECKPOINT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Evaluation Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL (using checkpoint)"
echo "  SFT Checkpoint: $CHECKPOINT"
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT"
echo "  Mode: $MODE"
echo "  Detailed Output: $DETAILED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# Run Evaluation

echo "Starting evaluation with SFT checkpoint..."
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

