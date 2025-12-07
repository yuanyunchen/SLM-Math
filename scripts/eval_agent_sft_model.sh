#!/bin/bash

################################################################################
# Evaluation Script for Agent SFT Trained Model
# Tests the fine-tuned model on solver_interactive_code workflow
################################################################################

set -e

################################################################################
# Configuration Variables

# Base model (must match what was used for training)
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path - point to the trained model
# Options:
#   - For SFT: checkpoints/agent_sft_v1_sft_<timestamp>/final_model
#   - For LoRA: checkpoints/agent_lora_v1_lora_r64_<timestamp>/final_model
#
# Example SFT:
# CHECKPOINT="checkpoints/agent_sft_v1_sft_20251206_163211/final_model"
#
# Example LoRA (update timestamp after training):
# CHECKPOINT="checkpoints/agent_lora_v1_lora_r64_<timestamp>/final_model"
CHECKPOINT="checkpoints/agent_lora_v1_lora_r32_20251207_082806/final_model"

# Agent method
AGENT="solver_interactive_code"

# Test round name
ROUND_NAME="eval_agent_lora_r32_v1"

# Dataset
# Options: gsm8k, math500, math
DATASET="gsm8k"

# Number of test cases (0 = full dataset)
COUNT=500

# Detailed output (true for debugging)
DETAILED="false"

# Apply chat template
# Set to "false" for models trained WITHOUT chat template (plain text format)
# Set to "true" for models trained WITH chat template
APPLY_CHAT_TEMPLATE="false"

# Resume from existing results (leave empty to start fresh)
RESUME_DIR=""

# Save interval
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "=============================================================================="
echo "          Agent SFT Model Evaluation - Interactive Code Solver                "
echo "=============================================================================="
echo ""

# Check base model
MODEL_PATH="pretrained_models/${MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Base model not found at $MODEL_PATH"
    exit 1
fi
echo "Base Model: $MODEL_PATH"

# Check checkpoint
if [ -n "$CHECKPOINT" ]; then
    if [ ! -d "$CHECKPOINT" ]; then
        echo "Error: Checkpoint not found at $CHECKPOINT"
        exit 1
    fi
    echo "Checkpoint: $CHECKPOINT"
fi

echo ""
echo "------------------------------------------------------------------------------"
echo "Evaluation Configuration:"
echo "------------------------------------------------------------------------------"
echo "  Base Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Agent Method: $AGENT"
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT"
echo "  Detailed Output: $DETAILED"
echo "  Apply Chat Template: $APPLY_CHAT_TEMPLATE"
if [ -n "$RESUME_DIR" ]; then
    echo "  Resume: $RESUME_DIR"
fi
echo "------------------------------------------------------------------------------"
echo ""

################################################################################
# Run Evaluation

echo "Starting evaluation..."
echo ""

# Build command
CMD="python -m evaluation.eval_agent \
    --model \"$MODEL\" \
    --checkpoint \"$CHECKPOINT\" \
    --agent \"$AGENT\" \
    --round \"$ROUND_NAME\" \
    --dataset \"$DATASET\" \
    --count \"$COUNT\" \
    --detailed \"$DETAILED\" \
    --apply_chat_template \"$APPLY_CHAT_TEMPLATE\" \
    --save_interval \"$SAVE_INTERVAL\""

# Add resume if specified
if [ -n "$RESUME_DIR" ]; then
    CMD="$CMD --resume \"$RESUME_DIR\""
fi

# Execute command
eval $CMD

################################################################################
# Complete

echo ""
echo "=============================================================================="
echo "                         Evaluation Complete!                                 "
echo "=============================================================================="
echo ""
echo "Results saved to: results/${ROUND_NAME}_${MODEL}_${DATASET}_*"
echo ""

