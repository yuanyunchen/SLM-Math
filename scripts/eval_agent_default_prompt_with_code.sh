#!/bin/bash

################################################################################
# Agent with Default Prompt + Code Execution Evaluation Script
# Test agent_default_prompt_with_code on dataset
# 
# Key difference from agent_with_code_feedback:
# - Uses default prompt (no tool instruction)
# - Still detects and executes code if model generates it
################################################################################

set -e

################################################################################
# Configuration Variables

# Model
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path (optional)
CHECKPOINT=""

# Agent method
AGENT="agent_default_prompt_with_code"

# Test round name
ROUND_NAME="test_default_prompt_code"

# Dataset
DATASET="math500"

# Number of test cases (0 = full dataset)
COUNT=500

# Detailed output
DETAILED="false"

# Apply chat template (for chat-tuned models)
APPLY_CHAT_TEMPLATE="false"

# Resume from existing results (leave empty to start fresh)
RESUME_DIR=""

# Save interval
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "=============================================================================="
echo "       Agent with Default Prompt + Code Execution - Evaluation Script         "
echo "=============================================================================="
echo ""

# Check model
MODEL_PATH="pretrained_models/${MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "Model: $MODEL_PATH"

echo ""
echo "------------------------------------------------------------------------------"
echo "Evaluation Configuration:"
echo "------------------------------------------------------------------------------"
echo "  Model: $MODEL"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
fi
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

echo "Starting agent evaluation..."
echo ""

# Build command
CMD="python -m evaluation.eval_agent \
    --model \"$MODEL\" \
    --agent \"$AGENT\" \
    --round \"$ROUND_NAME\" \
    --dataset \"$DATASET\" \
    --count \"$COUNT\" \
    --detailed \"$DETAILED\" \
    --apply_chat_template \"$APPLY_CHAT_TEMPLATE\" \
    --save_interval \"$SAVE_INTERVAL\""

# Add checkpoint parameter if specified
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint \"$CHECKPOINT\""
fi

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
echo "Analysis report: results/${ROUND_NAME}_${MODEL}_${DATASET}_*/analysis_report.txt"
echo ""

