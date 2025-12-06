#!/bin/bash

################################################################################
# Agent with Math RAG Evaluation Script
# Few-Shot Retrieval-Augmented Generation for Math Problems
#
# This agent retrieves similar problems from the training set and uses them
# as few-shot examples to help the model solve new problems.
################################################################################

set -e

################################################################################
# Configuration Variables

# Model
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path (optional)
# Set to empty string "" to use base pretrained model
CHECKPOINT=""

# Agent method (fixed for this script)
AGENT="agent_with_math_rag"

# Test round name
ROUND_NAME="test_math_rag"

# Dataset: gsm8k or math500
DATASET="gsm8k"

# Number of test cases (0 = full dataset)
COUNT=50

# Start index (for resuming or testing specific range)
START=0

################################################################################
# RAG Configuration

# Number of similar examples to retrieve (top-k)
RAG_TOP_K=3

# Include solution steps in retrieved examples
# true: Include full solution (more helpful but longer prompt)
# false: Only include question and answer (shorter prompt)
RAG_INCLUDE_SOLUTION="true"

# Enable Python code execution with RAG
# true: RAG + Python tools (best accuracy, slower)
# false: RAG only (faster)
RAG_ENABLE_TOOLS="false"

################################################################################
# Other Settings

# Detailed output
DETAILED="true"

# Apply chat template to prompts (default: true for Qwen models)
APPLY_CHAT_TEMPLATE="true"

# Resume from existing results (leave empty to start fresh)
RESUME_DIR=""

# Save interval
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "================================================================================"
echo "          Agent with Math RAG - Few-Shot Retrieval Evaluation"
echo "================================================================================"
echo ""

# Check model
MODEL_PATH="pretrained_models/${MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found at $MODEL_PATH"
    exit 1
fi
echo "[OK] Model: $MODEL_PATH"

# Check dataset
DATA_PATH="data/${DATASET}"
if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Dataset not found at $DATA_PATH"
    exit 1
fi
echo "[OK] Dataset: $DATA_PATH"

echo ""
echo "--------------------------------------------------------------------------------"
echo "Evaluation Configuration:"
echo "--------------------------------------------------------------------------------"
echo "  Model: $MODEL"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
fi
echo "  Agent: $AGENT"
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT (0 = full dataset)"
echo "  Start Index: $START"
echo ""
echo "RAG Configuration:"
echo "  Top-K Examples: $RAG_TOP_K"
echo "  Include Solution: $RAG_INCLUDE_SOLUTION"
echo "  Enable Python Tools: $RAG_ENABLE_TOOLS"
echo ""
echo "Other Settings:"
echo "  Detailed Output: $DETAILED"
echo "  Apply Chat Template: $APPLY_CHAT_TEMPLATE"
if [ -n "$RESUME_DIR" ]; then
    echo "  Resume: $RESUME_DIR"
fi
echo "--------------------------------------------------------------------------------"
echo ""

################################################################################
# Run Evaluation

echo "Starting Math RAG evaluation..."
echo ""

# Build command
CMD="python -m evaluation.eval_agent \
    --model \"$MODEL\" \
    --agent \"$AGENT\" \
    --round \"$ROUND_NAME\" \
    --dataset \"$DATASET\" \
    --count \"$COUNT\" \
    --start \"$START\" \
    --rag_top_k \"$RAG_TOP_K\" \
    --rag_include_solution \"$RAG_INCLUDE_SOLUTION\" \
    --rag_enable_tools \"$RAG_ENABLE_TOOLS\" \
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
echo "================================================================================"
echo "                         Evaluation Complete!"
echo "================================================================================"
echo ""
echo "Results saved to: results/${ROUND_NAME}_${MODEL}_${DATASET}_*"
echo "Analysis report: results/${ROUND_NAME}_${MODEL}_${DATASET}_*/analysis_report.txt"
echo ""

