#!/bin/bash

################################################################################
# Agent with RAG Review Evaluation Script
# Two-Stage RAG: Generate first, then review with retrieved examples
#
# This agent first generates an answer independently, then retrieves similar
# examples and asks the model to review its answer with CAUTION warnings.
################################################################################

set -e

################################################################################
# Configuration Variables

# Model
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path (optional)
CHECKPOINT=""

# Agent method (fixed for this script)
AGENT="agent_with_rag_review"

# Test round name
ROUND_NAME="test_rag_review"

# Dataset: gsm8k or math500
DATASET="gsm8k"

# Number of test cases (0 = full dataset)
COUNT=50

# Start index
START=0

################################################################################
# RAG Configuration

# Number of similar examples to retrieve (top-k)
RAG_TOP_K=3

# Include solution steps in retrieved examples
RAG_INCLUDE_SOLUTION="true"

# Enable Python code execution
RAG_ENABLE_TOOLS="false"

################################################################################
# Other Settings

# Detailed output
DETAILED="true"

# Apply chat template (required for Qwen models)
APPLY_CHAT_TEMPLATE="true"

# Resume from existing results
RESUME_DIR=""

# Save interval
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "================================================================================"
echo "          Agent with RAG Review - Two-Stage Evaluation"
echo "================================================================================"
echo ""
echo "Workflow:"
echo "  Stage 1: Generate answer independently (no RAG)"
echo "  Stage 2: Review with retrieved examples (with CAUTION warnings)"
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
echo "Configuration:"
echo "--------------------------------------------------------------------------------"
echo "  Model: $MODEL"
echo "  Agent: $AGENT"
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT"
echo ""
echo "RAG Settings:"
echo "  Top-K Examples: $RAG_TOP_K"
echo "  Include Solution: $RAG_INCLUDE_SOLUTION"
echo "  Enable Tools: $RAG_ENABLE_TOOLS"
echo "--------------------------------------------------------------------------------"
echo ""

################################################################################
# Run Evaluation

echo "Starting RAG Review evaluation..."
echo ""

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

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint \"$CHECKPOINT\""
fi

if [ -n "$RESUME_DIR" ]; then
    CMD="$CMD --resume \"$RESUME_DIR\""
fi

eval $CMD

################################################################################
# Complete

echo ""
echo "================================================================================"
echo "                         Evaluation Complete!"
echo "================================================================================"
echo ""
echo "Results: results/${ROUND_NAME}_${MODEL}_${DATASET}_*"
echo ""













