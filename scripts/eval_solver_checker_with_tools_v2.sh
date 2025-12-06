#!/bin/bash

################################################################################
# Solver-Checker with Tools V2 Evaluation Script
# Test solver_checker_with_tools_v2 on math datasets
################################################################################

set -e

################################################################################
# Configuration Variables

# Model
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path (optional)
CHECKPOINT=""

# Agent method (fixed for this script)
AGENT="solver_checker_with_tools_v2"

# Test round name
ROUND_NAME="test_solver_checker_v2"

# Dataset
# Options: gsm8k, math500, math
DATASET="math500"

# Number of test cases (0 = full dataset)
COUNT=500

# Maximum iterations per problem
MAX_ITERATIONS=3

# Enable code execution for solver
# Options: true, false
ENABLE_SOLVER_TOOLS="true"

# Enable code execution for checker
# Options: true, false
ENABLE_CHECKER_TOOLS="true"

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
echo "         Solver-Checker with Tools V2 - Evaluation Script                     "
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
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Solver Tools: $ENABLE_SOLVER_TOOLS"
echo "  Checker Tools: $ENABLE_CHECKER_TOOLS"
echo "  Detailed Output: $DETAILED"
echo "  Apply Chat Template: $APPLY_CHAT_TEMPLATE"
if [ -n "$RESUME_DIR" ]; then
    echo "  Resume: $RESUME_DIR"
fi
echo "------------------------------------------------------------------------------"
echo ""

################################################################################
# Run Evaluation

echo "Starting Solver-Checker with Tools V2 evaluation..."
echo ""

# Build command
CMD="python -m evaluation.eval_agent \
    --model \"$MODEL\" \
    --agent \"$AGENT\" \
    --round \"$ROUND_NAME\" \
    --dataset \"$DATASET\" \
    --count \"$COUNT\" \
    --max_iterations \"$MAX_ITERATIONS\" \
    --enable_solver_tools \"$ENABLE_SOLVER_TOOLS\" \
    --enable_checker_tools \"$ENABLE_CHECKER_TOOLS\" \
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


