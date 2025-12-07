#!/bin/bash

################################################################################
# Solver-Coder Evaluation Script
# Pure code-based math problem solving with debug iterations
#
# Workflow:
# 1. Code Solver generates Python code to solve the problem
# 2. Execute code, capture output or errors
# 3. If error: feed code + error back to Solver for fixing
# 4. If success: optionally validate with Checker
# 5. Iterate until valid answer or max iterations reached
################################################################################

set -e

################################################################################
# Configuration Variables

# Model to use for code generation
# Options: Qwen2.5-Math-1.5B, Qwen3-0.6B, Qwen3-1.7B, etc.
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path (optional, leave empty for base model)
CHECKPOINT=""

# Agent method (fixed for this script)
AGENT="solver_coder"

# Test round name (used for organizing results)
ROUND_NAME="test_solver_coder"

# Dataset to use
# Options: gsm8k (grade school math), math500, math (competition math)
DATASET="gsm8k"

# Number of test cases (0 = full dataset)
COUNT=100

# Maximum iterations per problem (for fixing errors)
MAX_ITERATIONS=3

# Enable checker validation (true/false)
# true: Checker validates code output for reasonableness
# false: Accept any successful code execution
ENABLE_CHECKER="true"

# Code execution timeout in seconds
CODE_TIMEOUT=10

# Detailed output (shows full workflow for each problem)
DETAILED="false"

# Apply chat template (for chat-tuned models)
APPLY_CHAT_TEMPLATE="false"

# Resume from existing results (leave empty to start fresh)
RESUME_DIR=""

# Save interval (save intermediate results every N samples)
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "=============================================================================="
echo "              Solver-Coder - Pure Code-Based Math Solving                     "
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
echo "  Enable Checker: $ENABLE_CHECKER"
echo "  Code Timeout: ${CODE_TIMEOUT}s"
echo "  Detailed Output: $DETAILED"
echo "  Apply Chat Template: $APPLY_CHAT_TEMPLATE"
if [ -n "$RESUME_DIR" ]; then
    echo "  Resume: $RESUME_DIR"
fi
echo "------------------------------------------------------------------------------"
echo ""

################################################################################
# Run Evaluation

echo "Starting Solver-Coder evaluation..."
echo ""

# Build command
CMD="python -m evaluation.eval_agent \
    --model \"$MODEL\" \
    --agent \"$AGENT\" \
    --round \"$ROUND_NAME\" \
    --dataset \"$DATASET\" \
    --count \"$COUNT\" \
    --max_iterations \"$MAX_ITERATIONS\" \
    --enable_code_checker \"$ENABLE_CHECKER\" \
    --code_timeout \"$CODE_TIMEOUT\" \
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














