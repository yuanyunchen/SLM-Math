#!/bin/bash

################################################################################
# GPU 3: Solver-Checker Stateless Agent Evaluation
# Dataset: GSM8K (500 samples)
################################################################################

set -e  # Exit on error

# Set GPU
export CUDA_VISIBLE_DEVICES=2

################################################################################
# Configuration Variables

# Model
MODEL="Qwen2.5-Math-1.5B"

# Agent method: solver_checker (stateless)
AGENT="solver_checker"

# For solver_checker: Checker model (leave empty to use same as solver)
CHECKER_MODEL=""

# For solver_checker: Max iterations per problem
MAX_ITERATIONS=5

# Test round name
ROUND_NAME="gpu3_solver_checker_stateless"

# Dataset
DATASET="gsm8k"

# Number of test cases
COUNT=500

# Detailed output
DETAILED="false"

# Resume from existing results (leave empty to start fresh)
RESUME_DIR=""

# Save interval
SAVE_INTERVAL=10

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║     GPU 3: Solver-Checker Stateless Agent Evaluation                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check model
MODEL_PATH="pretrained_models/${MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Model: $MODEL_PATH"

if [ "$AGENT" = "solver_checker" ] && [ -n "$CHECKER_MODEL" ]; then
    CHECKER_PATH="pretrained_models/${CHECKER_MODEL}"
    if [ ! -d "$CHECKER_PATH" ]; then
        echo "✗ Error: Checker model not found at $CHECKER_PATH"
        exit 1
    fi
    echo "✓ Checker Model: $CHECKER_PATH"
elif [ "$AGENT" = "solver_checker" ]; then
    echo "✓ Checker Model: Same as solver (shared model)"
fi

echo "✓ GPU: 3 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Evaluation Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL"
echo "  Agent Method: $AGENT (stateless)"
if [ "$AGENT" = "solver_checker" ]; then
    if [ -n "$CHECKER_MODEL" ]; then
        echo "  Checker Model: $CHECKER_MODEL"
    else
        echo "  Checker Model: Same as solver"
    fi
    echo "  Max Iterations: $MAX_ITERATIONS"
fi
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT"
echo "  Detailed Output: $DETAILED"
if [ -n "$RESUME_DIR" ]; then
    echo "  Resume: $RESUME_DIR"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
    --save_interval \"$SAVE_INTERVAL\""

# Add agent-specific parameters
if [ "$AGENT" = "solver_checker" ]; then
    CMD="$CMD --max_iterations \"$MAX_ITERATIONS\""
    if [ -n "$CHECKER_MODEL" ]; then
        CMD="$CMD --checker_model \"$CHECKER_MODEL\""
    fi
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
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Evaluation Complete!                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: results/${ROUND_NAME}_${MODEL}_${DATASET}_*"
echo "Analysis report: results/${ROUND_NAME}_${MODEL}_${DATASET}_*/analysis_report.txt"
echo ""

