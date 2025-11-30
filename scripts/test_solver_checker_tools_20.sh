#!/bin/bash

################################################################################
# Test Solver-Checker with Tools - 20 samples
# Testing code execution capability for math problem solving
################################################################################

# Model to evaluate
MODEL="Qwen2.5-Math-1.5B"

# Test round name
ROUND_NAME="test_tools_20"

# Dataset
DATASET="gsm8k"

# Number of test cases
COUNT=20

# Agent method
AGENT="solver_checker_with_tools"

# Max iterations
MAX_ITERATIONS=3

# Tool settings
ENABLE_SOLVER_TOOLS="true"
ENABLE_CHECKER_TOOLS="true"

# Detailed output (for debugging)
DETAILED="true"

################################################################################
# Run evaluation

# Set GPU
export CUDA_VISIBLE_DEVICES=4

echo "=============================================="
echo "Test: Solver-Checker with Tools"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Samples: $COUNT"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Solver Tools: $ENABLE_SOLVER_TOOLS"
echo "Checker Tools: $ENABLE_CHECKER_TOOLS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

cd /root/autodl-tmp/SLM-Math

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent "$AGENT" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --enable_solver_tools "$ENABLE_SOLVER_TOOLS" \
    --enable_checker_tools "$ENABLE_CHECKER_TOOLS" \
    --detailed "$DETAILED"








