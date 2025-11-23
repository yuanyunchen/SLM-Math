#!/bin/bash

################################################################################
# Multi-Agent Evaluation Script
#
# This script runs the multi-agent (Solver-Checker) evaluation workflow.
# The multi-agent approach uses iterative feedback between:
#   - Solver: Generates the answer
#   - Checker: Verifies the answer and provides feedback
#
################################################################################

# Model to evaluate (Solver)
# Options: Qwen2.5-Math-1.5B, Qwen2.5-Math-1.5B-Instruct, Qwen3-0.6B, etc.
MODEL="Qwen2.5-Math-1.5B"

# Test round name (used for organizing results)
# Results will be saved to: results/<ROUND_NAME>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/
ROUND_NAME="testMultiAgent"

# Dataset to use
# Options: gsm8k (grade school math), math (competition math)
DATASET="gsm8k"

# Number of test cases to run (set to 0 to run the entire dataset)
COUNT=20

# Evaluation mode - MUST be multi_agent for multi-agent workflow
MODE="multi_agent"

# Detailed output mode
# If true, logs will include full solver-checker conversations for each iteration
DETAILED="true"

# Optional: Use a different model for checker
# If not specified, the checker will use the same model as the solver
# Uncomment the line below to use a different checker model
# CHECKER_MODEL="Qwen2.5-Math-1.5B-Instruct"

################################################################################
# Run evaluation

# Go to project root (assuming this script is in agent/ directory)
cd "$(dirname "$0")/.."

# Run the evaluation pipeline
if [ -z "$CHECKER_MODEL" ]; then
    # Use same model for both solver and checker
    python -m evaluation.eval_pipeline \
        --model "$MODEL" \
        --round "$ROUND_NAME" \
        --dataset "$DATASET" \
        --count "$COUNT" \
        --mode "$MODE" \
        --detailed "$DETAILED"
else
    # Use different models for solver and checker
    python -m evaluation.eval_pipeline \
        --model "$MODEL" \
        --round "$ROUND_NAME" \
        --dataset "$DATASET" \
        --count "$COUNT" \
        --mode "$MODE" \
        --detailed "$DETAILED" \
        --checker_model "$CHECKER_MODEL"
fi

################################################################################
# After evaluation completes, run analysis

echo ""
echo "================================================================================"
echo "Evaluation complete! Running analysis..."
echo "================================================================================"
echo ""

# Analyze the results
python scripts/analyze_results.py

echo ""
echo "================================================================================"
echo "Analysis complete! Check the summary/ directory for detailed CSV reports."
echo "================================================================================"
echo ""

