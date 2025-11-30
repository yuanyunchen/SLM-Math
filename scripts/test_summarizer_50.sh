#!/bin/bash

################################################################################
# Test Solver-Checker-Summarizer Workflow - 50 samples
# For manual analysis and optimization
################################################################################

set -e

echo "================================================================================"
echo "  Solver-Checker-Summarizer Test - 50 Samples"
echo "================================================================================"
echo ""

# Configuration
MODEL="Qwen2.5-Math-1.5B"
DATASET="gsm8k"
COUNT=50
MAX_ITERATIONS=3
DETAILED="true"
CUDA_DEVICE=0
ROUND_NAME="summarizer_test"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Samples: $COUNT"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  GPU: $CUDA_DEVICE"
echo ""

# Export GPU settings
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "================================================================================"
echo "Running solver_checker_summarizer"
echo "================================================================================"
echo ""

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent solver_checker_summarizer \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

echo ""
echo "Test completed!"
echo ""

# Find results directory
RESULT_DIR=$(ls -td results/${ROUND_NAME}_* 2>/dev/null | head -1)

if [ -n "$RESULT_DIR" ]; then
    echo "================================================================================"
    echo "Results Location: $RESULT_DIR"
    echo "================================================================================"
    echo ""
    echo "Key files:"
    echo "  - $RESULT_DIR/metrics.csv"
    echo "  - $RESULT_DIR/analysis_report.txt"
    echo "  - $RESULT_DIR/answers/*.json"
    echo ""
    
    # Display metrics
    if [ -f "$RESULT_DIR/metrics.csv" ]; then
        echo "Metrics:"
        cat "$RESULT_DIR/metrics.csv"
        echo ""
    fi
fi

echo "================================================================================"








