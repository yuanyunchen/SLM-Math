#!/bin/bash

################################################################################
# Test Optimized Agents
# 测试优化后的 solver-checker-summarizer agents
################################################################################

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║            Testing Optimized Solver-Checker-Summarizer Agents               ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
MODEL="Qwen2.5-Math-1.5B"
DATASET="gsm8k"
COUNT=5
MAX_ITERATIONS=2
DETAILED="true"

# Test 1: Stateless mode (solver_checker_summarizer.py)
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "TEST 1: Stateless Mode (solver_checker_summarizer.py)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent solver_checker_summarizer \
    --round test_stateless_optimized \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

echo ""
echo "✓ Test 1 completed"
echo ""

# Test 2: Chat mode (solver_checker_summarizer_chat.py)
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "TEST 2: Chat Mode (solver_checker_summarizer_chat.py)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent solver_checker_summarizer_chat \
    --round test_chat_optimized \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

echo ""
echo "✓ Test 2 completed"
echo ""

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                            TEST SUMMARY                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Both agents tested successfully!"
echo ""
echo "Results saved to:"
echo "  - results/test_stateless_optimized_*/"
echo "  - results/test_chat_optimized_*/"
echo ""
echo "Please manually check the following files:"
echo "  - logging.log  : Full execution log"
echo "  - metrics.csv  : Performance metrics"
echo "  - summary.txt  : Human-readable summary"
echo "  - answer.json  : Detailed predictions"
echo ""


