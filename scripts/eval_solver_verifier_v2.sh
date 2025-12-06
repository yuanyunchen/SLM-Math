#!/bin/bash

################################################################################
# Solver-Verifier V2 Evaluation Script
# V2: Always verify, even when code == boxed
################################################################################

set -e

################################################################################
# Configuration

MODEL="Qwen2.5-Math-1.5B"
CHECKPOINT=""
AGENT="solver_verifier_v2"
ROUND_NAME="test_solver_verifier_v2"

# Dataset: gsm8k, math500, math
DATASET="gsm8k"

# Number of samples (0 = full dataset)
COUNT=500

# Max iterations per problem
MAX_ITERATIONS=3

DETAILED="false"
APPLY_CHAT_TEMPLATE="false"
RESUME_DIR=""
SAVE_INTERVAL=10

################################################################################
# Run

echo "=============================================================================="
echo "              Solver-Verifier V2 - Always Verify                              "
echo "=============================================================================="
echo ""
echo "Model: $MODEL"
echo "Agent: $AGENT"
echo "Dataset: $DATASET"
echo "Count: $COUNT"
echo "Max Iterations: $MAX_ITERATIONS"
echo ""

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent "$AGENT" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED" \
    --apply_chat_template "$APPLY_CHAT_TEMPLATE" \
    --save_interval "$SAVE_INTERVAL"

echo ""
echo "Done! Results: results/${ROUND_NAME}_${MODEL}_${DATASET}_*"











