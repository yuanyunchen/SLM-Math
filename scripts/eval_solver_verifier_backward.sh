#!/bin/bash

################################################################################
# Run solver + backward verifier (single-pass) on a dataset
# Uses agent/solver_verifier_backward.py via evaluation.eval_agent
################################################################################

set -e

# Model and checkpoint
MODEL="Qwen2.5-Math-1.5B"
CHECKPOINT=""
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Round name and dataset
ROUND_NAME="test_solver_verifier_backward"
DATASET="gsm8k"
COUNT=500   # 0 means full dataset

# Agent (new backward verifier wrapper)
AGENT="solver_verifier_backward"

# Other options
BATCH_SIZE=1
DETAILED="true"
APPLY_CHAT_TEMPLATE="false"
SAVE_INTERVAL=10

echo "=========================================="
echo "Solver + Backward Verifier (single-pass)"
echo "=========================================="
echo "Model: $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Round: $ROUND_NAME"
echo "Dataset: $DATASET"
echo "Count: $COUNT"
echo "Agent: $AGENT"
echo "Batch Size: $BATCH_SIZE"
echo "Chat Template: $APPLY_CHAT_TEMPLATE"
echo "=========================================="

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --agent "$AGENT" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED" \
    --apply_chat_template "$APPLY_CHAT_TEMPLATE" \
    --save_interval "$SAVE_INTERVAL"

echo ""
echo "Done. Results saved under results/${ROUND_NAME}_*/"


