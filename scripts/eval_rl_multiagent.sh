#!/bin/bash

################################################################################
# Quick Evaluation for RL Multi-Agent Trained Model
# Tests the trained model on GSM8K with interactive code execution
################################################################################

# Use cached models offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

################################################################################
# Configuration

# Base model (must match what was used for training)
MODEL="Qwen2.5-Math-1.5B"

# Trained LoRA checkpoint
CHECKPOINT="results/rl_multiagent_checkpoints/epoch_1"

# Agent method - interactive code execution
AGENT="solver_interactive_code"

# Test configuration
ROUND_NAME="eval_rl_multiagent"
DATASET="gsm8k"
COUNT=100               # Number of test samples

# Output settings
DETAILED="true"
APPLY_CHAT_TEMPLATE="false"
SAVE_INTERVAL=10

################################################################################
# Run Evaluation

echo "=============================================================================="
echo "          RL Multi-Agent Model Evaluation                                     "
echo "=============================================================================="
echo ""
echo "Base Model: $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET"
echo "Count: $COUNT"
echo "=============================================================================="

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --agent "$AGENT" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --detailed "$DETAILED" \
    --apply_chat_template "$APPLY_CHAT_TEMPLATE" \
    --save_interval "$SAVE_INTERVAL"

echo ""
echo "=============================================================================="
echo "                         Evaluation Complete!                                 "
echo "=============================================================================="
echo "Results saved to: results/${ROUND_NAME}_*"

