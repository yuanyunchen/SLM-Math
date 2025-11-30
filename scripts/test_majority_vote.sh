#!/bin/bash

################################################################################
# Majority Vote Agent Test Script
# Test majority_vote agent with Qwen2.5-Math-1.5B on GSM8K (20 samples)
################################################################################

# Model to evaluate
MODEL="Qwen2.5-Math-1.5B"

# Test round name
ROUND_NAME="testMajorityVote"

# Dataset
DATASET="gsm8k"

# Number of test cases
COUNT=20

# Agent method
AGENT="majority_vote"

# Majority vote specific parameters
NUM_RUNS=5
TEMPERATURE=0.7
TOP_P=0.95

# Detailed output
DETAILED="true"

# GPU selection
export CUDA_VISIBLE_DEVICES=5

################################################################################
# Run evaluation

cd /root/autodl-tmp/SLM-Math

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent "$AGENT" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --num_runs "$NUM_RUNS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --detailed "$DETAILED"








