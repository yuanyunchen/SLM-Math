#!/bin/bash

################################################################################
# Comparison Test: Baseline vs Agents
# Compare first-round generation of baseline model with each agent's first try
################################################################################

# Model to test
MODEL="Qwen2.5-Math-1.5B"

# Dataset to use
DATASET="gsm8k"

# Number of samples to test
COUNT=50

# GPU IDs (comma-separated)
# Use 2 GPUs for parallel baseline/agent testing
GPUS="0,1"

################################################################################
# Run comparison test

echo "========================================"
echo "Running Baseline vs Agents Comparison"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Count: $COUNT"
echo "GPUs: $GPUS"
echo "========================================"

cd /root/autodl-tmp/SLM-Math

python -m scripts.compare_baseline_agents \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --gpus "$GPUS"

echo ""
echo "Comparison complete!"








