#!/bin/bash

################################################################################
# Evaluation: LoRA r16 checkpoint-298 on MATH500
# GPU: 1
# Checkpoint: checkpoints/lora_r16_lora_r16_20251125_164902/checkpoint-298
################################################################################

set -e  # Exit on error

################################################################################
# Configuration

# GPU
export CUDA_VISIBLE_DEVICES=1

# Model base
BASE_MODEL="Qwen2.5-Math-1.5B"

# Checkpoint
CHECKPOINT="checkpoints/lora_r16_lora_r16_20251125_164902/checkpoint-298"

# Evaluation config
BATCH_SIZE=1
MODE="standard"
DETAILED="true"
COUNT=0  # 0 = full dataset

################################################################################
# Evaluation
################################################################################

echo "=========================================="
echo "Evaluating LoRA r16 checkpoint-298 on MATH500"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $BASE_MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: math500 (full)"
echo "=========================================="

python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --round "1125_milestone_lora_r16_ckpt298" \
    --dataset "math500" \
    --count "$COUNT" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED"

echo ""
echo "=========================================="
echo "Evaluation Completed!"
echo "=========================================="


