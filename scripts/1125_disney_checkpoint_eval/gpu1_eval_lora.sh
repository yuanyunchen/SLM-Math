#!/bin/bash

################################################################################
# Evaluation: LoRA Checkpoints from 1124 Monday Night Experiments
# GPU: 1
# Description: Evaluate LoRA checkpoints on GSM8K and MATH500
#              Checkpoint 1: LoRA rank=16
#              Checkpoint 2: LoRA rank=32
# Order: All GSM8K first, then all MATH500
################################################################################

set -e  # Exit on error

################################################################################
# Common Configuration

# GPU
export CUDA_VISIBLE_DEVICES=1

# Model base
BASE_MODEL="Qwen2.5-Math-1.5B"

# Evaluation config
BATCH_SIZE=1
MODE="standard"
DETAILED="true"

# Sample counts
GSM8K_COUNT=500
MATH500_COUNT=0  # 0 = full dataset (500 samples)

# Checkpoints from 1124 experiments
CHECKPOINT_LORA_R16="checkpoints/lora_r16_lora_r16_20251125_164902/final_model"
CHECKPOINT_LORA_R32="checkpoints/lora_r32_lora_r32_20251125_190336/final_model"

################################################################################
# Phase 1: All GSM8K Evaluations
################################################################################

echo "=========================================="
echo "Phase 1: GSM8K Evaluations (${GSM8K_COUNT} samples)"
echo "=========================================="

# Checkpoint 1: LoRA r=16 on GSM8K
echo ""
echo ">> [1/2] LoRA r=16 on GSM8K..."
python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT_LORA_R16" \
    --round "1125_disney_lora_r16" \
    --dataset "gsm8k" \
    --count "$GSM8K_COUNT" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED"
echo ">> [1/2] Completed"

# Checkpoint 2: LoRA r=32 on GSM8K
echo ""
echo ">> [2/2] LoRA r=32 on GSM8K..."
python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT_LORA_R32" \
    --round "1125_disney_lora_r32" \
    --dataset "gsm8k" \
    --count "$GSM8K_COUNT" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED"
echo ">> [2/2] Completed"

echo ""
echo "=========================================="
echo "Phase 1 Completed: All GSM8K Evaluations Done"
echo "=========================================="

################################################################################
# Phase 2: All MATH500 Evaluations
################################################################################

echo ""
echo "=========================================="
echo "Phase 2: MATH500 Evaluations (full dataset)"
echo "=========================================="

# Checkpoint 1: LoRA r=16 on MATH500
echo ""
echo ">> [1/2] LoRA r=16 on MATH500..."
python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT_LORA_R16" \
    --round "1125_disney_lora_r16" \
    --dataset "math500" \
    --count "$MATH500_COUNT" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED"
echo ">> [1/2] Completed"

# Checkpoint 2: LoRA r=32 on MATH500
echo ""
echo ">> [2/2] LoRA r=32 on MATH500..."
python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT_LORA_R32" \
    --round "1125_disney_lora_r32" \
    --dataset "math500" \
    --count "$MATH500_COUNT" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED"
echo ">> [2/2] Completed"

echo ""
echo "=========================================="
echo "Phase 2 Completed: All MATH500 Evaluations Done"
echo "=========================================="

################################################################################
# Summary

echo ""
echo "=========================================="
echo "GPU 1: All LoRA Checkpoint Evaluations Completed!"
echo "=========================================="
echo "Phase 1 (GSM8K ${GSM8K_COUNT}): LoRA r=16, LoRA r=32"
echo "Phase 2 (MATH500):              LoRA r=16, LoRA r=32"
echo "Results saved to: results/<round>_<model>_<dataset>_*/answers/*.json"
echo "=========================================="
