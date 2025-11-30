#!/bin/bash

################################################################################
# Evaluation: SFT Checkpoints from 1124 Monday Night Experiments
# GPU: 0
# Description: Evaluate SFT checkpoints on GSM8K and MATH500
#              Checkpoint 1: SFT lr=1e-5
#              Checkpoint 2: SFT lr=5e-5
# Order: All GSM8K first, then all MATH500
################################################################################

set -e  # Exit on error

################################################################################
# Common Configuration

# GPU
export CUDA_VISIBLE_DEVICES=0

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
CHECKPOINT_SFT_LR1E5="checkpoints/sft_lr1e5_sft_20251125_162701/final_model"
CHECKPOINT_SFT_LR5E5="checkpoints/sft_lr5e5_sft_20251125_190347/final_model"

################################################################################
# Phase 1: All GSM8K Evaluations
################################################################################

echo "=========================================="
echo "Phase 1: GSM8K Evaluations (${GSM8K_COUNT} samples)"
echo "=========================================="

# Checkpoint 1: SFT lr=1e-5 on GSM8K
echo ""
echo ">> [1/2] SFT lr=1e-5 on GSM8K..."
python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT_SFT_LR1E5" \
    --round "1125_disney_sft_lr1e5" \
    --dataset "gsm8k" \
    --count "$GSM8K_COUNT" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED"
echo ">> [1/2] Completed"

# Checkpoint 2: SFT lr=5e-5 on GSM8K
echo ""
echo ">> [2/2] SFT lr=5e-5 on GSM8K..."
python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT_SFT_LR5E5" \
    --round "1125_disney_sft_lr5e5" \
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

# Checkpoint 1: SFT lr=1e-5 on MATH500
echo ""
echo ">> [1/2] SFT lr=1e-5 on MATH500..."
python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT_SFT_LR1E5" \
    --round "1125_disney_sft_lr1e5" \
    --dataset "math500" \
    --count "$MATH500_COUNT" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED"
echo ">> [1/2] Completed"

# Checkpoint 2: SFT lr=5e-5 on MATH500
echo ""
echo ">> [2/2] SFT lr=5e-5 on MATH500..."
python -m evaluation.eval \
    --model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT_SFT_LR5E5" \
    --round "1125_disney_sft_lr5e5" \
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
echo "GPU 0: All SFT Checkpoint Evaluations Completed!"
echo "=========================================="
echo "Phase 1 (GSM8K ${GSM8K_COUNT}): SFT lr=1e-5, SFT lr=5e-5"
echo "Phase 2 (MATH500):              SFT lr=1e-5, SFT lr=5e-5"
echo "Results saved to: results/<round>_<model>_<dataset>_*/answers/*.json"
echo "=========================================="
