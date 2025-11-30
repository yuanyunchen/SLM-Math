#!/bin/bash

################################################################################
# Experiment: LoRA 16 Training with Training-Time Evaluation
# GPU: 1
# Description: LoRA rank=16, batch 16*4, 1 epoch, export checkpoint
#              Training-time eval on GSM8K and MATH500 at epoch end
################################################################################

set -e  # Exit on error

################################################################################
# Common Configuration

# GPU
export CUDA_VISIBLE_DEVICES=0
GPUS="1"

# W&B
WANDB_PROJECT="1125_Disney_eval_experiments"

# Model and data
MODEL="pretrained_models/Qwen2.5-Math-1.5B"
DATA_PATH="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"

# Training config
MODE="lora"
LORA_RANK=16
NUM_EPOCHS=1
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4

# Checkpoint and evaluation config
SAVE_EVERY_N_EPOCHS=1    # Save after epoch 1
EVAL_EVERY_N_EPOCHS=1    # Training-time eval at epoch 1
EVAL_SAMPLES=500         # Samples per dataset for training-time eval

# Round name
ROUND_NAME="lora_r16_1ep"
WANDB_RUN_NAME="gpu0_lora_r16_1ep"

################################################################################
# Training with Training-Time Evaluation
################################################################################

echo "=========================================="
echo "LoRA 16 Training with Training-Time Eval"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Round Name: $ROUND_NAME"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME"
echo "Model: $MODEL"
echo "LoRA Rank: $LORA_RANK"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Eval: every $EVAL_EVERY_N_EPOCHS epoch(s), $EVAL_SAMPLES samples per dataset"
echo "Save: every $SAVE_EVERY_N_EPOCHS epoch(s)"
echo "=========================================="

python models/train_sft_baseline.py \
    --mode "$MODE" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$ROUND_NAME" \
    --lora_rank "$LORA_RANK" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --enable_eval \
    --gradient_checkpointing \
    --eval_samples "$EVAL_SAMPLES"

################################################################################
# Summary
################################################################################

echo ""
echo "=========================================="
echo "GPU 1: Training Completed!"
echo "=========================================="
echo "Training:"
echo "  - LoRA rank=$LORA_RANK, $NUM_EPOCHS epoch(s)"
echo "  - Batch: ${BATCH_SIZE}x${GRADIENT_ACCUMULATION_STEPS} = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Training-Time Eval:"
echo "  - GSM8K: $EVAL_SAMPLES samples"
echo "  - MATH500: $EVAL_SAMPLES samples"
echo "W&B Project: $WANDB_PROJECT"
echo "=========================================="
