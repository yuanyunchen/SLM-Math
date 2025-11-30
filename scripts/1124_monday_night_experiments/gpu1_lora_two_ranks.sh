#!/bin/bash

################################################################################
# Experiment: LoRA Training - Two Ranks
# GPU: 1
# Description: LoRA fine-tuning Qwen2.5-Math-1.5B with 2 epochs
#              Run 1: LoRA rank 16
#              Run 2: LoRA rank 32
################################################################################

set -e  # Exit on error

################################################################################
# Common Configuration

# GPU
export CUDA_VISIBLE_DEVICES=1
GPUS="1"

# W&B
WANDB_PROJECT="1124_monday_night_experiments"

# Model and data
MODEL="pretrained_models/Qwen2.5-Math-1.5B"
DATA_PATH="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"

# Training config
MODE="lora"
NUM_EPOCHS=2
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=8

# Checkpoint and evaluation config
SAVE_EVERY_N_EPOCHS=999  # Only save final model
EVAL_EVERY_N_EPOCHS=2  # Only eval at final
EVAL_SAMPLES=500  # Samples per dataset for evaluation

################################################################################
# Run 1: LoRA Rank 16

ROUND_NAME_1="lora_r16"
WANDB_RUN_NAME_1="gpu1_lora_r16"
LORA_RANK_1=16

echo "=========================================="
echo "Run 1: LoRA Rank 16"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Round Name: $ROUND_NAME_1"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME_1"
echo "Model: $MODEL"
echo "LoRA Rank: $LORA_RANK_1"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Eval: final only, $EVAL_SAMPLES samples"
echo "Save: final model only"
echo "=========================================="

python models/train_sft_baseline.py \
    --mode "$MODE" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$ROUND_NAME_1" \
    --lora_rank "$LORA_RANK_1" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME_1" \
    --enable_eval \
    --gradient_checkpointing \
    --eval_samples "$EVAL_SAMPLES"

echo "=========================================="
echo "Run 1 Completed: LoRA Rank 16"
echo "=========================================="

################################################################################
# Run 2: LoRA Rank 32

ROUND_NAME_2="lora_r32"
WANDB_RUN_NAME_2="gpu1_lora_r32"
LORA_RANK_2=32

echo ""
echo "=========================================="
echo "Run 2: LoRA Rank 32"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Round Name: $ROUND_NAME_2"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME_2"
echo "Model: $MODEL"
echo "LoRA Rank: $LORA_RANK_2"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Eval: final only, $EVAL_SAMPLES samples"
echo "Save: final model only"
echo "=========================================="

python models/train_sft_baseline.py \
    --mjhode "$MODE" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$ROUND_NAME_2" \
    --lora_rank "$LORA_RANK_2" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME_2" \
    --enable_eval \
    --gradient_checkpointing \
    --eval_samples "$EVAL_SAMPLES"

echo "=========================================="
echo "Run 2 Completed: LoRA Rank 32"
echo "=========================================="

################################################################################
# Summary

echo ""
echo "=========================================="
echo "GPU 1: All LoRA Experiments Completed!"
echo "=========================================="
echo "Run 1: LoRA Rank $LORA_RANK_1"
echo "Run 2: LoRA Rank $LORA_RANK_2"
echo "W&B Project: $WANDB_PROJECT"
echo "=========================================="

