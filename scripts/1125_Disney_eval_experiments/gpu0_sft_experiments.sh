#!/bin/bash

################################################################################
# Experiment: SFT Training - Two Learning Rates + LoRA 16 with High LR
# GPU: 0
# Description: Full fine-tuning Qwen2.5-Math-1.5B with 2 epochs
#              Run 1: Conservative LR (1e-5)
#              Run 2: Aggressive LR (5e-5)
#              Run 3: LoRA 16 with 5x LR (5e-4)
#              All runs: Eval at epoch 2, no checkpoint export
################################################################################

set -e  # Exit on error

################################################################################
# Common Configuration

# GPU
export CUDA_VISIBLE_DEVICES=0
GPUS="0"

# W&B
WANDB_PROJECT="1125_Disney_eval_experiments"

# Model and data
MODEL="pretrained_models/Qwen2.5-Math-1.5B"
DATA_PATH="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"

# Training config
NUM_EPOCHS=2
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=8

# Checkpoint and evaluation config
SAVE_EVERY_N_EPOCHS=999  # Effectively disable intermediate saves
EVAL_EVERY_N_EPOCHS=2    # Only eval at final (epoch 2)
EVAL_SAMPLES=500         # Samples per dataset for evaluation

################################################################################
# Run 1: Conservative Learning Rate (1e-5)

ROUND_NAME_1="sft_lr1e5"
WANDB_RUN_NAME_1="gpu0_sft_lr1e5"
LEARNING_RATE_1=1e-5

echo "=========================================="
echo "Run 1: SFT with Conservative LR"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Round Name: $ROUND_NAME_1"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME_1"
echo "Model: $MODEL"
echo "Learning Rate: $LEARNING_RATE_1"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Eval: epoch 2 only, $EVAL_SAMPLES samples"
echo "Save: disabled (eval only)"
echo "=========================================="

python models/train_sft_baseline.py \
    --mode "sft" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$ROUND_NAME_1" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE_1" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME_1" \
    --enable_eval \
    --gradient_checkpointing \
    --eval_samples "$EVAL_SAMPLES" \
    --skip_save

echo "=========================================="
echo "Run 1 Completed: Conservative LR"
echo "=========================================="

################################################################################
# Run 2: Aggressive Learning Rate (5e-5)

ROUND_NAME_2="sft_lr5e5"
WANDB_RUN_NAME_2="gpu0_sft_lr5e5"
LEARNING_RATE_2=5e-5

echo ""
echo "=========================================="
echo "Run 2: SFT with Aggressive LR"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Round Name: $ROUND_NAME_2"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME_2"
echo "Model: $MODEL"
echo "Learning Rate: $LEARNING_RATE_2"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Eval: epoch 2 only, $EVAL_SAMPLES samples"
echo "Save: disabled (eval only)"
echo "=========================================="

python models/train_sft_baseline.py \
    --mode "sft" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$ROUND_NAME_2" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE_2" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME_2" \
    --enable_eval \
    --gradient_checkpointing \
    --eval_samples "$EVAL_SAMPLES" \
    --skip_save

echo "=========================================="
echo "Run 2 Completed: Aggressive LR"
echo "=========================================="

################################################################################
# Run 3: LoRA 16 with 5x Learning Rate (5e-4)

ROUND_NAME_3="lora_r16_lr5e4"
WANDB_RUN_NAME_3="gpu0_lora_r16_lr5e4"
LEARNING_RATE_3=5e-4
LORA_RANK_3=16

echo ""
echo "=========================================="
echo "Run 3: LoRA 16 with 5x LR"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Round Name: $ROUND_NAME_3"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME_3"
echo "Model: $MODEL"
echo "LoRA Rank: $LORA_RANK_3"
echo "Learning Rate: $LEARNING_RATE_3 (5x higher)"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "Eval: epoch 2 only, $EVAL_SAMPLES samples"
echo "Save: disabled (eval only)"
echo "=========================================="

python models/train_sft_baseline.py \
    --mode "lora" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$ROUND_NAME_3" \
    --lora_rank "$LORA_RANK_3" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE_3" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME_3" \
    --enable_eval \
    --gradient_checkpointing \
    --eval_samples "$EVAL_SAMPLES" \
    --skip_save

echo "=========================================="
echo "Run 3 Completed: LoRA 16 with 5x LR"
echo "=========================================="

################################################################################
# Summary

echo ""
echo "=========================================="
echo "GPU 0: All Experiments Completed!"
echo "=========================================="
echo "Run 1 (SFT Conservative): LR=$LEARNING_RATE_1"
echo "Run 2 (SFT Aggressive):   LR=$LEARNING_RATE_2"
echo "Run 3 (LoRA 16 High LR):  LR=$LEARNING_RATE_3, Rank=$LORA_RANK_3"
echo "W&B Project: $WANDB_PROJECT"
echo "=========================================="



