#!/bin/bash

################################################################################
# Experiment: RL Training - Two Configurations
# GPU: 2
# Description: 
#   Run 1: RL training directly from base model
#   Run 2: LoRA 16 for 1 epoch, then RL training
# RL Config: 200 total steps, eval every 200 steps
# All runs: Eval only, minimal checkpoint export
################################################################################

set -e  # Exit on error

################################################################################
# Common Configuration

# GPU
export CUDA_VISIBLE_DEVICES=2
GPUS="2"

# W&B
WANDB_PROJECT="1125_Disney_eval_experiments"

# Model and data
BASE_MODEL="pretrained_models/Qwen2.5-Math-1.5B"
DATA_PATH="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"
CONFIG_FILE="configs/rl_grpo_config.yaml"

# RL Training config
RL_BATCH_SIZE=4
RL_GRADIENT_ACCUMULATION_STEPS=4
RL_LEARNING_RATE=5e-6
RL_NUM_RETURN_SEQUENCES=2
RL_TEMPERATURE=0.7
RL_KL_COEF=0

# RL Steps config
RL_LOGGING_STEPS=10
RL_EVAL_STEPS=200
RL_SAVE_STEPS=200    # Save checkpoint at step 200
RL_TOTAL_STEPS=200   # Note: not used currently, training runs full epoch

# Evaluation config
EVAL_SAMPLES=500

################################################################################
# Run 1: RL Training from Base Model

WANDB_RUN_NAME_1="gpu2_rl_from_base"
OUTPUT_DIR_1="results/rl_from_base_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Run 1: RL Training from Base Model"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME_1"
echo "Model: $BASE_MODEL"
echo "Output: $OUTPUT_DIR_1"
echo "Batch size: $RL_BATCH_SIZE x $RL_GRADIENT_ACCUMULATION_STEPS = $(($RL_BATCH_SIZE * $RL_GRADIENT_ACCUMULATION_STEPS))"
echo "Learning rate: $RL_LEARNING_RATE"
echo "Eval steps: $RL_EVAL_STEPS"
echo "Save steps: $RL_SAVE_STEPS"
echo "Eval samples: $EVAL_SAMPLES per dataset"
echo "=========================================="

python models/train_rl_base.py \
    --config "$CONFIG_FILE" \
    --model_path "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR_1" \
    --max_samples -1 \
    --num_epochs 1 \
    --batch_size "$RL_BATCH_SIZE" \
    --gradient_accumulation_steps "$RL_GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$RL_LEARNING_RATE" \
    --num_return_sequences "$RL_NUM_RETURN_SEQUENCES" \
    --temperature "$RL_TEMPERATURE" \
    --kl_coef "$RL_KL_COEF" \
    --logging_steps "$RL_LOGGING_STEPS" \
    --eval_steps "$RL_EVAL_STEPS" \
    --save_steps "$RL_SAVE_STEPS" \
    --eval_samples "$EVAL_SAMPLES" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME_1"

echo "=========================================="
echo "Run 1 Completed: RL from Base Model"
echo "Output: $OUTPUT_DIR_1"
echo "=========================================="

################################################################################
# Run 2: LoRA 16 for 1 epoch, then RL

echo ""
echo "=========================================="
echo "Run 2: Phase 1 - LoRA 16 Training (1 epoch)"
echo "=========================================="

# Phase 1: LoRA Training (need to save for RL phase)
LORA_ROUND_NAME="lora_r16_for_rl"
LORA_WANDB_RUN="gpu2_lora_r16_for_rl"
LORA_RANK=16
LORA_EPOCHS=1
LORA_BATCH_SIZE=24
LORA_GRAD_ACCUM=4
LORA_SAVE_EVERY=1    # Save at epoch 1 for RL phase
LORA_EVAL_EVERY=999  # Skip eval in LoRA phase

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $LORA_WANDB_RUN"
echo "Model: $BASE_MODEL"
echo "LoRA Rank: $LORA_RANK"
echo "Epochs: $LORA_EPOCHS"
echo "Batch size: $LORA_BATCH_SIZE x $LORA_GRAD_ACCUM = $(($LORA_BATCH_SIZE * $LORA_GRAD_ACCUM))"
echo "Save: required for RL phase"
echo "=========================================="

python models/train_sft_baseline.py \
    --mode "lora" \
    --model_name "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$LORA_ROUND_NAME" \
    --lora_rank "$LORA_RANK" \
    --num_epochs "$LORA_EPOCHS" \
    --batch_size "$LORA_BATCH_SIZE" \
    --gradient_accumulation_steps "$LORA_GRAD_ACCUM" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$LORA_SAVE_EVERY" \
    --eval_every_n_epochs "$LORA_EVAL_EVERY" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$LORA_WANDB_RUN" \
    --gradient_checkpointing

echo "=========================================="
echo "Phase 1 Completed: LoRA 16 Training"
echo "=========================================="

################################################################################
# Find the LoRA checkpoint

LORA_CHECKPOINT_DIR=$(ls -td checkpoints/${LORA_ROUND_NAME}_lora_r${LORA_RANK}_* 2>/dev/null | head -1)

if [ -z "$LORA_CHECKPOINT_DIR" ]; then
    echo "Error: Could not find LoRA checkpoint directory"
    exit 1
fi

LORA_FINAL_MODEL="${LORA_CHECKPOINT_DIR}/final_model"

if [ ! -d "$LORA_FINAL_MODEL" ]; then
    echo "Error: LoRA final model not found at $LORA_FINAL_MODEL"
    exit 1
fi

echo "Found LoRA checkpoint: $LORA_FINAL_MODEL"

################################################################################
# Phase 2: RL Training on LoRA Checkpoint

WANDB_RUN_NAME_2="gpu2_rl_on_lora"
OUTPUT_DIR_2="results/rl_on_lora_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=========================================="
echo "Run 2: Phase 2 - RL Training on LoRA Checkpoint"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME_2"
echo "LoRA Checkpoint: $LORA_FINAL_MODEL"
echo "Output: $OUTPUT_DIR_2"
echo "Batch size: $RL_BATCH_SIZE x $RL_GRADIENT_ACCUMULATION_STEPS = $(($RL_BATCH_SIZE * $RL_GRADIENT_ACCUMULATION_STEPS))"
echo "Learning rate: $RL_LEARNING_RATE"
echo "Eval steps: $RL_EVAL_STEPS"
echo "Save steps: $RL_SAVE_STEPS"
echo "Eval samples: $EVAL_SAMPLES per dataset"
echo "=========================================="

python models/train_rl_base.py \
    --config "$CONFIG_FILE" \
    --model_path "$LORA_FINAL_MODEL" \
    --output_dir "$OUTPUT_DIR_2" \
    --max_samples -1 \
    --num_epochs 1 \
    --batch_size "$RL_BATCH_SIZE" \
    --gradient_accumulation_steps "$RL_GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$RL_LEARNING_RATE" \
    --num_return_sequences "$RL_NUM_RETURN_SEQUENCES" \
    --temperature "$RL_TEMPERATURE" \
    --kl_coef "$RL_KL_COEF" \
    --logging_steps "$RL_LOGGING_STEPS" \
    --eval_steps "$RL_EVAL_STEPS" \
    --save_steps "$RL_SAVE_STEPS" \
    --eval_samples "$EVAL_SAMPLES" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME_2"

echo "=========================================="
echo "Run 2 Completed: RL on LoRA Checkpoint"
echo "Output: $OUTPUT_DIR_2"
echo "=========================================="

################################################################################
# Cleanup: Remove LoRA checkpoint after RL training (optional)

echo ""
echo "Cleaning up LoRA checkpoint..."
rm -rf "$LORA_CHECKPOINT_DIR"
echo "Removed: $LORA_CHECKPOINT_DIR"

################################################################################
# Summary

echo ""
echo "=========================================="
echo "GPU 2: All RL Experiments Completed!"
echo "=========================================="
echo "Run 1 (RL from Base): $OUTPUT_DIR_1"
echo "Run 2 (LoRA then RL): $OUTPUT_DIR_2"
echo "W&B Project: $WANDB_PROJECT"
echo "=========================================="


