#!/bin/bash

################################################################################
# Experiment: LoRA SFT -> RL Pipeline
# GPU: 2
# Description: 
#   Phase 1: LoRA SFT training (5 epochs, eval every 1 epoch, save every 2 epochs)
#   Phase 2: RL training on LoRA checkpoint
################################################################################

set -e  # Exit on error

################################################################################
# Configuration

# GPU
export CUDA_VISIBLE_DEVICES=2
GPUS="2"

# Experiment naming
WANDB_PROJECT="slm_math_experiments_1124"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Model and data
BASE_MODEL="pretrained_models/Qwen2.5-Math-1.5B"
DATA_PATH="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"
CONFIG_FILE="configs/rl_grpo_config.yaml"

################################################################################
# Phase 1: LoRA SFT Training

PHASE1_ROUND_NAME="exp_lora_sft"
PHASE1_WANDB_RUN="gpu2_lora_sft_5ep"
PHASE1_EPOCHS=5
PHASE1_BATCH_SIZE=24
PHASE1_GRAD_ACCUM=4
PHASE1_SAVE_EVERY=999  # Only save final model (set higher than total epochs)
PHASE1_EVAL_EVERY=1
PHASE1_EVAL_SAMPLES=200  # Samples per dataset for evaluation
LORA_RANK=16

echo "=========================================="
echo "Phase 1: LoRA SFT Training"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $PHASE1_WANDB_RUN"
echo "Model: $BASE_MODEL"
echo "LoRA Rank: $LORA_RANK"
echo "Epochs: $PHASE1_EPOCHS"
echo "Eval every: $PHASE1_EVAL_EVERY epoch(s)"
echo "Save: final model only"
echo "=========================================="

python models/train_sft_baseline.py \
    --mode "lora" \
    --model_name "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$PHASE1_ROUND_NAME" \
    --lora_rank "$LORA_RANK" \
    --num_epochs "$PHASE1_EPOCHS" \
    --batch_size "$PHASE1_BATCH_SIZE" \
    --gradient_accumulation_steps "$PHASE1_GRAD_ACCUM" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$PHASE1_SAVE_EVERY" \
    --eval_every_n_epochs "$PHASE1_EVAL_EVERY" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$PHASE1_WANDB_RUN" \
    --enable_eval \
    --gradient_checkpointing \
    --eval_samples "$PHASE1_EVAL_SAMPLES"

echo "=========================================="
echo "Phase 1: LoRA SFT Training Completed!"
echo "=========================================="

################################################################################
# Find the latest LoRA checkpoint

# Find the most recent checkpoint directory
LORA_CHECKPOINT_DIR=$(ls -td checkpoints/${PHASE1_ROUND_NAME}_lora_r${LORA_RANK}_* 2>/dev/null | head -1)

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

PHASE2_WANDB_RUN="gpu2_rl_on_lora"
PHASE2_OUTPUT_DIR="results/exp_rl_on_lora_${TIMESTAMP}"
PHASE2_EPOCHS=3
PHASE2_LOGGING_STEPS=10
PHASE2_EVAL_STEPS=100
PHASE2_SAVE_STEPS=300

# RL-specific parameters
PHASE2_BATCH_SIZE=4
PHASE2_GRAD_ACCUM=4
PHASE2_LEARNING_RATE=5e-6
PHASE2_NUM_RETURN_SEQUENCES=2
PHASE2_TEMPERATURE=0.7
# KL coefficient: Set to 0 to disable (saves ~3x speed and ~3GB VRAM)
PHASE2_KL_COEF=0
PHASE2_EVAL_SAMPLES=200

echo ""
echo "=========================================="
echo "Phase 2: RL Training on LoRA Checkpoint"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $PHASE2_WANDB_RUN"
echo "LoRA Checkpoint: $LORA_FINAL_MODEL"
echo "Output: $PHASE2_OUTPUT_DIR"
echo "Epochs: $PHASE2_EPOCHS"
echo "Batch size: $PHASE2_BATCH_SIZE x $PHASE2_GRAD_ACCUM = $(($PHASE2_BATCH_SIZE * $PHASE2_GRAD_ACCUM))"
echo "Learning rate: $PHASE2_LEARNING_RATE"
echo "Num return sequences: $PHASE2_NUM_RETURN_SEQUENCES"
echo "Temperature: $PHASE2_TEMPERATURE"
if [ "$PHASE2_KL_COEF" = "0" ]; then
    echo "KL Penalty: Disabled (faster training)"
else
    echo "KL Penalty: Enabled (kl_coef=$PHASE2_KL_COEF)"
fi
echo "Logging steps: $PHASE2_LOGGING_STEPS"
echo "Eval steps: $PHASE2_EVAL_STEPS"
echo "Save steps: $PHASE2_SAVE_STEPS"
echo "Eval samples: $PHASE2_EVAL_SAMPLES per dataset"
echo "=========================================="

python models/train_rl_base.py \
    --config "$CONFIG_FILE" \
    --model_path "$LORA_FINAL_MODEL" \
    --output_dir "$PHASE2_OUTPUT_DIR" \
    --max_samples -1 \
    --num_epochs "$PHASE2_EPOCHS" \
    --batch_size "$PHASE2_BATCH_SIZE" \
    --gradient_accumulation_steps "$PHASE2_GRAD_ACCUM" \
    --learning_rate "$PHASE2_LEARNING_RATE" \
    --num_return_sequences "$PHASE2_NUM_RETURN_SEQUENCES" \
    --temperature "$PHASE2_TEMPERATURE" \
    --kl_coef "$PHASE2_KL_COEF" \
    --logging_steps "$PHASE2_LOGGING_STEPS" \
    --eval_steps "$PHASE2_EVAL_STEPS" \
    --save_steps "$PHASE2_SAVE_STEPS" \
    --eval_samples "$PHASE2_EVAL_SAMPLES" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$PHASE2_WANDB_RUN"

echo "=========================================="
echo "Phase 2: RL Training Completed!"
echo "Output: $PHASE2_OUTPUT_DIR"
echo "=========================================="

################################################################################
# Summary

echo ""
echo "=========================================="
echo "GPU 2: Full Pipeline Completed!"
echo "=========================================="
echo "Phase 1 (LoRA SFT): $LORA_CHECKPOINT_DIR"
echo "Phase 2 (RL): $PHASE2_OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "=========================================="

