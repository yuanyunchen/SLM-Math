#!/bin/bash

################################################################################
# Experiment: RL Training 50 steps from SFT lr=5e-5 checkpoint
# GPU: 2
# Base Model: checkpoints/sft_lr5e5_sft_20251125_190347/final_model
# Description: Run 50 steps of RL training, export checkpoint, run evaluation
################################################################################

set -e  # Exit on error

################################################################################
# Configuration

# GPU
export CUDA_VISIBLE_DEVICES=2

# W&B
WANDB_PROJECT="1125_milestone_last"

# Base model (SFT lr=5e-5 checkpoint)
BASE_MODEL="checkpoints/sft_lr5e5_sft_20251125_190347/final_model"
CONFIG_FILE="configs/rl_grpo_config.yaml"

# RL Training config
RL_BATCH_SIZE=4
RL_GRADIENT_ACCUMULATION_STEPS=4
RL_LEARNING_RATE=5e-6
RL_NUM_RETURN_SEQUENCES=2
RL_TEMPERATURE=0.7
RL_KL_COEF=0

# Steps config
RL_MAX_STEPS=50      # Only run 50 steps
RL_LOGGING_STEPS=10
RL_EVAL_STEPS=50     # Eval at step 50
RL_SAVE_STEPS=50     # Save at step 50

# Evaluation config
EVAL_SAMPLES=500

# Output
WANDB_RUN_NAME="gpu2_rl_50steps_sft_lr5e5"
OUTPUT_DIR="results/rl_50steps_sft_lr5e5_$(date +%Y%m%d_%H%M%S)"

################################################################################
# Training
################################################################################

echo "=========================================="
echo "RL Training: 50 steps from SFT lr=5e-5"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME"
echo "Base Model: $BASE_MODEL"
echo "Output: $OUTPUT_DIR"
echo "Max Steps: $RL_MAX_STEPS"
echo "Batch size: $RL_BATCH_SIZE x $RL_GRADIENT_ACCUMULATION_STEPS = $(($RL_BATCH_SIZE * $RL_GRADIENT_ACCUMULATION_STEPS))"
echo "Learning rate: $RL_LEARNING_RATE"
echo "Eval steps: $RL_EVAL_STEPS"
echo "Save steps: $RL_SAVE_STEPS"
echo "Eval samples: $EVAL_SAMPLES per dataset"
echo "=========================================="

python models/train_rl_base.py \
    --config "$CONFIG_FILE" \
    --model_path "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples -1 \
    --num_epochs 1 \
    --max_steps "$RL_MAX_STEPS" \
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
    --wandb_run_name "$WANDB_RUN_NAME"

################################################################################
# Summary
################################################################################

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo "Base Model: $BASE_MODEL"
echo "Output: $OUTPUT_DIR"
echo "Steps: $RL_MAX_STEPS"
echo "Checkpoint saved at: $OUTPUT_DIR/checkpoint-$RL_MAX_STEPS"
echo "W&B Project: $WANDB_PROJECT"
echo "=========================================="


