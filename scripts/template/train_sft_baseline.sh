#!/bin/bash

################################################################################
# SFT Baseline Training Script (Unified Template)
# Supports both Full Fine-tuning and LoRA modes
# 
# Usage:
#   1. Copy this file: cp template/train_sft_baseline.sh train_my_exp.sh
#   2. Edit the configuration variables below
#   3. Run: bash train_my_exp.sh
#
################################################################################

################################################################################
# Training Settings

# Round name (custom identifier for this training run)
ROUND_NAME="sft_experiment"

# Model to train
MODEL="pretrained_models/Qwen2.5-Math-1.5B"

# Training data path
DATA_PATH="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"

# Training mode
# Options:
#   - "sft"  : Full fine-tuning (all parameters)
#   - "lora" : LoRA fine-tuning (parameter efficient)
MODE="lora"

# Number of training epochs
NUM_EPOCHS=5

# Learning rate (used for both modes)
# Recommended: 1e-5 for full SFT, 2e-4 for LoRA
LEARNING_RATE=""  # Leave empty to use default based on MODE

################################################################################
# LoRA Settings (only used when MODE="lora")

# LoRA rank (higher = more capacity, more memory)
# Options: 8, 16, 32, 64
LORA_RANK=64

################################################################################
# Batch Settings

# Batch size per GPU
# Recommended: 16-18 for LoRA, 8-16 for full SFT (depends on GPU memory)
BATCH_SIZE=18

# Gradient accumulation steps
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
GRADIENT_ACCUMULATION_STEPS=8

################################################################################
# Checkpoint & Evaluation Settings

# Save checkpoint every N epochs (999 = only save final model)
SAVE_EVERY_N_EPOCHS=999

# Evaluation settings
EVAL_EVERY_N_EPOCHS=1
EVAL_SAMPLES=200  # Samples per dataset for evaluation

# Enable evaluation during training
ENABLE_EVAL="--enable_eval"

################################################################################
# GPU Settings

# GPU to use (set CUDA_VISIBLE_DEVICES and GPUS count)
export CUDA_VISIBLE_DEVICES=0
GPUS="1"

################################################################################
# Weights & Biases Settings

# Enable Weights & Biases logging (comment out USE_WANDB to disable)
USE_WANDB="--use_wandb"

# W&B project name
WANDB_PROJECT="slm_math_experiments"

# W&B run name (leave empty for auto-generated name)
WANDB_RUN_NAME=""

################################################################################
# Derived Settings (auto-configured based on MODE)

# Set default learning rate based on mode if not specified
if [ -z "$LEARNING_RATE" ]; then
    if [ "$MODE" = "lora" ]; then
        LEARNING_RATE="2e-4"
    else
        LEARNING_RATE="1e-5"
    fi
fi

# Set display name based on mode
if [ "$MODE" = "lora" ]; then
    MODE_DISPLAY="LoRA (rank $LORA_RANK)"
    OUTPUT_SUFFIX="lora_r${LORA_RANK}"
else
    MODE_DISPLAY="Full Fine-tuning"
    OUTPUT_SUFFIX="sft"
fi

################################################################################
# Run Training

echo "=========================================="
echo "Starting SFT Baseline Training"
echo "=========================================="
echo "Round Name:     $ROUND_NAME"
echo "Model:          $MODEL"
echo "Data:           $DATA_PATH"
echo "Mode:           $MODE_DISPLAY"
echo "Epochs:         $NUM_EPOCHS"
echo "Learning Rate:  $LEARNING_RATE"
echo "Batch Size:     $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "GPU:            $CUDA_VISIBLE_DEVICES"
echo "Eval Every:     $EVAL_EVERY_N_EPOCHS epoch(s)"
echo "Eval Samples:   $EVAL_SAMPLES per dataset"
echo "Save:           final model only"
echo ""
echo "Output directories:"
echo "  Checkpoints:  checkpoints/${ROUND_NAME}_${OUTPUT_SUFFIX}_<timestamp>/"
echo "  Logs:         logs/${ROUND_NAME}_${OUTPUT_SUFFIX}_<timestamp>/"
echo "=========================================="

# Build wandb arguments
WANDB_ARGS=""
if [ -n "$USE_WANDB" ]; then
    WANDB_ARGS="$USE_WANDB --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

# Build LoRA arguments (only if MODE="lora")
LORA_ARGS=""
if [ "$MODE" = "lora" ]; then
    LORA_ARGS="--lora_rank $LORA_RANK"
fi

# Run training
python models/train_sft_baseline.py \
    --mode "$MODE" \
    --model_name "$MODEL" \
    --data_path "$DATA_PATH" \
    --round_name "$ROUND_NAME" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --eval_every_n_epochs "$EVAL_EVERY_N_EPOCHS" \
    --eval_samples "$EVAL_SAMPLES" \
    --gradient_checkpointing \
    $LORA_ARGS \
    $WANDB_ARGS \
    $ENABLE_EVAL

################################################################################
# Completion Summary

echo "=========================================="
echo "Training completed!"
echo ""
echo "Output files:"
echo "  Checkpoints: checkpoints/${ROUND_NAME}_${OUTPUT_SUFFIX}_<timestamp>/"
echo "    - final_model/: Final trained model"
echo ""
echo "  Logs: logs/${ROUND_NAME}_${OUTPUT_SUFFIX}_<timestamp>/"
echo "    - training.log: Full training logs"
echo "    - training_metrics.csv: Training metrics + evaluation results"
echo "    - metrics_summary.txt: Summary with evaluation accuracy"
echo ""
echo "Note: <timestamp> format is YYYYMMDD_HHMMSS"
echo "=========================================="

