#!/bin/bash

################################################################################
# Agent LoRA Training Script
# Trains Qwen2.5-Math-1.5B with LoRA for interactive code execution agent
# Target: solver_with_interactive_code.py
#
# Data:
#   - rstar_100k_clean.csv: Pure correct reasoning with code execution
#   - correction_gsm8k: Self-correction samples (~7.5k)
#   - correction_math: Self-correction samples (~12k)
#
# Format: Plain text (NO chat template) for consistency with previous experiments
################################################################################

set -e

################################################################################
# Training Settings

# Round name (custom identifier for this training run)
ROUND_NAME="agent_lora_v1"

# Model to train
MODEL="pretrained_models/Qwen2.5-Math-1.5B"

# Training mode
MODE="lora"

# LoRA rank (higher = more capacity but slower)
LORA_RANK=64

# Number of training epochs (LoRA converges faster, 3-5 is usually enough)
NUM_EPOCHS=1

# Batch size (reduced to avoid OOM on 24GB GPU)
BATCH_SIZE=18

# Gradient accumulation steps
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 4 * 16 = 64
GRADIENT_ACCUMULATION_STEPS=8

# Maximum sequence length (interactive code responses can be long)
MAX_SEQ_LENGTH=2048

# Save checkpoint every N epochs (999 = only save final model)
SAVE_EVERY_N_EPOCHS=1

# GPU to use (set the GPU ID here)
export CUDA_VISIBLE_DEVICES=0
GPUS="0"  # Must match CUDA_VISIBLE_DEVICES (use "0" for single GPU)

################################################################################
# Data Settings

# Data mix options:
#   - all: rstar + correction data (recommended)
#   - rstar: rstar only (pure correct samples)
#   - correction_only: correction data only (for debugging)
DATA_MIX="all"

# Maximum samples from rstar (set to limit training time)
# Options: leave empty for all (~100k), or set a number like 30000
MAX_RSTAR_SAMPLES=30000

# Maximum correction samples (leave empty for all)
# General limit (applies to both if specific limits not set)
MAX_CORRECTION_SAMPLES=""

# Specific limits for each correction dataset
# Recommendation: keep low to avoid "fake error" pattern (15-20% of total data)
# GSM8K correction: ~7.5k total available
MAX_GSM8K_CORRECTION=2000

# MATH correction: ~12k total available  
MAX_MATH_CORRECTION=3000

################################################################################
# Logging Settings

# Enable Weights & Biases logging
USE_WANDB="--use_wandb"  # Set to "" to disable

# W&B project name
WANDB_PROJECT="slm_math_agent"

################################################################################
# Run Training

echo "=============================================================================="
echo "          Agent LoRA Training - Interactive Code Execution                    "
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Round Name: $ROUND_NAME"
echo "  Model: $MODEL"
echo "  Mode: $MODE (rank=$LORA_RANK)"
echo "  Data Mix: $DATA_MIX"
echo "  Max rstar samples: ${MAX_RSTAR_SAMPLES:-all}"
echo "  Max GSM8K correction: ${MAX_GSM8K_CORRECTION:-all}"
echo "  Max MATH correction: ${MAX_MATH_CORRECTION:-all}"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "  Max sequence length: $MAX_SEQ_LENGTH"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Chat Template: NO (plain text format)"
echo ""
echo "Output directories:"
echo "  Checkpoints: checkpoints/${ROUND_NAME}_lora_r${LORA_RANK}_<timestamp>/"
echo "  Logs: logs/${ROUND_NAME}_lora_r${LORA_RANK}_<timestamp>/"
echo "=============================================================================="
echo ""

# Build command
CMD="python -m models.train_agent_sft \
    --mode $MODE \
    --model_name $MODEL \
    --round_name $ROUND_NAME \
    --data_mix $DATA_MIX \
    --lora_rank $LORA_RANK \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --save_every_n_epochs $SAVE_EVERY_N_EPOCHS \
    --gpus $GPUS \
    --gradient_checkpointing"

# Add optional parameters
if [ -n "$MAX_RSTAR_SAMPLES" ]; then
    CMD="$CMD --max_rstar_samples $MAX_RSTAR_SAMPLES"
fi

if [ -n "$MAX_CORRECTION_SAMPLES" ]; then
    CMD="$CMD --max_correction_samples $MAX_CORRECTION_SAMPLES"
fi

if [ -n "$MAX_GSM8K_CORRECTION" ]; then
    CMD="$CMD --max_gsm8k_correction $MAX_GSM8K_CORRECTION"
fi

if [ -n "$MAX_MATH_CORRECTION" ]; then
    CMD="$CMD --max_math_correction $MAX_MATH_CORRECTION"
fi

if [ -n "$USE_WANDB" ]; then
    CMD="$CMD $USE_WANDB --wandb_project $WANDB_PROJECT"
fi

# NOTE: Do NOT add --apply_chat_template (default is false = plain text format)

# Run
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=============================================================================="
echo "                         Training Complete!                                   "
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Check logs in: logs/${ROUND_NAME}_lora_r${LORA_RANK}_*/"
echo "  2. Evaluate with: scripts/eval_agent_sft_model.sh"
echo "     (update CHECKPOINT path to point to the trained LoRA adapter)"
echo ""

