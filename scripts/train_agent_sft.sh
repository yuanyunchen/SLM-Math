#!/bin/bash

################################################################################
# Agent SFT Training Script
# Trains model on interactive code execution data with self-correction
#
# Data Sources:
#   - rstar_100k_clean.csv: Pure correct reasoning (~100k samples)
#   - correction_gsm8k_*.json: Self-correction data (~7.5k samples)
#   - correction_math_*.json: Self-correction data (~15k samples)
#
# Loss Strategy:
#   - Train on FULL sequences including error + reflection + correction
#   - This teaches the model to identify errors, reflect, and self-correct

################################################################################
# Training Settings

# Round name (custom identifier for this training run)
# Results will be saved to: checkpoints/<ROUND_NAME>_sft_<timestamp>/
ROUND_NAME="agent_sft_v1"

# Model to train
MODEL="pretrained_models/Qwen2.5-Math-1.5B"

# Training mode (sft = full fine-tuning, lora = LoRA)
MODE="sft"

# Data mix options:
#   - all: rstar + correction (both GSM8K and MATH)
#   - rstar: rstar only (pure correct reasoning)
#   - correction: correction only (self-correction data)
#   - correction_only: same as correction
DATA_MIX="all"

# Maximum samples from each data source (set to empty for all)
# Useful for quick experiments
MAX_RSTAR_SAMPLES=""       # e.g., 50000 for 50k samples, empty for all ~100k
MAX_CORRECTION_SAMPLES=""  # e.g., 5000 per dataset, empty for all

# Number of training epochs
NUM_EPOCHS=3

# Batch size configuration
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4

# Learning rate
# Recommended: 1e-5 for full SFT, 1e-4 for LoRA
LEARNING_RATE=1e-5

# Maximum sequence length
# Agent responses can be longer due to code blocks and corrections
# Note: 4096 may cause OOM on 24GB GPU, use 2048 for safety
MAX_SEQ_LENGTH=2048

# Save checkpoint every N epochs
SAVE_EVERY_N_EPOCHS=1

# GPU to use
export CUDA_VISIBLE_DEVICES=0
GPUS="0"

# Memory optimization
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# Enable Weights & Biases logging (set to "--use_wandb" to enable)
USE_WANDB="--use_wandb"
WANDB_PROJECT="slm_math_agent_sft"
WANDB_RUN_NAME=""  # Leave empty for auto-generated name

# LoRA settings (only used when MODE=lora)
LORA_RANK=16


################################################################################
# Build command arguments

ARGS=""

# Data mix settings
if [ -n "$MAX_RSTAR_SAMPLES" ]; then
    ARGS="$ARGS --max_rstar_samples $MAX_RSTAR_SAMPLES"
fi
if [ -n "$MAX_CORRECTION_SAMPLES" ]; then
    ARGS="$ARGS --max_correction_samples $MAX_CORRECTION_SAMPLES"
fi

# Wandb settings
if [ -n "$USE_WANDB" ]; then
    ARGS="$ARGS $USE_WANDB --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        ARGS="$ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

# LoRA settings
if [ "$MODE" == "lora" ]; then
    ARGS="$ARGS --lora_rank $LORA_RANK"
fi


################################################################################
# Run training

echo "=========================================="
echo "Agent SFT Training"
echo "=========================================="
echo "Round Name: $ROUND_NAME"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "Data Mix: $DATA_MIX"
echo "Epochs: $NUM_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Output directories:"
echo "  Checkpoints: checkpoints/${ROUND_NAME}_${MODE}_<timestamp>/"
echo "  Logs: logs/${ROUND_NAME}_${MODE}_<timestamp>/"
echo "=========================================="

python models/train_agent_sft.py \
    --mode "$MODE" \
    --model_name "$MODEL" \
    --data_mix "$DATA_MIX" \
    --round_name "$ROUND_NAME" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --gpus "$GPUS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    $GRADIENT_CHECKPOINTING \
    $ARGS

echo "=========================================="
echo "Training completed!"
echo ""
echo "Output files:"
echo "  Checkpoints: checkpoints/${ROUND_NAME}_${MODE}_<timestamp>/"
echo "    - final_model/: Final trained model"
echo ""
echo "  Logs: logs/${ROUND_NAME}_${MODE}_<timestamp>/"
echo "    - training.log: Full training logs"
echo "    - training_metrics.csv: Training metrics"
echo "    - metrics_summary.txt: Summary"
echo "    - training_config.json: Training configuration"
echo "=========================================="

