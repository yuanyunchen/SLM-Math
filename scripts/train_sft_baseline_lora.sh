#!/bin/bash

################################################################################
# LoRA Training Script (Rank 16)
# Trains Qwen2.5-Math-1.5B with LoRA rank 16
# Configuration based on exp_gpu2_lora_then_rl.sh Phase 1 settings

################################################################################
# Training Settings

# Round name (custom identifier for this training run)
ROUND_NAME="reasoning_code_lora"  # Change this to identify your training round

# Model to train
MODEL="pretrained_models/Qwen2.5-Math-1.5B"

# Training data path
DATA_PATH="data/reasoning_code/run_1208_1337/reasoning_code_gsm8k_train_math_train_1208_1337.json"

# Training mode (lora = LoRA fine-tuning)
MODE="lora"

# LoRA rank
LORA_RANK=32

# Number of training epochs
NUM_EPOCHS=3

# Batch size (single GPU, LoRA can use larger batch)
BATCH_SIZE=20

# Gradient accumulation steps
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
# = 24 * 4 = 96
GRADIENT_ACCUMULATION_STEPS=8

# Save checkpoint every N epochs (999 = only save final model)
SAVE_EVERY_N_EPOCHS=1

# Evaluation settings
EVAL_EVERY_N_EPOCHS=1
EVAL_SAMPLES=50  # Samples per dataset for evaluation

# GPU to use (default to cuda:1)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
GPUS="2"

# Enable Weights & Biases logging
USE_WANDB="--use_wandb"  # Comment out to disable W&B logging

# W&B project and run name
WANDB_PROJECT="test_lora_batch"
WANDB_RUN_NAME=""  # Leave empty for auto-generated name, or set custom name

# Enable evaluation during training
ENABLE_EVAL="--enable_eval"

################################################################################
# Run training

echo "=========================================="
echo "Starting LoRA Training (Rank $LORA_RANK)"
echo "=========================================="
echo "Round Name: $ROUND_NAME"
echo "Model: $MODEL"
echo "Data: $DATA_PATH"
echo "Mode: $MODE"
echo "LoRA Rank: $LORA_RANK"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Eval every: $EVAL_EVERY_N_EPOCHS epoch(s)"
echo "Eval samples: $EVAL_SAMPLES per dataset"
echo "Save: final model only"
echo ""
echo "Output directories:"
echo "  Checkpoints: checkpoints/${ROUND_NAME}_lora_r${LORA_RANK}_<timestamp>/"
echo "  Logs: logs/${ROUND_NAME}_lora_r${LORA_RANK}_<timestamp>/"
echo "=========================================="

# Build wandb arguments
WANDB_ARGS=""
if [ -n "$USE_WANDB" ]; then
    WANDB_ARGS="$USE_WANDB --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

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
    $WANDB_ARGS \
    $ENABLE_EVAL \
    --gradient_checkpointing \
    --eval_samples "$EVAL_SAMPLES"

echo "=========================================="
echo "Training completed!"
echo ""
echo "Output files:"
echo "  Checkpoints: checkpoints/${ROUND_NAME}_lora_r${LORA_RANK}_<timestamp>/"
echo "    - final_model/: Final LoRA adapter"
echo ""
echo "  Logs: logs/${ROUND_NAME}_lora_r${LORA_RANK}_<timestamp>/"
echo "    - training.log: Full training logs"
echo "    - training_metrics.csv: Training metrics + evaluation results"
echo "    - metrics_summary.txt: Summary with evaluation accuracy"
echo ""
echo "Note: <timestamp> format is YYYYMMDD_HHMMSS"
echo "=========================================="
