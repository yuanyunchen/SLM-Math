#!/bin/bash

################################################################################
# Solver-Verifier GRPO RL Training Script
# - Rewards incorporate code execution success and consistency
# - Supports GSM8K + MATH datasets via HuggingFace load_from_disk
################################################################################

set -e

################################################################################
# Training Settings

# Model path (SFT or LoRA-ready checkpoint)
MODEL_PATH="pretrained_models/Qwen2.5-Math-1.5B"

# Config file (reuses rl_grpo_config.yaml defaults)
CONFIG_FILE="configs/rl_grpo_config.yaml"

# Output directory
OUTPUT_DIR="results/rl_solver_verifier_$(date +%Y%m%d_%H%M%S)"

# Datasets to train on (comma-separated)
DATASETS="gsm8k,math500"

# GPU
export CUDA_VISIBLE_DEVICES=0

# W&B (optional)
WANDB_PROJECT="slm_math_solver_verifier_rl"
WANDB_RUN_NAME=""
USE_WANDB="--use_wandb"

################################################################################
# Hyperparameters

NUM_EPOCHS=1
MAX_SAMPLES=4000       # Total samples across datasets (-1 for all)
BATCH_SIZE=10
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-6
NUM_RETURN_SEQUENCES=1
TEMPERATURE=0.7
KL_COEF=0
LOGGING_STEPS=10
EVAL_STEPS=200        # evaluate less frequently
SAVE_STEPS=200

# Reward shaping
REWARD_CODE_ERROR=-0.2
REWARD_CODE_INCONSISTENT=-0.2
REWARD_CODE_CONSISTENT=0.1
REWARD_DUPLICATE_ANSWER=-0.1
REWARD_MISSING_BOX=-0.1

################################################################################
# Pre-flight

echo "=========================================="
echo "Solver-Verifier GRPO RL Training"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Datasets: $DATASETS"
echo "Epochs: $NUM_EPOCHS"
echo "Max samples: $MAX_SAMPLES"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS"
echo "Learning rate: $LEARNING_RATE"
echo "Num return sequences: $NUM_RETURN_SEQUENCES"
echo "Temperature: $TEMPERATURE"
echo "KL Coef: $KL_COEF"
echo "Logging/Eval/Save steps: $LOGGING_STEPS / $EVAL_STEPS / $SAVE_STEPS"
echo "Reward(code_error): $REWARD_CODE_ERROR"
echo "Reward(code_inconsistent): $REWARD_CODE_INCONSISTENT"
echo "Reward(code_consistent): $REWARD_CODE_CONSISTENT"
echo "Reward(duplicate_answer): $REWARD_DUPLICATE_ANSWER"
echo "Reward(missing_box): $REWARD_MISSING_BOX"
echo "=========================================="

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

################################################################################
# Run

# Mitigate CUDA fragmentation on tight VRAM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB_ARGS=""
if [ -n "$USE_WANDB" ]; then
    WANDB_ARGS="$USE_WANDB --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

python models/train_rl_solver_verifier.py \
    --config "$CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --datasets "$DATASETS" \
    --max_samples "$MAX_SAMPLES" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --temperature "$TEMPERATURE" \
    --kl_coef "$KL_COEF" \
    --logging_steps "$LOGGING_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_samples 10 \
    --reward_code_error "$REWARD_CODE_ERROR" \
    --reward_code_inconsistent "$REWARD_CODE_INCONSISTENT" \
    --reward_code_consistent "$REWARD_CODE_CONSISTENT" \
    --reward_duplicate_answer "$REWARD_DUPLICATE_ANSWER" \
    --reward_missing_box "$REWARD_MISSING_BOX" \
    $WANDB_ARGS

################################################################################
# Post

echo ""
echo "=========================================="
echo "Training Completed!"
echo "Output directory: $OUTPUT_DIR"
echo "Logs: $OUTPUT_DIR/training.log"
echo "=========================================="

