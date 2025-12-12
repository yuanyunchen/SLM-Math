#!/bin/bash

# Train Agent with Code Feedback using GRPO RL
# Configuration matches report: Code Feedback (SFT+RL) → GSM8K 84.6%, MATH 67.8%
# Usage: bash scripts/train_rl_code_feedback.sh

# Model checkpoint (should be SFT checkpoint)
SFT_CHECKPOINT="checkpoints/sft_lora_final"

# Configuration
CONFIG_FILE="configs/rl_grpo_config.yaml"
OUTPUT_DIR="results/rl_code_feedback_$(date +%Y%m%d_%H%M%S)"

# Training parameters (matches report)
NUM_EPOCHS=1
LEARNING_RATE=5e-6
KL_COEF=0.05
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8

# GPU
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "RL Training: Agent with Code Feedback"
echo "=========================================="
echo "SFT Checkpoint: $SFT_CHECKPOINT"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "KL coefficient: $KL_COEF"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

python models/train_rl_code_feedback.py \
    --sft_checkpoint "$SFT_CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    --epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --kl_coef "$KL_COEF" \
    --gpus "${CUDA_VISIBLE_DEVICES}"

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo "To evaluate: bash scripts/eval_agent_with_code_feedback.sh"
echo "=========================================="

