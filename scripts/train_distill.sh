#!/bin/bash

################################################################################
# On-Policy Distillation Training Script
# Student learns from teacher's online generations
#
# Features:
# - Online generation from teacher model
# - Knowledge distillation (KL divergence + CE loss)
# - Quality filtering
# - Support for iterative and self-distillation
################################################################################

set -e  # Exit on error

################################################################################
# Training Settings

# Teacher model (larger/better model)
TEACHER_MODEL="Qwen2.5-Math-7B"
TEACHER_PATH="pretrained_models/Qwen2.5-Math-7B"

# Student model (smaller model to train)
STUDENT_MODEL="Qwen2.5-Math-1.5B"
STUDENT_PATH="pretrained_models/Qwen2.5-Math-1.5B"

# Run name
RUN_NAME="distill_7b_to_1.5b"

# Training data
DATASET="gsm8k"
DATA_FILE="data/cot_generated/cot_x_ai_grok_4_1_fast_reasoning_gsm8k_train_7473_1120_2035/cot_data.json"

# Training duration
NUM_EPOCHS=3
BATCH_SIZE=4

# Distillation parameters
TEMPERATURE=2.0
KL_WEIGHT=0.5
CE_WEIGHT=0.5

# Configuration file
CONFIG_FILE="configs/distill_config.yaml"

################################################################################
# Advanced Settings

# Use online generation from teacher
ONLINE_GENERATION=true

# Filter by quality
FILTER_BY_CORRECTNESS=true

# Iterative distillation (student becomes teacher)
ITERATIVE_DISTILL=false

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                  On-Policy Distillation Training                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if teacher model exists
if [ ! -d "$TEACHER_PATH" ]; then
    echo "✗ Error: Teacher model not found at $TEACHER_PATH"
    echo "  Please download or train the teacher model first."
    exit 1
fi
echo "✓ Teacher model found: $TEACHER_PATH"

# Check if student model exists
if [ ! -d "$STUDENT_PATH" ]; then
    echo "✗ Error: Student model not found at $STUDENT_PATH"
    exit 1
fi
echo "✓ Student model found: $STUDENT_PATH"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "✗ Error: Data file not found: $DATA_FILE"
    exit 1
fi
echo "✓ Data file found: $DATA_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file: $CONFIG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Distillation Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Teacher: $TEACHER_MODEL"
echo "  Student: $STUDENT_MODEL"
echo "  Run name: $RUN_NAME"
echo "  Dataset: $DATASET"
echo "  Epochs: $NUM_EPOCHS"
echo "  Temperature: $TEMPERATURE"
echo "  KL weight: $KL_WEIGHT"
echo "  CE weight: $CE_WEIGHT"
echo "  Online generation: $ONLINE_GENERATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Confirm before starting
read -p "Start distillation training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

################################################################################
# Run Training

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    Starting Distillation Training...                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Build command
CMD="python -m models.train_distill --config $CONFIG_FILE"

# Add optional overrides
if [ ! -z "$TEACHER_MODEL" ]; then
    CMD="$CMD --teacher_model $TEACHER_MODEL"
fi

if [ ! -z "$STUDENT_MODEL" ]; then
    CMD="$CMD --student_model $STUDENT_MODEL"
fi

# Run training
echo "Command: $CMD"
echo ""
$CMD

################################################################################
# Training Complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    Distillation Training Complete!                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: results/distill_checkpoints/${RUN_NAME}_*/"
echo ""
echo "Next steps:"
echo "  1. View training logs:"
echo "     tensorboard --logdir=logs/distill"
echo ""
echo "  2. Compare student vs teacher:"
echo "     python -m evaluation.eval_pipeline --model results/distill_checkpoints/${RUN_NAME}_*/final_model"
echo "     python -m evaluation.eval_pipeline --model $TEACHER_PATH"
echo ""
echo "  3. Check distillation quality:"
echo "     # Student should achieve 80-95% of teacher's performance"
echo "     # With significantly reduced model size and faster inference"
echo ""


