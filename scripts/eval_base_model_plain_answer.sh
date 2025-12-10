#!/bin/bash

################################################################################
# Evaluation Script (plain prompt + explicit Answer: section)
# Calls evaluation/eval.py with --append_answer_section true
################################################################################

set -e  # Exit on error

################################################################################
# Configuration Variables

# Model to evaluate
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path (optional)
# e.g., checkpoints/plain_sft_sft_*/final_model
CHECKPOINT="checkpoints/plain_sft_sft_20251209_011851/final_model"

# Test round name
ROUND_NAME="my_evaluation_sft_plain_answer"

# Dataset: "gsm8k" or "math500"
DATASET="math500"

# Number of test cases (0 = full dataset)
COUNT=500

# Evaluation mode: "standard"
MODE="standard"

# Streaming console output
DETAILED="true"

# Log full sample details to file
LOG_SAMPLES="true"

# Batch size
BATCH_SIZE=1

# Greedy decoding
GREEDY="true"

# Apply chat template (keep false for plain SFT)
APPLY_CHAT_TEMPLATE="false"

# Append "Answer:" section to prompt (matches train_sft_plain)
APPEND_ANSWER_SECTION="true"

################################################################################
# GPU / workers

# Single-process
WORKERS=0
GPUS="1"

################################################################################
# Advanced

RESUME_DIR=""
SAVE_INTERVAL=10

################################################################################
# Run Evaluation

echo "=========================================="
echo "Evaluation (plain prompt + Answer:)"
echo "=========================================="
echo "Model: $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Round: $ROUND_NAME"
echo "Dataset: $DATASET"
echo "Count: $COUNT"
echo "Mode: $MODE"
echo "Batch Size: $BATCH_SIZE"
echo "Greedy: $GREEDY"
echo "Append Answer Section: $APPEND_ANSWER_SECTION"
echo "Apply Chat Template: $APPLY_CHAT_TEMPLATE"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=$(echo "$GPUS" | cut -d',' -f1) python -m evaluation.eval \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --detailed "$DETAILED" \
    --log_samples "$LOG_SAMPLES" \
    --greedy "$GREEDY" \
    --apply_chat_template "$APPLY_CHAT_TEMPLATE" \
    --append_answer_section "$APPEND_ANSWER_SECTION" \
    --save_interval "$SAVE_INTERVAL"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results: results/${ROUND_NAME}_*/"
echo "=========================================="

