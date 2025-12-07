#!/bin/bash

################################################################################
# GRPO RL Training for Agent with Code Feedback
#
# This trains RL on the 2-step agent workflow:
#   Step 1: Generate reasoning + code
#   Step 2: Execute code, provide feedback, generate final answer
#
# Uses the exact same prompt/workflow as agent_with_code_feedback.py
################################################################################

# Use cached models offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

################################################################################
# Configuration

# Model - Qwen2.5-Math-1.5B
MODEL="pretrained_models/Qwen2.5-Math-1.5B"

# Training Data
TRAIN_FILE="data/rstart_sft_interactive/rstar_100k_clean.csv"

# Output directory
OUTPUT_DIR="results/rl_agent_code_feedback_checkpoints"

# Training parameters - Optimized for H20 GPU
NUM_EPOCHS=1
BATCH_SIZE=4            # 4 samples per batch (each needs 2 generations)
GRADIENT_ACCUMULATION=2 # Effective batch = 8
LEARNING_RATE=1e-5

# GRPO parameters
NUM_RETURN_SEQUENCES=2  # 2 generations per prompt for variance
KL_COEF=0               # Skip reference model for speed

# Generation parameters (MUST match generation_config.py & agent_with_code_feedback.py)
TEMPERATURE=0.7         # Same as TEMPERATURE in generation_config.py
TOP_P=0.95              # Same as TOP_P in generation_config.py
REPETITION_PENALTY=1.15 # Same as REPETITION_PENALTY in generation_config.py
MAX_NEW_TOKENS=2048     # Same as MAX_NEW_TOKENS in generation_config.py

# Prompt formatting (match agent_with_code_feedback.py)
APPLY_CHAT_TEMPLATE="false"

# Logging
LOGGING_STEPS=5
SAVE_STEPS=50
MAX_STEPS=-1

# Data - 150 samples as requested
MAX_SAMPLES=150

# Random seed
SEED=42

# Weights & Biases
USE_WANDB="--use_wandb"
WANDB_ENTITY="nlp_final_math"
WANDB_PROJECT="rl_agent_code_feedback"

################################################################################
# Run Training

echo "=============================================================================="
echo "     GRPO RL Training for Agent with Code Feedback                           "
echo "=============================================================================="
echo ""
echo "Model: $MODEL"
echo "Samples: $MAX_SAMPLES"
echo "Batch: $BATCH_SIZE x $GRADIENT_ACCUMULATION = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Agent: 2-step code feedback workflow"
echo "=============================================================================="

python -m models.train_rl_agent_code_feedback \
    --model_path "$MODEL" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --kl_coef $KL_COEF \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --repetition_penalty $REPETITION_PENALTY \
    --max_new_tokens $MAX_NEW_TOKENS \
    --apply_chat_template $APPLY_CHAT_TEMPLATE \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --max_steps $MAX_STEPS \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    $USE_WANDB \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT"

echo ""
echo "=============================================================================="
echo "                         Training Complete!                                   "
echo "=============================================================================="

