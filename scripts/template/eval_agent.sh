#!/bin/bash

################################################################################
# Agent Evaluation Script Template
# 统一测试脚本，支持不同的agent方法
################################################################################

set -e

################################################################################
# Configuration Variables

# Model
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path (optional)
# Set to empty string "" to use base pretrained model
# For LoRA: "checkpoints/lora_r16_20251124_130958/checkpoint-1188"
# For SFT: "checkpoints/sft_20251124_131423/checkpoint-891"
CHECKPOINT=""

# Agent method: solver_checker, solver_checker_chat, or majority_vote
AGENT="solver_checker"

# For solver_checker: Checker model (leave empty to use same as solver)
CHECKER_MODEL=""

# For solver_checker: Checker checkpoint (optional, leave empty to use same as solver checkpoint)
CHECKER_CHECKPOINT=""

# For solver_checker and solver_checker_chat: Max iterations per problem
MAX_ITERATIONS=5

# For majority_vote: Number of runs
NUM_RUNS=5

# For majority_vote: Temperature
TEMPERATURE=0.7

# For majority_vote: Top-p
TOP_P=0.95

# Test round name
ROUND_NAME="test_agent"

# Dataset
DATASET="gsm8k"

# Number of test cases (0 = full dataset)
COUNT=10

# Detailed output
DETAILED="true"

# Resume from existing results (leave empty to start fresh)
RESUME_DIR=""

# Save interval
SAVE_INTERVAL=5

################################################################################
# Pre-flight Checks

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          Agent Evaluation - Unified Testing Script                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check model
MODEL_PATH="pretrained_models/${MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Model: $MODEL_PATH"

if [ "$AGENT" = "solver_checker" ] && [ -n "$CHECKER_MODEL" ]; then
    CHECKER_PATH="pretrained_models/${CHECKER_MODEL}"
    if [ ! -d "$CHECKER_PATH" ]; then
        echo "✗ Error: Checker model not found at $CHECKER_PATH"
        exit 1
    fi
    echo "✓ Checker Model: $CHECKER_PATH"
elif [ "$AGENT" = "solver_checker_chat" ]; then
    echo "✓ Agent Mode: Chat-based (using shared model)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Evaluation Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
fi
echo "  Agent Method: $AGENT"
if [ "$AGENT" = "solver_checker" ]; then
if [ -n "$CHECKER_MODEL" ]; then
    echo "  Checker Model: $CHECKER_MODEL"
    if [ -n "$CHECKER_CHECKPOINT" ]; then
        echo "  Checker Checkpoint: $CHECKER_CHECKPOINT"
    fi
    else
        echo "  Checker Model: Same as solver"
    fi
    echo "  Max Iterations: $MAX_ITERATIONS"
elif [ "$AGENT" = "solver_checker_chat" ]; then
    echo "  Mode: Chat-based (shared model)"
    echo "  Max Iterations: $MAX_ITERATIONS"
elif [ "$AGENT" = "majority_vote" ]; then
    echo "  Num Runs: $NUM_RUNS"
    echo "  Temperature: $TEMPERATURE"
    echo "  Top-p: $TOP_P"
fi
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT (0 = full dataset)"
echo "  Detailed Output: $DETAILED"
if [ -n "$RESUME_DIR" ]; then
    echo "  Resume: $RESUME_DIR"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# Run Evaluation

echo "Starting agent evaluation..."
echo ""

# Build command
CMD="python -m evaluation.eval_agent \
    --model \"$MODEL\" \
    --agent \"$AGENT\" \
    --round \"$ROUND_NAME\" \
    --dataset \"$DATASET\" \
    --count \"$COUNT\" \
    --detailed \"$DETAILED\" \
    --save_interval \"$SAVE_INTERVAL\""

# Add checkpoint parameter if specified
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint \"$CHECKPOINT\""
fi

# Add agent-specific parameters
if [ "$AGENT" = "solver_checker" ] || [ "$AGENT" = "solver_checker_chat" ]; then
    CMD="$CMD --max_iterations \"$MAX_ITERATIONS\""
    if [ "$AGENT" = "solver_checker" ] && [ -n "$CHECKER_MODEL" ]; then
        CMD="$CMD --checker_model \"$CHECKER_MODEL\""
        if [ -n "$CHECKER_CHECKPOINT" ]; then
            CMD="$CMD --checker_checkpoint \"$CHECKER_CHECKPOINT\""
        fi
    fi
elif [ "$AGENT" = "majority_vote" ]; then
    CMD="$CMD --num_runs \"$NUM_RUNS\""
    CMD="$CMD --temperature \"$TEMPERATURE\""
    CMD="$CMD --top_p \"$TOP_P\""
fi

# Add resume if specified
if [ -n "$RESUME_DIR" ]; then
    CMD="$CMD --resume \"$RESUME_DIR\""
fi

# Execute command
eval $CMD

################################################################################
# Complete

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Evaluation Complete!                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: results/${ROUND_NAME}_${MODEL}_${DATASET}_*"
echo "Analysis report: results/${ROUND_NAME}_${MODEL}_${DATASET}_*/analysis_report.txt"
echo ""

