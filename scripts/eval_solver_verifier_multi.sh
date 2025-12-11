#!/bin/bash

################################################################################
# Solver-Verifier Multi Evaluation (general)
################################################################################

set -euo pipefail

# Model and optional checkpoint
MODEL="Qwen2.5-Math-1.5B"
CHECKPOINT=""
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Experiment tags
ROUND_NAME="test_solver_verifier_multi"
DATASET="gsm8k"
SPLIT="test"
COUNT=500          # 0 = full split
START=0

# Iterations and tools
MAX_ITERATIONS=3
ENABLE_SOLVER_TOOLS="true"
# Verifier in multi does not execute code; keep flag for interface parity
ENABLE_CHECKER_TOOLS="false"

# Inference / logging
BATCH_SIZE=1
INFERENCE_BACKEND="transformers"
APPLY_CHAT_TEMPLATE="false"
DETAILED="true"
SAVE_INTERVAL=10
RESUME_DIR=""

echo "=============================================================================="
echo "                Solver-Verifier Multi - Evaluation                            "
echo "=============================================================================="
echo "Model: $MODEL"
[ -n "$CHECKPOINT" ] && echo "Checkpoint: $CHECKPOINT"
echo "Round: $ROUND_NAME"
echo "Dataset: $DATASET  Split: $SPLIT  Count: $COUNT"
echo "Max Iterations: $MAX_ITERATIONS  Detailed: $DETAILED"
echo "Apply Chat Template: $APPLY_CHAT_TEMPLATE"
[ -n "$RESUME_DIR" ] && echo "Resume: $RESUME_DIR"
echo "=============================================================================="

CMD=(
  python -m evaluation.eval_agent
    --model "$MODEL"
    --agent solver_verifier_multi
    --round "$ROUND_NAME"
    --dataset "$DATASET"
    --split "$SPLIT"
    --count "$COUNT"
    --start "$START"
    --max_iterations "$MAX_ITERATIONS"
    --enable_solver_tools "$ENABLE_SOLVER_TOOLS"
    --enable_checker_tools "$ENABLE_CHECKER_TOOLS"
    --batch_size "$BATCH_SIZE"
    --inference_backend "$INFERENCE_BACKEND"
    --apply_chat_template "$APPLY_CHAT_TEMPLATE"
    --detailed "$DETAILED"
    --save_interval "$SAVE_INTERVAL"
)

[ -n "$CHECKPOINT" ] && CMD+=("--checkpoint" "$CHECKPOINT")
[ -n "$RESUME_DIR" ] && CMD+=("--resume" "$RESUME_DIR")

echo ""
echo "Running command:"
printf '  %q ' "${CMD[@]}"; echo
echo ""

"${CMD[@]}"

echo ""
echo "Done. Check results under: results/${ROUND_NAME}_${MODEL}_${DATASET}_*"
echo "Key files: logging.log, metrics.csv, summary.txt, answer.json, log/"

