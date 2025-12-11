#!/bin/bash

################################################################################
# Solver-Verifier Multi (math) - data generation for verifier finetune
################################################################################

set -euo pipefail

# MODEL: model name or path
MODEL="Qwen2.5-Math-1.5B"
# Optional checkpoint (empty = none)
CHECKPOINT=""
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# Round name for results directory
ROUND_NAME="vf_multi_math_full"
# Dataset and split
DATASET="math"
SPLIT="train"       # 使用 math 训练集；如需测试集改回 test
# 0 = full split
COUNT=0
# Iterations for solver-verifier loop
MAX_ITERATIONS=1
# Tools (verifier side does not run code)
ENABLE_SOLVER_TOOLS="true"
# Checker tool flag kept for interface parity, unused in multi
ENABLE_CHECKER_TOOLS="false"
# Inference settings
BATCH_SIZE=1
INFERENCE_BACKEND="transformers"
APPLY_CHAT_TEMPLATE="false"
DETAILED="true"
SAVE_INTERVAL=10
START=0
# Resume path under results/ (leave empty to start fresh)
RESUME_DIR=""

echo "=============================================================================="
echo "        Solver-Verifier Multi Evaluation (math)                               "
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

