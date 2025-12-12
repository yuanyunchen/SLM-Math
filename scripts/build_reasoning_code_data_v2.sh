#!/bin/bash

################################################################################
# Reasoning + Code Data Generation (V2 Prompt, Multi-Model, Multi-Dataset)
#
# Uses the improved prompt in dataset/build_reasoning_code_data_v2.py.
# Two rounds:
#   - Round 1: teacher models with retries
#   - Round 2: expert models for remaining failures (optional)
# Code output is treated as the source of truth for the final answer.
################################################################################

set -euo pipefail

# ============================================================================
# Model Configuration
# ============================================================================
# Teacher models (Round 1)
TEACHER_MODELS="x-ai/grok-4-1-fast-reasoning"

# Expert models (Round 2) - optional (set empty for single round)
EXPERT_MODELS=""

# ============================================================================
# Dataset Configuration
# ============================================================================
# Supported: gsm8k-train, gsm8k-test, math-train, math500, competition_math-train
DATASETS="gsm8k-train,math-train"
COUNT=0          # 0 = full split
START_INDEX=0    # starting sample index

# ============================================================================
# Round Configuration
# ============================================================================
# Round 1 (Teachers)
ROUND1_ATTEMPTS=3
ROUND1_EFFORTS="medium"
ROUND1_EFFORT_RATIOS=""          # e.g., "0.7,0.3" (empty = even split)
ROUND1_TEMPS="0.7,0.5,0.9"

# Round 2 (Experts)
ROUND2_ATTEMPTS=5
ROUND2_EFFORTS="high"
ROUND2_EFFORT_RATIOS=""          # e.g., "0.6,0.4" (empty = even split)
ROUND2_TEMPS="0.7,0.5,0.9,0.3,1.0"

# ============================================================================
# General Configuration
# ============================================================================
MAX_TOKENS=2048   # Keep shorter for small models per request
WORKERS=512
CODE_TIMEOUT=8    # seconds to run generated code

# API configuration (use env var AIML_API_KEY if present)
API_KEY="6f9c61d1f5be40b6afd1dff6a878d378"
API_BASE="https://api.aimlapi.com/v1"

# GPU selection (default to cuda:1)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Output directory
OUTPUT_DIR="data/reasoning_code_v2/run_$(date +%m%d_%H%M)"

echo "================================================================================"
echo "          Reasoning + Code Data Generation - V2 Prompt"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Teacher Models: $TEACHER_MODELS"
echo "  Expert Models:  ${EXPERT_MODELS:-<disabled>}"
echo "  Datasets:       $DATASETS"
echo "  Round 1:        $ROUND1_ATTEMPTS attempts, efforts=$ROUND1_EFFORTS"
echo "  Round 2:        $ROUND2_ATTEMPTS attempts, efforts=$ROUND2_EFFORTS"
echo "  Max Tokens:     $MAX_TOKENS"
echo "  Workers:        $WORKERS"
echo "  Code Timeout:   ${CODE_TIMEOUT}s"
echo "  Output:         $OUTPUT_DIR"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""
echo "================================================================================"
echo ""

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: API_KEY is empty. Set AIML_API_KEY or edit this script."
  exit 1
fi

python -m dataset.build_reasoning_code_data_v2 \
    --teacher "$TEACHER_MODELS" \
    --expert "$EXPERT_MODELS" \
    --dataset "$DATASETS" \
    --count $COUNT \
    --start $START_INDEX \
    --round1-attempts $ROUND1_ATTEMPTS \
    --round1-efforts "$ROUND1_EFFORTS" \
    --round1-effort-ratios "$ROUND1_EFFORT_RATIOS" \
    --round1-temps "$ROUND1_TEMPS" \
    --round2-attempts $ROUND2_ATTEMPTS \
    --round2-efforts "$ROUND2_EFFORTS" \
    --round2-effort-ratios "$ROUND2_EFFORT_RATIOS" \
    --round2-temps "$ROUND2_TEMPS" \
    --max-tokens $MAX_TOKENS \
    --workers $WORKERS \
    --code-timeout $CODE_TIMEOUT \
    --api-key "$API_KEY" \
    --api-base "$API_BASE" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "Generation Complete!"
echo "Outputs:"
echo "  - JSON/CSV/Statistics in: $OUTPUT_DIR"
echo "================================================================================"
echo ""

