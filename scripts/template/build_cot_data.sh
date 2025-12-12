#!/bin/bash

################################################################################
# Optimized CoT Data Generation Script (Multi-Model Multi-Dataset)
#
# Features:
#   - Two-round architecture with multi-model support
#   - Round 1: Multiple teacher models with retry
#   - Round 2: Multiple expert models for failures
#   - Multiple datasets in one run
#   - Clean final output (only successful samples)
#
# Architecture:
#   1. Round 1: Test all teacher models on all samples
#      - Each model retries up to ROUND1_ATTEMPTS times
#      - If ANY model succeeds, keep ALL correct solutions
#      - Only samples where ALL models fail go to Round 2
#   
#   2. Round 2: Test all expert models on failed samples
#      - Each model retries up to ROUND2_ATTEMPTS times
#      - Keep ALL correct solutions
#      - Discard samples that fail in both rounds
#
# Output:
#   - Single JSON file with all successful solutions
#   - Single CSV file with tabular format
#   - Statistics TXT file with per-dataset and per-model metrics
#
# Prerequisites:
#   - AIML API key
#   - Datasets downloaded in data/ directory
#
# Usage:
#   1. Configure parameters below
#   2. Run: bash scripts/build_cot_data.sh
#
################################################################################

# ============================================================================
# Model Configuration
# ============================================================================

# Teacher Models (Round 1) - comma-separated list
# These models are tested first. If any succeeds, the sample doesn't go to Round 2.
# All correct solutions from the same round are kept.
#
# Recommended Teacher Models (Good performance, cost-effective):
TEACHER_MODELS="x-ai/grok-4-1-fast-reasoning,deepseek/deepseek-reasoner-v3.1"

# Expert Models (Round 2) - comma-separated list
# These models handle samples that failed in Round 1.
# Typically more powerful/expensive models.
#
# Recommended Expert Models (High accuracy for difficult problems):
EXPERT_MODELS="alibaba/qwen3-235b-a22b-thinking-2507,minimax/m2"

# ============== Model Reference (validated 2025-11-23) ==============
# Pricing source: https://aimlapi.com/ai-ml-api-pricing
#
# Model Name                            | Price ($/1M)    | Think length | Accuracy | Suggested use
# --------------------------------------|----------------|--------------|----------|------------------
# ✅ alibaba/qwen3-next-80b-a3b-thinking   | $0.158/$1.600  | ~1,700 chars | ★★★★    | 🏆 Teacher (best value)
# ✅ x-ai/grok-4-fast-reasoning            | $0.210/$0.525  | ~1,300 chars | ★★★★    | 🏆 Teacher (fast)
# ✅ x-ai/grok-4-1-fast-reasoning          | $0.210/$0.530  | ~1,800 chars | ★★★★☆   | 🏆 Teacher (recommended)
# ✅ alibaba/qwen3-235b-a22b-thinking-2507 | $0.242/$2.415  | ~2,200 chars | ★★★★★   | 🏆 Expert (high quality)
# ✅ deepseek/deepseek-reasoner-v3.1       | $0.294/$0.441  | ~5,600 chars | ★★★★★   | 🏆 Teacher (long thinking)
# ✅ deepseek-reasoner                     | $0.294/$0.441  | ~500 chars   | ★★★★    | Teacher (standard)
# ✅ minimax/m2                            | $0.315/$1.260  | ~1,900 chars | ★★★★☆   | Expert (stable)
#
# 💡 Combo recommendations:
#   Budget: grok-4-1-fast-reasoning (R1) → qwen3-235b (R2)
#   Balanced: deepseek-reasoner-v3.1,grok-4-1 (R1) → qwen3-235b,minimax/m2 (R2)
#   High quality: deepseek-v3.1,qwen3-next-80b,grok-4-1 (R1) → qwen3-235b,minimax/m2 (R2)


# ============================================================================
# Dataset Configuration
# ============================================================================

# Datasets - comma-separated list
# Format: dataset-split (e.g., gsm8k-train, math-train)
# Available: gsm8k-train, gsm8k-test, math-train, math500, competition_math-train
DATASETS="gsm8k-train,math-train"

# Number of samples per dataset (0 = all)
COUNT=0

# Starting index (useful for testing or resuming)
START_INDEX=0


# ============================================================================
# Round Configuration
# ============================================================================

# Round 1 (Teacher Models)
ROUND1_ATTEMPTS=3           # Max retry attempts per model
ROUND1_EFFORT="medium"      # Thinking effort: low, medium, high, very_high
ROUND1_TEMPS="0.7,0.5,0.9"  # Temperature schedule for retries

# Round 2 (Expert Models)
ROUND2_ATTEMPTS=5           # Max retry attempts per model
ROUND2_EFFORT="high"        # Usually higher than Round 1
ROUND2_TEMPS="0.7,0.5,0.9,0.3,1.0"  # More diverse for difficult problems


# ============================================================================
# Thinking Effort Reference
# ============================================================================
# Effort level | Thinking length | Recommended scenario
# ------------|-----------------|---------------------
# low         | ~300-500 chars  | Quick checks, simple problems
# medium      | ~500-800 chars  | Balanced quality (recommended for R1)
# high        | ~800-1500 chars | High quality, deeper reasoning (recommended for R2)
# very_high   | ~1500-3000 chars| Maximum quality, complex problems
#
# Temperature reference:
#   - 0.0-0.3: Highly deterministic, use when stability matters
#   - 0.5-0.7: Moderate diversity, standard CoT generation (recommended)
#   - 0.9-1.0: High diversity, explore many solutions


# ============================================================================
# General Configuration
# ============================================================================

# Max tokens for model output
MAX_TOKENS=4096

# Number of parallel workers
# Suggested: 4-8 for teacher models, 2-4 for expert models (avoid rate limits)
WORKERS=8

# API Configuration
API_KEY="7111797f767441e88949afd7a633a2dd"
API_BASE="https://api.aimlapi.com/v1"

# Output directory
# Auto-generated name: cot_data_<datasets>_<timestamp>
OUTPUT_DIR="data/cot_generated/multi_model_$(date +%m%d_%H%M)"


################################################################################
# Build and Run
################################################################################

echo "================================================================================"
echo "               CoT Data Generation - Multi-Model Multi-Dataset"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Teacher Models: $TEACHER_MODELS"
echo "  Expert Models: $EXPERT_MODELS"
echo "  Datasets: $DATASETS"
echo "  Round 1: $ROUND1_ATTEMPTS attempts, $ROUND1_EFFORT effort"
echo "  Round 2: $ROUND2_ATTEMPTS attempts, $ROUND2_EFFORT effort"
echo "  Workers: $WORKERS"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "================================================================================"
echo ""

# Build command
python -m dataset.build_CoT_data \
    --teacher "$TEACHER_MODELS" \
    --expert "$EXPERT_MODELS" \
    --dataset "$DATASETS" \
    --count $COUNT \
    --start $START_INDEX \
    --round1-attempts $ROUND1_ATTEMPTS \
    --round1-effort $ROUND1_EFFORT \
    --round1-temps "$ROUND1_TEMPS" \
    --round2-attempts $ROUND2_ATTEMPTS \
    --round2-effort $ROUND2_EFFORT \
    --round2-temps "$ROUND2_TEMPS" \
    --max-tokens $MAX_TOKENS \
    --workers $WORKERS \
    --api-key "$API_KEY" \
    --api-base "$API_BASE" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "Generation Complete!"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  - JSON: $OUTPUT_DIR/cot_data_*.json"
echo "  - CSV: $OUTPUT_DIR/cot_data_*.csv"
echo "  - Statistics: $OUTPUT_DIR/cot_data_*_statistics.txt"
echo ""
echo "================================================================================"
