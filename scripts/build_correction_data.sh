#!/bin/bash

################################################################################
# Self-Correction Training Data Generation Script
#
# Generates multi-turn self-correction trajectories for fine-tuning
# the interactive code agent model.
#
# Error Type Distribution:
#   - logic_error:  50% - Code runs but gives wrong answer
#   - name_error:   20% - NameError (undefined variable)
#   - wrong_output: 15% - Output doesn't make sense
#   - syntax_error:  5% - Python syntax error
#   - type_error:    5% - Type mismatch
#   - multi_step:    5% - Multiple corrections needed
#
# Data Sources:
#   - GSM8K train: ~60% (numerical answers, good for code)
#   - rstar:       ~40% (diverse math problems)
#
# Usage:
#   1. Configure API key and parameters below
#   2. Run: bash scripts/build_correction_data.sh
#
################################################################################

# ============================================================================
# Model Configuration
# ============================================================================

# Model for generation (recommended: fast reasoning models)
# Options:
#   - x-ai/grok-3-fast           (fast, good quality)
#   - x-ai/grok-4-1-fast-reasoning (better reasoning)
#   - deepseek/deepseek-reasoner-v3.1 (long thinking)
# Only single model supported (not comma-separated list)
MODEL="x-ai/grok-4-1-fast-reasoning"


# ============================================================================
# Data Configuration
# ============================================================================

# Total number of samples to generate
# GSM8K train has ~7473 samples, MATH train has ~12000 samples
COUNT=100000	

# Data source selection
# Options: auto, gsm8k, math, rstar, gsm8k+math
# - gsm8k: Only GSM8K train (~7473 samples)
# - math: Only MATH train (~12000 samples)
# - rstar: Only rstar (~100k samples)
# - gsm8k+math: 50% GSM8K + 50% MATH
# - auto: Use ratios below
SOURCE="rstar"

# Ratios (only used when SOURCE="auto")
GSM8K_RATIO=0.5
MATH_RATIO=0.5


# ============================================================================
# Generation Configuration
# ============================================================================

# Max retry attempts per sample
# Higher = better success rate but slower
MAX_ATTEMPTS=3

# Temperatures for retry attempts
# More diverse temperatures help with difficult problems
TEMPERATURES="0.7,0.8,0.9"

# Max tokens for generation
# Self-correction trajectories can be long
MAX_TOKENS=2048

# Number of parallel workers
# WARNING: Too high (>8) will trigger API rate limits!
# Recommended: 4-8 for stable generation
WORKERS=128


# ============================================================================
# API Configuration
# ============================================================================

# API Key (AIML API)
API_KEY="7111797f767441e88949afd7a633a2dd"

# API Base URL
API_BASE="https://api.aimlapi.com/v1"


# ============================================================================
# Output Configuration
# ============================================================================

# Output directory
OUTPUT_DIR="data/agent_sft/correction_gsm8k_$(date +%m%d_%H%M)"


################################################################################
# Run Generation
################################################################################

echo "================================================================================"
echo "           Self-Correction Training Data Generation"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Total Samples: $COUNT"
echo "  Source: $SOURCE"
echo "  Max Attempts: $MAX_ATTEMPTS"
echo "  Workers: $WORKERS"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Error Type Distribution:"
echo "  logic_error:  50%"
echo "  name_error:   20%"
echo "  wrong_output: 15%"
echo "  syntax_error:  5%"
echo "  type_error:    5%"
echo "  multi_step:    5%"
echo ""
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the generation script
python -m dataset.build_correction_data \
    --model "$MODEL" \
    --count $COUNT \
    --source "$SOURCE" \
    --gsm8k-ratio $GSM8K_RATIO \
    --math-ratio $MATH_RATIO \
    --max-attempts $MAX_ATTEMPTS \
    --temperatures "$TEMPERATURES" \
    --max-tokens $MAX_TOKENS \
    --workers $WORKERS \
    --api-key "$API_KEY" \
    --api-base "$API_BASE" \
    --output-dir "$OUTPUT_DIR"

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Generation Complete!"
    echo "================================================================================"
    echo ""
    echo "Output files in: $OUTPUT_DIR"
    echo ""
    ls -lh "$OUTPUT_DIR"
    echo ""
    
    # Show sample
    echo "Sample trajectory:"
    echo "---"
    python -c "import json; import glob; files = glob.glob('$OUTPUT_DIR/correction_data_*.json'); f = open(files[0]) if files else None; data = json.load(f) if f else []; print(f'Query: {data[0][\"query\"][:100]}...') if data else print('No data'); print(f'Response: {data[0][\"response\"][:300]}...') if data else None"
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "Generation failed! Check logs for errors."
    exit 1
fi

