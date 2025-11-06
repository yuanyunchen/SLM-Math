#!/bin/bash

################################################################################
# Evaluation Settings

# Model to evaluate
# Options: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B-Thinking-2507, Qwen3-8B
MODEL="Qwen3-0.6B"

# Test round name (used for organizing results)
# Results will be saved to: results/<ROUND_NAME>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/
ROUND_NAME="test"

# Dataset to use
# Options: gsm8k (grade school math), math (competition math)
DATASET="math"

# Number of test cases to run (set to 0 to run the entire dataset)
COUNT=100

# Evaluation mode
# Options:
#   - standard: Direct answer generation
#   - thinking: Detailed reasoning with Analysis + Chain-of-Thought + Final Answer
MODE="standard"

# Detailed output mode
DETAILED="true"


################################################################################
# Run evaluation

cd "$(dirname "$0")/.."

source ~/anaconda3/bin/activate slm_math

python scripts/evaluate_batch.py \
    --model "$MODEL" \
    --round "$ROUND_NAME" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --mode "$MODE" \
    --detailed "$DETAILED"

