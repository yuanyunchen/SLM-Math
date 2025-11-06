#!/bin/bash

################################################################################

# Model to evaluate
# Options: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B-Thinking-2507, Qwen3-8B
MODEL="Qwen3-0.6B"

# Test round name (used for organizing results in results/<ROUND_NAME>/)
ROUND_NAME="test"

# Dataset to use
# Options: gsm8k (grade school math), math (competition math)
DATASET="gsm8k"

# Number of test cases to run
COUNT=10

# Evaluation mode
# Options:
#   - standard: Direct answer generation
#   - thinking: Detailed reasoning with Analysis + Chain-of-Thought + Final Answer
MODE="thinking"

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

################################################################################

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo ""
echo "Check results:"
echo "  Summary: cat results/$ROUND_NAME/metrics_$MODE.txt"
echo "  CSV:     cat results/$ROUND_NAME/metrics_$MODE.csv"
echo "  Answers: cat results/$ROUND_NAME/answers/${MODEL}_${DATASET}_${MODE}_answers.json"
echo "  Log:     cat results/$ROUND_NAME/log/${MODEL}_${DATASET}_${MODE}.log"
echo "================================================================================"
