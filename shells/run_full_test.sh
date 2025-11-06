#!/bin/bash
# Run full evaluation suite
# Tests all 4 models on 2 datasets with 100 samples each in both standard and thinking modes

set -e

# Configuration
ROUND_NAME="round1_full"
COUNT=100
DETAILED="false"  # Set to "true" for token-by-token output, "false" for compact progress only
MODELS=("Qwen3-0.6B" "Qwen3-1.7B" "Qwen3-4B-Thinking-2507" "Qwen3-8B")
DATASETS=("gsm8k" "math")
MODES=("standard")  # Only standard mode for now (thinking mode needs debugging)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "SLM-Math Full Evaluation Suite"
echo "=============================================="
echo "Round: $ROUND_NAME"
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "Modes: ${MODES[@]}"
echo "Count per test: $COUNT"
echo "Detailed output: $DETAILED"
echo "Total tests: $((${#MODELS[@]} * ${#DATASETS[@]} * ${#MODES[@]}))"
echo "=============================================="
echo ""

TOTAL_TESTS=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#MODES[@]}))
CURRENT_TEST=0

START_TIME=$(date +%s)

cd "$PROJECT_ROOT"
source ~/anaconda3/bin/activate slm_math

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for MODE in "${MODES[@]}"; do
            CURRENT_TEST=$((CURRENT_TEST + 1))
            
            echo ""
            echo "=============================================="
            echo "Test $CURRENT_TEST of $TOTAL_TESTS"
            echo "Model: $MODEL"
            echo "Dataset: $DATASET"
            echo "Mode: $MODE"
            echo "=============================================="
            echo ""
            
            # Run evaluation
            python scripts/evaluate_batch.py \
                --model "$MODEL" \
                --round "$ROUND_NAME" \
                --dataset "$DATASET" \
                --count "$COUNT" \
                --mode "$MODE" \
                --detailed "$DETAILED"
            
            if [ $? -ne 0 ]; then
                echo "ERROR: Test failed for $MODEL on $DATASET ($MODE mode)"
                echo "Continuing with remaining tests..."
            fi
            
            echo ""
            echo "Completed test $CURRENT_TEST of $TOTAL_TESTS"
            echo ""
            
            # Small delay between tests
            sleep 5
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "Full Evaluation Suite Completed!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results saved in: results/$ROUND_NAME/"
echo "=============================================="
echo ""
echo "View results:"
echo "  cat results/$ROUND_NAME/metrics_standard.txt"
echo "  cat results/$ROUND_NAME/metrics_thinking.txt"
echo "  cat results/$ROUND_NAME/metrics_standard.csv"
echo "  cat results/$ROUND_NAME/metrics_thinking.csv"

