#!/bin/bash

################################################################################
# Test Checkpoint Loading
# Verify that checkpoints are correctly loaded
################################################################################

set -e

export CUDA_VISIBLE_DEVICES=1

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                 Checkpoint Loading Test                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Test with a simple 3-sample evaluation
MODEL="Qwen2.5-Math-1.5B"
CHECKPOINT="checkpoints/lora_r16_20251124_130958/checkpoint-891"
DATASET="gsm8k"
COUNT=3

echo "Test Configuration:"
echo "  Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: $DATASET (first $COUNT samples)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running test evaluation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m evaluation.eval \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --round "test_checkpoint_loading" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --mode "standard" \
    --detailed "true" \
    --inference_backend "transformers" 2>&1 | tee /tmp/checkpoint_test.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checking if checkpoint was loaded..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check log for checkpoint loading messages
if grep -q "Loading from checkpoint" /tmp/checkpoint_test.log; then
    echo "✓ Checkpoint path detected in logs"
fi

if grep -q "Detected LoRA adapter" /tmp/checkpoint_test.log; then
    echo "✓ LoRA adapter detected"
fi

if grep -q "Loading base model from" /tmp/checkpoint_test.log; then
    echo "✓ Base model loaded"
fi

if grep -q "LoRA adapter loaded successfully" /tmp/checkpoint_test.log; then
    echo "✓ LoRA adapter loaded successfully"
else
    echo "⚠ Warning: Did not find 'LoRA adapter loaded successfully' message"
fi

echo ""
echo "To compare with base model, run:"
echo "  python -m evaluation.eval --model \"$MODEL\" --round \"test_base\" --dataset \"$DATASET\" --count $COUNT --mode \"standard\""

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Test Complete                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
















