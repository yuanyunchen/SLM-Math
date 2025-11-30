#!/bin/bash

################################################################################
# Quick Comparison Test: Base vs Checkpoint
# Run a small test (10 samples) to verify checkpoint is working
################################################################################

set -e

export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen2.5-Math-1.5B"
CHECKPOINT="checkpoints/lora_r16_20251124_130958/final_model"
DATASET="gsm8k"
COUNT=10

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    Quick Comparison Test                                     ║"
echo "║           Base Model vs LoRA Checkpoint (10 samples)                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: Base model
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Evaluating BASE MODEL (no checkpoint)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m evaluation.eval \
    --model "$MODEL" \
    --round "quick_test_base" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --mode "standard" \
    --detailed "false" 2>&1 | grep -E "(Accuracy|correct|Loading model)"

BASE_RESULT=$(find results/quick_test_base_* -name "metrics.csv" 2>/dev/null | head -1)

# Test 2: Checkpoint model
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Evaluating CHECKPOINT MODEL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m evaluation.eval \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --round "quick_test_checkpoint" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --mode "standard" \
    --detailed "false" 2>&1 | grep -E "(Accuracy|correct|Loading|LoRA|adapter)"

CKPT_RESULT=$(find results/quick_test_checkpoint_* -name "metrics.csv" 2>/dev/null | head -1)

# Compare results
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                          Comparison Results                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "$BASE_RESULT" ] && [ -f "$CKPT_RESULT" ]; then
    echo "Base Model Results:"
    cat "$BASE_RESULT"
    echo ""
    echo "Checkpoint Model Results:"
    cat "$CKPT_RESULT"
    echo ""
    
    BASE_ACC=$(tail -1 "$BASE_RESULT" | cut -d',' -f6)
    CKPT_ACC=$(tail -1 "$CKPT_RESULT" | cut -d',' -f6)
    
    echo "Summary:"
    echo "  Base Model Accuracy:       $BASE_ACC"
    echo "  Checkpoint Model Accuracy: $CKPT_ACC"
    echo ""
    
    # Use python to compare floating point numbers
    python3 -c "
base = float('$BASE_ACC')
ckpt = float('$CKPT_ACC')
diff = (ckpt - base) * 100

print('Accuracy Difference: {:+.1f}%'.format(diff))
print()

if abs(diff) < 0.5:
    print('⚠ WARNING: Accuracies are nearly identical!')
    print('  Possible reasons:')
    print('  1. Sample size too small (only 10 samples)')
    print('  2. Checkpoint not loaded correctly')
    print('  3. Model not trained enough')
    print()
    print('Next steps:')
    print('  - Check logs for \"LoRA adapter loaded successfully\"')
    print('  - Run comparison script: python scripts/compare_base_vs_checkpoint.py')
    print('  - Test with more samples (COUNT=100)')
elif diff > 0:
    print('✓ Checkpoint model is BETTER (+{:.1f}%)'.format(diff))
else:
    print('✗ Checkpoint model is WORSE ({:.1f}%)'.format(diff))
"
else
    echo "⚠ Could not find result files"
    echo "Base: $BASE_RESULT"
    echo "Checkpoint: $CKPT_RESULT"
fi

echo ""
echo "Full results saved in:"
echo "  Base:       results/quick_test_base_*/"
echo "  Checkpoint: results/quick_test_checkpoint_*/"
















