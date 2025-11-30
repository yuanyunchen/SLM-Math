#!/bin/bash

################################################################################
# 测试修复后的 Workflow
################################################################################

set -e

echo "════════════════════════════════════════════════════════════════════════════"
echo "  测试修复后的 Workflow - 10样本"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "修复内容:"
echo "  ✓ Summarizer: 添加 repetition_penalty=1.5"
echo "  ✓ Summarizer: 降低 temperature=0.3"
echo "  ✓ Summarizer: 添加垃圾输出检测"
echo "  ✓ Summarizer: 添加 Fallback 机制"
echo ""

MODEL="Qwen2.5-Math-1.5B"
DATASET="gsm8k"
COUNT=10
MAX_ITERATIONS=3
DETAILED="true"
export CUDA_VISIBLE_DEVICES=0

echo "════════════════════════════════════════════════════════════════════════════"
echo "测试 Stateless 模式 (修复后)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

conda run -n slm_math python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent solver_checker_summarizer \
    --round fixed_workflow_stateless \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "结果对比"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# 找到最新的结果目录
FIXED_DIR=$(ls -td results/fixed_workflow_stateless_* 2>/dev/null | head -1)
OLD_DIR="results/detailed_test_stateless_Qwen2.5-Math-1.5B_gsm8k_10_1124_2017"

if [ -n "$FIXED_DIR" ] && [ -d "$OLD_DIR" ]; then
    echo "修复前 (原始测试):"
    grep "^Qwen" "$OLD_DIR/metrics.csv" | awk -F',' '{printf "  准确率: %s%%\n  False Positives: %s\n  False Negatives: %s\n", $6*100, $18, $19}'
    
    echo ""
    echo "修复后 (当前测试):"
    grep "^Qwen" "$FIXED_DIR/metrics.csv" | awk -F',' '{printf "  准确率: %s%%\n  False Positives: %s\n  False Negatives: %s\n", $6*100, $18, $19}'
    
    echo ""
    echo "详细结果: $FIXED_DIR"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"


