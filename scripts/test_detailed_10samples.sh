#!/bin/bash

################################################################################
# 详细测试脚本 - 10个样本
# 用于手动检查错题日志并优化workflow
################################################################################

set -e

echo "════════════════════════════════════════════════════════════════════════════"
echo "  详细测试 - 10个样本分析"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# 配置
MODEL="Qwen2.5-Math-1.5B"
DATASET="gsm8k"
COUNT=10
MAX_ITERATIONS=3
DETAILED="true"
CUDA_DEVICE=0

echo "配置信息:"
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  样本数: $COUNT"
echo "  最大迭代: $MAX_ITERATIONS"
echo "  GPU: $CUDA_DEVICE"
echo ""

# 导出GPU设置
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "════════════════════════════════════════════════════════════════════════════"
echo "测试 1: Stateless 模式"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent solver_checker_summarizer \
    --round detailed_test_stateless \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

echo ""
echo "✓ Stateless 测试完成"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "测试 2: Chat 模式"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

python -m evaluation.eval_agent \
    --model "$MODEL" \
    --agent solver_checker_summarizer_chat \
    --round detailed_test_chat \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

echo ""
echo "✓ Chat 测试完成"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "测试完成 - 结果位置"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# 找到最新的结果目录
STATELESS_DIR=$(ls -td results/detailed_test_stateless_* 2>/dev/null | head -1)
CHAT_DIR=$(ls -td results/detailed_test_chat_* 2>/dev/null | head -1)

if [ -n "$STATELESS_DIR" ]; then
    echo "Stateless 结果: $STATELESS_DIR"
    echo "  - metrics.csv"
    echo "  - analysis_report.txt"
    echo "  - answers/*.json"
fi

if [ -n "$CHAT_DIR" ]; then
    echo ""
    echo "Chat 结果: $CHAT_DIR"
    echo "  - metrics.csv"
    echo "  - analysis_report.txt"
    echo "  - answers/*.json"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"


