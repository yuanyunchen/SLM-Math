#!/bin/bash

################################################################################
# 完整评估脚本 - 所有模型配置和工作流
# 
# 用途: 系统性评估所有agent在不同模型配置下的性能
# 
# 模型配置:
#   1. Qwen2.5-Math-1.5B (solver only)
#   2. Qwen3-1.7B (solver only)
#   3. Qwen2.5-Math-1.5B (solver) + Qwen3-1.7B (checker)
# 
# 工作流:
#   - Base Direct (配置1, 2)
#   - Majority Vote (配置1, 2)
#   - Solver-Checker系列 (配置2, 3)
#   - Plan-and-Reflection (配置1, 2)
#
# 总测试数: 18个
################################################################################

set -e  # Exit on error

################################################################################
# 配置参数
################################################################################

# Sample数量 (可调整)
# 推荐: 40 (约8.8小时) 或 45 (约9.9小时)
COUNT=${1:-40}

# Dataset
DATASET="gsm8k"

# Round名称 (添加时间戳)
TIMESTAMP=$(date +"%m%d_%H%M")
ROUND_NAME="full_eval_${TIMESTAMP}"

# Batch processing parameters (NEW)
BATCH_SIZE=${2:-1}  # Default: 1 (no batching), recommended: 16-32 for A100
INFERENCE_BACKEND=${3:-transformers}  # Default: transformers, options: transformers, vllm

# 详细输出
DETAILED="false"  # 改为true可以看到详细过程，但会很慢

# 迭代/运行次数限制
MAX_ITERATIONS=5
NUM_RUNS=5

################################################################################
# 颜色输出
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo "================================================================================"
    echo "  $1"
    echo "================================================================================"
    echo ""
}

################################################################################
# 预检查
################################################################################

log_section "预检查"

# 检查Python环境
if ! command -v python &> /dev/null; then
    log_error "Python not found!"
    exit 1
fi

log_info "Python version: $(python --version)"

# 检查工作目录
cd /Users/yuanyunchen/Desktop/GitHub/SLM-Math || exit 1
log_info "Working directory: $(pwd)"

# 检查模型文件
MODEL1="Qwen2.5-Math-1.5B"
MODEL2="Qwen3-1.7B"

if [ ! -d "pretrained_models/$MODEL1" ]; then
    log_error "Model $MODEL1 not found!"
    exit 1
fi

if [ ! -d "pretrained_models/$MODEL2" ]; then
    log_error "Model $MODEL2 not found!"
    exit 1
fi

log_success "Models found: $MODEL1, $MODEL2"

# 显示配置
log_section "评估配置"
echo "Round Name:       $ROUND_NAME"
echo "Dataset:          $DATASET"
echo "Samples/Test:     $COUNT"
echo "Batch Size:       $BATCH_SIZE"
echo "Backend:          $INFERENCE_BACKEND"
echo "Max Iterations:   $MAX_ITERATIONS"
echo "Num Runs (MV):    $NUM_RUNS"
echo "Detailed Output:  $DETAILED"
echo ""

# 估算时间
ESTIMATED_HOURS=$(python3 -c "
workflows = [
    (2, 10),   # Base Direct (1-2)
    (2, 55),   # Majority Vote (3-4)
    (2, 35),   # Stateless (5-6)
    (2, 45),   # Summarizer (7-8)
    (2, 50),   # Summarizer Chat (9-10)
    (2, 40),   # With Tools (11-12)
    (2, 40),   # Trivial Chat (13-14)
    (2, 55),   # Chat Opt (15-16)
    (2, 65),   # Plan-and-Reflection (17-18)
]
total = sum(configs * $COUNT * time for configs, time in workflows)
print(f'{total/3600:.1f}')
")

echo "预计总时间:      约 ${ESTIMATED_HOURS} 小时"
echo ""

read -p "确认开始评估? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_warning "评估已取消"
    exit 0
fi

################################################################################
# 开始评估
################################################################################

START_TIME=$(date +%s)
TOTAL_TESTS=18
CURRENT_TEST=0

log_section "开始完整评估"

################################################################################
# 1. Base Direct - Qwen2.5-Math-1.5B
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Base Direct - $MODEL1"

python -m evaluation.eval \
    --model "$MODEL1" \
    --round "${ROUND_NAME}_base_${MODEL1}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --mode "standard" \
    --detailed "$DETAILED" \
    --batch_size "$BATCH_SIZE" \
    --inference_backend "$INFERENCE_BACKEND"

log_success "完成: Base Direct - $MODEL1"

################################################################################
# 2. Base Direct - Qwen3-1.7B
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Base Direct - $MODEL2"

python -m evaluation.eval \
    --model "$MODEL2" \
    --round "${ROUND_NAME}_base_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --mode "standard" \
    --detailed "$DETAILED" \
    --batch_size "$BATCH_SIZE" \
    --inference_backend "$INFERENCE_BACKEND"

log_success "完成: Base Direct - $MODEL2"

################################################################################
# 3. Majority Vote - Qwen2.5-Math-1.5B
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Majority Vote - $MODEL1"

python -m evaluation.eval_agent \
    --model "$MODEL1" \
    --agent "majority_vote" \
    --round "${ROUND_NAME}_mv_${MODEL1}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --num_runs "$NUM_RUNS" \
    --temperature 0.7 \
    --top_p 0.95 \
    --detailed "$DETAILED" \
    --batch_size "$BATCH_SIZE" \
    --inference_backend "$INFERENCE_BACKEND"

log_success "完成: Majority Vote - $MODEL1"

################################################################################
# 4. Majority Vote - Qwen3-1.7B
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Majority Vote - $MODEL2"

python -m evaluation.eval_agent \
    --model "$MODEL2" \
    --agent "majority_vote" \
    --round "${ROUND_NAME}_mv_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --num_runs "$NUM_RUNS" \
    --temperature 0.7 \
    --top_p 0.95 \
    --detailed "$DETAILED" \
    --batch_size "$BATCH_SIZE" \
    --inference_backend "$INFERENCE_BACKEND"

log_success "完成: Majority Vote - $MODEL2"

################################################################################
# 5. Solver-Checker Stateless (Qwen2.5-Math + Qwen3)
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Stateless"

python -m evaluation.eval_agent \
    --model "$MODEL1" \
    --checker_model "$MODEL2" \
    --agent "solver_checker" \
    --round "${ROUND_NAME}_stateless" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Stateless"

################################################################################
# 6. Solver-Checker Stateless - Qwen3 only
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Stateless - $MODEL2 only"

python -m evaluation.eval_agent \
    --model "$MODEL2" \
    --agent "solver_checker" \
    --round "${ROUND_NAME}_stateless_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Stateless - $MODEL2 only"

################################################################################
# 7. Solver-Checker Summarizer (Qwen2.5-Math + Qwen3)
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Summarizer"

python -m evaluation.eval_agent \
    --model "$MODEL1" \
    --checker_model "$MODEL2" \
    --agent "solver_checker_summarizer" \
    --round "${ROUND_NAME}_summarizer" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Summarizer"

################################################################################
# 8. Solver-Checker Summarizer - Qwen3 only
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Summarizer - $MODEL2 only"

python -m evaluation.eval_agent \
    --model "$MODEL2" \
    --agent "solver_checker_summarizer" \
    --round "${ROUND_NAME}_summarizer_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Summarizer - $MODEL2 only"

################################################################################
# 9. Solver-Checker Summarizer Chat (Qwen2.5-Math only)
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Summarizer Chat"

python -m evaluation.eval_agent \
    --model "$MODEL1" \
    --checker_model "$MODEL2" \
    --agent "solver_checker_summarizer_chat" \
    --round "${ROUND_NAME}_summarizer_chat" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Summarizer Chat"

################################################################################
# 10. Solver-Checker Summarizer Chat - Qwen3 only
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Summarizer Chat - $MODEL2 only"

python -m evaluation.eval_agent \
    --model "$MODEL2" \
    --agent "solver_checker_summarizer_chat" \
    --round "${ROUND_NAME}_summarizer_chat_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Summarizer Chat - $MODEL2 only"

################################################################################
# 11. Solver-Checker With Tools (Qwen2.5-Math + Qwen3)
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker With Tools"

python -m evaluation.eval_agent \
    --model "$MODEL1" \
    --checker_model "$MODEL2" \
    --agent "solver_checker_with_tools" \
    --round "${ROUND_NAME}_with_tools" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --enable_solver_tools "true" \
    --enable_checker_tools "true" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker With Tools"

################################################################################
# 12. Solver-Checker With Tools - Qwen3 only
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker With Tools - $MODEL2 only"

python -m evaluation.eval_agent \
    --model "$MODEL2" \
    --agent "solver_checker_with_tools" \
    --round "${ROUND_NAME}_with_tools_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --enable_solver_tools "true" \
    --enable_checker_tools "true" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker With Tools - $MODEL2 only"

################################################################################
# 13. Solver-Checker Trivial Chat (Qwen2.5-Math only)
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Trivial Chat"

python -m evaluation.eval_agent \
    --model "$MODEL1" \
    --agent "solver_checker_trivial_chat" \
    --round "${ROUND_NAME}_trivial_chat" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Trivial Chat"

################################################################################
# 14. Solver-Checker Trivial Chat - Qwen3 only
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Trivial Chat - $MODEL2 only"

python -m evaluation.eval_agent \
    --model "$MODEL2" \
    --agent "solver_checker_trivial_chat" \
    --round "${ROUND_NAME}_trivial_chat_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Trivial Chat - $MODEL2 only"

################################################################################
# 15. Solver-Checker Chat (Optimized) (Qwen2.5-Math only)
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Chat (Optimized)"

python -m evaluation.eval_agent \
    --model "$MODEL1" \
    --agent "solver_checker_chat" \
    --round "${ROUND_NAME}_chat_opt" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Chat (Optimized)"

################################################################################
# 16. Solver-Checker Chat (Optimized) - Qwen3 only
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Solver-Checker Chat (Optimized) - $MODEL2 only"

python -m evaluation.eval_agent \
    --model "$MODEL2" \
    --agent "solver_checker_chat" \
    --round "${ROUND_NAME}_chat_opt_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --detailed "$DETAILED"

log_success "完成: Solver-Checker Chat (Optimized) - $MODEL2 only"

################################################################################
# 17. Plan-and-Reflection - Qwen2.5-Math-1.5B
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Plan-and-Reflection - $MODEL1"

python -m evaluation.eval_agent \
    --model "$MODEL1" \
    --agent "plan_and_reflection" \
    --round "${ROUND_NAME}_planref_${MODEL1}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --max_subproblems 5 \
    --detailed "$DETAILED"

log_success "完成: Plan-and-Reflection - $MODEL1"

################################################################################
# 12. Plan-and-Reflection - Qwen3-1.7B
################################################################################

CURRENT_TEST=$((CURRENT_TEST + 1))
log_section "[$CURRENT_TEST/$TOTAL_TESTS] Plan-and-Reflection - $MODEL2"

python -m evaluation.eval_agent \
    --model "$MODEL2" \
    --agent "plan_and_reflection" \
    --round "${ROUND_NAME}_planref_${MODEL2}" \
    --dataset "$DATASET" \
    --count "$COUNT" \
    --max_iterations "$MAX_ITERATIONS" \
    --max_subproblems 5 \
    --detailed "$DETAILED"

log_success "完成: Plan-and-Reflection - $MODEL2"

################################################################################
# 评估完成
################################################################################

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

log_section "评估完成！"

echo "总耗时: ${HOURS}小时 ${MINUTES}分钟"
echo "完成测试: $TOTAL_TESTS 个"
echo "总样本数: $((COUNT * TOTAL_TESTS))"
echo ""
echo "结果目录: results/"
echo "Round前缀: ${ROUND_NAME}"
echo ""

# 生成结果汇总
log_info "生成结果汇总..."

SUMMARY_FILE="results/${ROUND_NAME}_SUMMARY.txt"

cat > "$SUMMARY_FILE" << EOF
================================================================================
完整评估结果汇总
================================================================================

评估时间: $(date)
Round: $ROUND_NAME
Dataset: $DATASET
Samples/Test: $COUNT
总耗时: ${HOURS}h ${MINUTES}m

================================================================================
测试列表
================================================================================

1.  Base Direct - $MODEL1
    目录: results/${ROUND_NAME}_base_${MODEL1}_${DATASET}_${COUNT}_*/

2.  Base Direct - $MODEL2
    目录: results/${ROUND_NAME}_base_${MODEL2}_${DATASET}_${COUNT}_*/

3.  Majority Vote - $MODEL1
    目录: results/${ROUND_NAME}_mv_${MODEL1}_${DATASET}_${COUNT}_*/

4.  Majority Vote - $MODEL2
    目录: results/${ROUND_NAME}_mv_${MODEL2}_${DATASET}_${COUNT}_*/

5.  Solver-Checker Stateless (2.5+3)
    目录: results/${ROUND_NAME}_stateless_${MODEL1}_${DATASET}_${COUNT}_*/

6.  Solver-Checker Stateless (3 only)
    目录: results/${ROUND_NAME}_stateless_${MODEL2}_${DATASET}_${COUNT}_*/

7.  Solver-Checker Summarizer (2.5+3)
    目录: results/${ROUND_NAME}_summarizer_${MODEL1}_${DATASET}_${COUNT}_*/

8.  Solver-Checker Summarizer (3 only)
    目录: results/${ROUND_NAME}_summarizer_${MODEL2}_${DATASET}_${COUNT}_*/

9.  Solver-Checker Summarizer Chat (2.5 only)
    目录: results/${ROUND_NAME}_summarizer_chat_${MODEL1}_${DATASET}_${COUNT}_*/

10. Solver-Checker Summarizer Chat (3 only)
    目录: results/${ROUND_NAME}_summarizer_chat_${MODEL2}_${DATASET}_${COUNT}_*/

11. Solver-Checker With Tools (2.5+3)
    目录: results/${ROUND_NAME}_with_tools_${MODEL1}_${DATASET}_${COUNT}_*/

12. Solver-Checker With Tools (3 only)
    目录: results/${ROUND_NAME}_with_tools_${MODEL2}_${DATASET}_${COUNT}_*/

13. Solver-Checker Trivial Chat (2.5 only)
    目录: results/${ROUND_NAME}_trivial_chat_${MODEL1}_${DATASET}_${COUNT}_*/

14. Solver-Checker Trivial Chat (3 only)
    目录: results/${ROUND_NAME}_trivial_chat_${MODEL2}_${DATASET}_${COUNT}_*/

15. Solver-Checker Chat (Optimized) (2.5 only)
    目录: results/${ROUND_NAME}_chat_opt_${MODEL1}_${DATASET}_${COUNT}_*/

16. Solver-Checker Chat (Optimized) (3 only)
    目录: results/${ROUND_NAME}_chat_opt_${MODEL2}_${DATASET}_${COUNT}_*/

17. Plan-and-Reflection - $MODEL1
    目录: results/${ROUND_NAME}_planref_${MODEL1}_${DATASET}_${COUNT}_*/

18. Plan-and-Reflection - $MODEL2
    目录: results/${ROUND_NAME}_planref_${MODEL2}_${DATASET}_${COUNT}_*/

================================================================================
下一步分析
================================================================================

使用以下Python脚本分析所有结果:

python3 << 'PYEOF'
import json
from pathlib import Path
import glob

# 查找所有结果目录
result_dirs = glob.glob("results/${ROUND_NAME}_*_${DATASET}_${COUNT}_*/")

results = []
for result_dir in result_dirs:
    answer_file = Path(result_dir) / "answer.json"
    if answer_file.exists():
        with open(answer_file) as f:
            data = json.load(f)
            results.append({
                'name': Path(result_dir).name,
                'accuracy': data.get('accuracy', 0),
                'correct': data.get('correct', 0),
                'total': data.get('total', 0)
            })

# 排序并显示
results.sort(key=lambda x: x['accuracy'], reverse=True)
print(f"\n{'排名':<5} {'准确率':<10} {'正确数':<10} {'测试名称'}")
print("-" * 80)
for i, r in enumerate(results, 1):
    print(f"{i:<5} {r['accuracy']*100:>6.2f}%   {r['correct']:>3}/{r['total']:<3}     {r['name']}")
PYEOF

================================================================================
EOF

log_success "结果汇总已保存到: $SUMMARY_FILE"

log_section "🎉 所有评估完成！"

echo "查看结果汇总:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "查看具体结果:"
echo "  ls -d results/${ROUND_NAME}_*/"
echo ""

exit 0

