#!/bin/bash

################################################################################
# 多阶段评估脚本 - 支持任意多轮配置
# 
# 用法: bash run_full_evaluation_multi_phase.sh "25,25,25,25"
# 
# 每轮结束后自动更新:
#   - metrics.csv
#   - analysis_report.txt
#   - answers/*.json
# 
# 可以随时查看中间结果！
################################################################################

set -e  # Exit on error

################################################################################
# 配置参数
################################################################################

# 解析样本配置（逗号或空格分隔）
# 示例: "25,25,25" 或 "25 25 25" = 3轮，每轮25个样本，总共75个
SAMPLES_CONFIG=${1:-"25,25"}

# 将逗号或空格分隔的字符串转换为数组
# 先替换逗号为空格，然后按空格分割
SAMPLES_CONFIG_NORMALIZED=$(echo "$SAMPLES_CONFIG" | tr ',' ' ')
read -ra SAMPLE_PHASES <<< "$SAMPLES_CONFIG_NORMALIZED"

# 计算总轮数和总样本数
NUM_PHASES=${#SAMPLE_PHASES[@]}
TOTAL_SAMPLES=0
for samples in "${SAMPLE_PHASES[@]}"; do
    TOTAL_SAMPLES=$((TOTAL_SAMPLES + samples))
done

echo "样本配置: $SAMPLES_CONFIG"
echo "总轮数: $NUM_PHASES"
echo "总样本数: $TOTAL_SAMPLES"
echo ""

# Dataset
DATASET="gsm8k"

# Round名称 (添加时间戳)
TIMESTAMP=$(date +"%m%d_%H%M")
ROUND_NAME="full_eval_${TIMESTAMP}"

# 详细输出
DETAILED="false"

# 迭代/运行次数限制
MAX_ITERATIONS=5
NUM_RUNS=5

# Model配置
MODEL1="Qwen2.5-Math-1.5B"
MODEL2="Qwen3-1.7B"

################################################################################
# 颜色输出
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

log_phase() {
    echo ""
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC} $1"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
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
echo "Total Phases:     $NUM_PHASES"
echo "Samples Config:   $SAMPLES_CONFIG"
echo "Total Samples:    $TOTAL_SAMPLES"
echo "Max Iterations:   $MAX_ITERATIONS"
echo "Num Runs (MV):    $NUM_RUNS"
echo "Detailed Output:  $DETAILED"
echo ""

# 显示每轮详情
echo "各轮详情:"
cumulative=0
for i in "${!SAMPLE_PHASES[@]}"; do
    phase_num=$((i + 1))
    samples=${SAMPLE_PHASES[$i]}
    start=$cumulative
    end=$((cumulative + samples))
    cumulative=$end
    echo "  Phase $phase_num: Samples $((start+1))-$end (共${samples}个)"
done
echo ""

# 估算时间
ESTIMATED_TOTAL=$(python -c "
workflows = [
    (2, 10),   # Base Direct
    (2, 55),   # Majority Vote
    (2, 35),   # Stateless
    (2, 45),   # Summarizer
    (2, 50),   # Summarizer Chat
    (2, 40),   # With Tools
    (2, 40),   # Trivial Chat
    (2, 55),   # Chat Opt
    (2, 65),   # Plan-and-Reflection
]
total = sum(configs * $TOTAL_SAMPLES * time for configs, time in workflows)
print(f'{total/3600:.1f}')
")

echo "预计总时间: 约 ${ESTIMATED_TOTAL} 小时"
echo ""

read -p "确认开始多阶段评估? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_warning "评估已取消"
    exit 0
fi

################################################################################
# 辅助函数
################################################################################

CURRENT_TEST=0
TOTAL_TESTS=18

# 存储所有结果目录 (按test索引)
declare -A FINAL_RESULT_DIRS

run_test_multi_phase() {
    local test_name="$1"
    local model="$2"
    local agent="$3"
    local round_suffix="$4"
    shift 4
    local extra_args="$@"
    
    CURRENT_TEST=$((CURRENT_TEST + 1))
    local round_full="${ROUND_NAME}_${round_suffix}"
    
    # 多轮运行
    local cumulative_start=0
    local resume_dir=""
    
    for phase_idx in "${!SAMPLE_PHASES[@]}"; do
        local phase_num=$((phase_idx + 1))
        local phase_count=${SAMPLE_PHASES[$phase_idx]}
        local phase_end=$((cumulative_start + phase_count))
        
        # 构建进度显示
        local progress_overall="[Test ${CURRENT_TEST}/${TOTAL_TESTS}]"
        local progress_phase="[Phase ${phase_num}/${NUM_PHASES}]"
        local progress_samples="[Samples $((cumulative_start+1))-${phase_end}/${TOTAL_SAMPLES}]"
        
        log_phase "${progress_overall} ${progress_phase} ${progress_samples} ${test_name}"
        
        # 构建命令
        local cmd="python -m evaluation.eval_agent \
            --model \"$model\" \
            --agent \"$agent\" \
            --round \"$round_full\" \
            --dataset \"$DATASET\" \
            --count \"$phase_count\" \
            --start \"$cumulative_start\" \
            --detailed \"$DETAILED\" \
            $extra_args"
        
        # 从Phase 2开始需要resume
        if [ $phase_num -gt 1 ] && [ -n "$resume_dir" ]; then
            cmd="$cmd --resume \"$resume_dir\""
            log_info "Resuming from: $resume_dir"
        fi
        
        # 执行命令
        if eval $cmd; then
            log_success "完成: ${test_name} - Phase ${phase_num}"
            
            # 查找结果目录
            if [ -z "$resume_dir" ]; then
                resume_dir=$(find results -maxdepth 1 -type d -name "${round_full}_*_${DATASET}_*" | head -1)
            fi
            
            # 每轮结束后显示当前结果
            if [ -f "$resume_dir/metrics.csv" ]; then
                log_info "当前结果 (Phase ${phase_num}):"
                python << PYEOF
import pandas as pd
df = pd.read_csv("$resume_dir/metrics.csv")
print(f"  样本数: {df['total_samples'].iloc[0]}")
print(f"  准确率: {df['accuracy'].iloc[0]*100:.2f}%")
print(f"  正确数: {df['correct'].iloc[0]}/{df['total_samples'].iloc[0]}")
PYEOF
            fi
        else
            log_error "失败: ${test_name} - Phase ${phase_num}"
            return 1
        fi
        
        cumulative_start=$phase_end
    done
    
    # 所有phase完成后，记录最终结果目录
    FINAL_RESULT_DIRS[$CURRENT_TEST]="$resume_dir"
}

run_test_base_direct_multi_phase() {
    local test_name="$1"
    local model="$2"
    local round_suffix="$3"
    
    CURRENT_TEST=$((CURRENT_TEST + 1))
    local round_full="${ROUND_NAME}_${round_suffix}"
    
    # 多轮运行
    local cumulative_start=0
    local resume_dir=""
    
    for phase_idx in "${!SAMPLE_PHASES[@]}"; do
        local phase_num=$((phase_idx + 1))
        local phase_count=${SAMPLE_PHASES[$phase_idx]}
        local phase_end=$((cumulative_start + phase_count))
        
        # 构建进度显示
        local progress_overall="[Test ${CURRENT_TEST}/${TOTAL_TESTS}]"
        local progress_phase="[Phase ${phase_num}/${NUM_PHASES}]"
        local progress_samples="[Samples $((cumulative_start+1))-${phase_end}/${TOTAL_SAMPLES}]"
        
        log_phase "${progress_overall} ${progress_phase} ${progress_samples} ${test_name}"
        
        # Base Direct 使用 eval.py
        local cmd="python -m evaluation.eval \
            --model \"$model\" \
            --round \"$round_full\" \
            --dataset \"$DATASET\" \
            --count \"$phase_count\" \
            --start \"$cumulative_start\" \
            --mode \"standard\" \
            --detailed \"$DETAILED\""
        
        # 从Phase 2开始需要resume
        if [ $phase_num -gt 1 ] && [ -n "$resume_dir" ]; then
            cmd="$cmd --resume \"$resume_dir\""
            log_info "Resuming from: $resume_dir"
        fi
        
        # 执行命令
        if eval $cmd; then
            log_success "完成: ${test_name} - Phase ${phase_num}"
            
            # 查找结果目录
            if [ -z "$resume_dir" ]; then
                resume_dir=$(find results -maxdepth 1 -type d -name "${round_full}_*_${DATASET}_*" | head -1)
            fi
            
            # 每轮结束后显示当前结果
            if [ -f "$resume_dir/metrics.csv" ]; then
                log_info "当前结果 (Phase ${phase_num}):"
                python << PYEOF
import pandas as pd
df = pd.read_csv("$resume_dir/metrics.csv")
print(f"  样本数: {df['total_samples'].iloc[0]}")
print(f"  准确率: {df['accuracy'].iloc[0]*100:.2f}%")
print(f"  正确数: {df['correct'].iloc[0]}/{df['total_samples'].iloc[0]}")
PYEOF
            fi
        else
            log_error "失败: ${test_name} - Phase ${phase_num}"
            return 1
        fi
        
        cumulative_start=$phase_end
    done
    
    # 所有phase完成后，记录最终结果目录
    FINAL_RESULT_DIRS[$CURRENT_TEST]="$resume_dir"
}

################################################################################
# 测试定义
################################################################################

declare -a TESTS=(
    "Majority Vote - $MODEL1|$MODEL1|majority_vote|mv_${MODEL1}|--num_runs $NUM_RUNS --temperature 0.7 --top_p 0.95"
    "Majority Vote - $MODEL2|$MODEL2|majority_vote|mv_${MODEL2}|--num_runs $NUM_RUNS --temperature 0.7 --top_p 0.95"
    "S-C Stateless (2.5+3)|$MODEL1|solver_checker|stateless|--checker_model $MODEL2 --max_iterations $MAX_ITERATIONS"
    "S-C Stateless (3 only)|$MODEL2|solver_checker|stateless_${MODEL2}|--max_iterations $MAX_ITERATIONS"
    "S-C Summarizer (2.5+3)|$MODEL1|solver_checker_summarizer|summarizer|--checker_model $MODEL2 --max_iterations $MAX_ITERATIONS"
    "S-C Summarizer (3 only)|$MODEL2|solver_checker_summarizer|summarizer_${MODEL2}|--max_iterations $MAX_ITERATIONS"
    "S-C Sum Chat (2.5)|$MODEL1|solver_checker_summarizer_chat|summarizer_chat|--max_iterations $MAX_ITERATIONS"
    "S-C Sum Chat (3)|$MODEL2|solver_checker_summarizer_chat|summarizer_chat_${MODEL2}|--max_iterations $MAX_ITERATIONS"
    "S-C With Tools (2.5+3)|$MODEL1|solver_checker_with_tools|with_tools|--checker_model $MODEL2 --max_iterations $MAX_ITERATIONS --enable_solver_tools true --enable_checker_tools true"
    "S-C With Tools (3 only)|$MODEL2|solver_checker_with_tools|with_tools_${MODEL2}|--max_iterations $MAX_ITERATIONS --enable_solver_tools true --enable_checker_tools true"
    "S-C Trivial Chat (2.5)|$MODEL1|solver_checker_trivial_chat|trivial_chat|--max_iterations $MAX_ITERATIONS"
    "S-C Trivial Chat (3)|$MODEL2|solver_checker_trivial_chat|trivial_chat_${MODEL2}|--max_iterations $MAX_ITERATIONS"
    "S-C Chat Opt (2.5)|$MODEL1|solver_checker_chat|chat_opt|--max_iterations $MAX_ITERATIONS"
    "S-C Chat Opt (3)|$MODEL2|solver_checker_chat|chat_opt_${MODEL2}|--max_iterations $MAX_ITERATIONS"
    "Plan-and-Reflection - $MODEL1|$MODEL1|plan_and_reflection|planref_${MODEL1}|--max_iterations $MAX_ITERATIONS --max_subproblems 5"
    "Plan-and-Reflection - $MODEL2|$MODEL2|plan_and_reflection|planref_${MODEL2}|--max_iterations $MAX_ITERATIONS --max_subproblems 5"
)

################################################################################
# 开始多阶段评估
################################################################################

OVERALL_START_TIME=$(date +%s)

log_section "开始多阶段评估 (${NUM_PHASES}轮)"

echo "各轮配置:"
for i in "${!SAMPLE_PHASES[@]}"; do
    echo "  Phase $((i+1)): ${SAMPLE_PHASES[$i]} samples"
done
echo ""

# 记录每个phase的时间
declare -a PHASE_TIMES

################################################################################
# 运行所有测试 (多轮)
################################################################################

CURRENT_TEST=0

# Test 1-2: Base Direct
run_test_base_direct_multi_phase "Base Direct - $MODEL1" "$MODEL1" "base_${MODEL1}"
run_test_base_direct_multi_phase "Base Direct - $MODEL2" "$MODEL2" "base_${MODEL2}"

# Tests 3-18: Agent workflows
for i in "${!TESTS[@]}"; do
    IFS='|' read -r test_name model agent round_suffix extra_args <<< "${TESTS[$i]}"
    run_test_multi_phase "$test_name" "$model" "$agent" "$round_suffix" $extra_args
done

################################################################################
# 评估完成
################################################################################

OVERALL_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((OVERALL_END_TIME - OVERALL_START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))

log_section "评估完成！"

echo "总耗时: ${TOTAL_HOURS}小时 ${TOTAL_MINUTES}分钟"
echo "完成测试: $TOTAL_TESTS 个"
echo "每个测试轮数: $NUM_PHASES 轮"
echo "每个测试总样本: $TOTAL_SAMPLES"
echo "总样本数: $((TOTAL_SAMPLES * TOTAL_TESTS))"
echo ""
echo "结果目录: results/"
echo "Round前缀: ${ROUND_NAME}"
echo ""

# 生成结果汇总
log_info "生成结果汇总..."

SUMMARY_FILE="results/${ROUND_NAME}_SUMMARY_MULTI_PHASE.txt"

cat > "$SUMMARY_FILE" << EOFSUM
================================================================================
多阶段评估结果汇总
================================================================================

评估时间: $(date)
Round: $ROUND_NAME
Dataset: $DATASET

阶段配置:
  总轮数: $NUM_PHASES
  样本配置: $SAMPLES_CONFIG
  总样本/测试: $TOTAL_SAMPLES

总耗时: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m

================================================================================
测试列表 (18个)
================================================================================

EOFSUM

# 列出所有结果目录
TEST_NUM=1
for test_idx in {1..18}; do
    if [ -n "${FINAL_RESULT_DIRS[$test_idx]}" ]; then
        echo "${TEST_NUM}.  $(basename ${FINAL_RESULT_DIRS[$test_idx]})" >> "$SUMMARY_FILE"
        echo "     目录: ${FINAL_RESULT_DIRS[$test_idx]}" >> "$SUMMARY_FILE"
        
        # 添加最终指标
        if [ -f "${FINAL_RESULT_DIRS[$test_idx]}/metrics.csv" ]; then
            metrics_info=$(python << PYEOF
import pandas as pd
df = pd.read_csv("${FINAL_RESULT_DIRS[$test_idx]}/metrics.csv")
print(f"     准确率: {df['accuracy'].iloc[0]*100:.2f}% ({df['correct'].iloc[0]}/{df['total_samples'].iloc[0]})")
PYEOF
)
            echo "$metrics_info" >> "$SUMMARY_FILE"
        fi
        
        echo "" >> "$SUMMARY_FILE"
        TEST_NUM=$((TEST_NUM + 1))
    fi
done

cat >> "$SUMMARY_FILE" << 'EOFSUM'

================================================================================
合并分析
================================================================================

使用以下Python脚本合并和分析所有结果:

python << 'PYEOF'
import pandas as pd
import glob

# 查找所有metrics.csv
all_metrics = []
for csv_file in glob.glob("results/full_eval_*/metrics.csv"):
    df = pd.read_csv(csv_file)
    all_metrics.append(df)

if not all_metrics:
    print("未找到metrics.csv文件")
else:
    combined = pd.concat(all_metrics, ignore_index=True)
    combined = combined.sort_values('accuracy', ascending=False)
    
    print("\n准确率排名 (Top 10):")
    print("="*100)
    print(combined[['agent', 'model', 'accuracy', 'total_samples', 'first_try_accuracy']].head(10).to_string(index=False))
    
    print("\n\n改进效果排名:")
    print("="*100)
    combined['improvement'] = combined['accuracy'] - combined['first_try_accuracy']
    print(combined[['agent', 'model', 'improvement', 'improved_cases']].sort_values('improvement', ascending=False).head(10).to_string(index=False))
    
    # 保存合并结果
    combined.to_csv("results/all_metrics_combined.csv", index=False)
    print("\n✅ 合并结果已保存到: results/all_metrics_combined.csv")
PYEOF

================================================================================
EOFSUM

log_success "结果汇总已保存到: $SUMMARY_FILE"

log_section "🎉 多阶段评估全部完成！"

echo "查看结果汇总:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "合并所有metrics.csv:"
echo "  # 运行汇总文件中的Python代码"
echo ""
echo "查看具体结果:"
echo "  ls -d results/${ROUND_NAME}_*/"
echo ""

exit 0

