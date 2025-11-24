#!/bin/bash

################################################################################
# Parallel Evaluation Runner
# 同时启动三个GPU上的评估任务
# - GPU 0: Qwen2.5-Math-1.5B Base (GSM8K + MATH500)
# - GPU 1: Qwen3-1.7B Base (GSM8K + MATH500)
# - GPU 3: Qwen2.5-Math-1.5B Solver-Checker (GSM8K)
################################################################################

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log files
LOG_GPU0="$LOG_DIR/gpu0_qwen25_math_${TIMESTAMP}.log"
LOG_GPU1="$LOG_DIR/gpu1_qwen3_17b_${TIMESTAMP}.log"
LOG_GPU3="$LOG_DIR/gpu3_solver_checker_${TIMESTAMP}.log"

# PID file to track background processes
PID_FILE="$LOG_DIR/evaluation_pids_${TIMESTAMP}.txt"

################################################################################
# Color output
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

################################################################################
# Pre-flight Checks
################################################################################

log_phase "Parallel Evaluation Runner - Pre-flight Checks"

# Check if scripts exist
SCRIPT_GPU0="$SCRIPT_DIR/eval_gpu0_qwen25_math.sh"
SCRIPT_GPU1="$SCRIPT_DIR/eval_gpu1_qwen3_17b.sh"
SCRIPT_GPU3="$SCRIPT_DIR/eval_gpu2_solver_checker.sh"

if [ ! -f "$SCRIPT_GPU0" ]; then
    log_error "Script not found: $SCRIPT_GPU0"
    exit 1
fi

if [ ! -f "$SCRIPT_GPU1" ]; then
    log_error "Script not found: $SCRIPT_GPU1"
    exit 1
fi

if [ ! -f "$SCRIPT_GPU3" ]; then
    log_error "Script not found: $SCRIPT_GPU3"
    exit 1
fi

log_success "All scripts found"

# Check GPU availability
log_info "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name --format=csv,noheader | while IFS=',' read -r idx name; do
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs)
        log_info "GPU $idx: $name"
    done
else
    log_warning "nvidia-smi not found, cannot verify GPU availability"
fi

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    log_warning "Conda environment not activated. Make sure to activate 'slm_math' environment."
    log_info "Run: conda activate slm_math"
else
    log_success "Conda environment: $CONDA_DEFAULT_ENV"
fi

echo ""
log_info "Log files will be saved to:"
log_info "  GPU 0: $LOG_GPU0"
log_info "  GPU 1: $LOG_GPU1"
log_info "  GPU 3: $LOG_GPU3"
echo ""

################################################################################
# Start Background Processes
################################################################################

log_phase "Starting Parallel Evaluations"

# Function to start a background task
start_task() {
    local gpu_num=$1
    local script_path=$2
    local log_file=$3
    local task_name=$4
    
    log_info "Starting GPU $gpu_num: $task_name"
    log_info "  Script: $script_path"
    log_info "  Log: $log_file"
    
    # Change to project root and run script in background
    (
        cd "$PROJECT_ROOT"
        bash "$script_path" > "$log_file" 2>&1
        echo $? > "$LOG_DIR/gpu${gpu_num}_exit_code.txt"
    ) &
    
    local pid=$!
    echo "$pid" >> "$PID_FILE"
    log_success "GPU $gpu_num started (PID: $pid)"
    
    return $pid
}

# Clear PID file
> "$PID_FILE"

# Start GPU 0 task
start_task 0 "$SCRIPT_GPU0" "$LOG_GPU0" "Qwen2.5-Math-1.5B Base (GSM8K + MATH500)"
PID_GPU0=$!

# Small delay to avoid resource contention
sleep 2

# Start GPU 1 task
start_task 1 "$SCRIPT_GPU1" "$LOG_GPU1" "Qwen3-1.7B Base (GSM8K + MATH500)"
PID_GPU1=$!

# Small delay
sleep 2

# Start GPU 3 task
start_task 3 "$SCRIPT_GPU3" "$LOG_GPU3" "Qwen2.5-Math-1.5B Solver-Checker (GSM8K)"
PID_GPU3=$!

echo ""
log_success "All tasks started!"
log_info "PIDs: GPU0=$PID_GPU0, GPU1=$PID_GPU1, GPU3=$PID_GPU3"
log_info "PID file: $PID_FILE"
echo ""

################################################################################
# Monitor Progress
################################################################################

log_phase "Monitoring Progress"

# Function to check if process is still running
is_running() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

# Function to get exit code
get_exit_code() {
    local gpu_num=$1
    local exit_file="$LOG_DIR/gpu${gpu_num}_exit_code.txt"
    if [ -f "$exit_file" ]; then
        cat "$exit_file"
    else
        echo ""
    fi
}

# Function to show tail of log
show_log_tail() {
    local log_file=$1
    local lines=${2:-5}
    if [ -f "$log_file" ]; then
        tail -n "$lines" "$log_file" 2>/dev/null | sed 's/^/    /'
    fi
}

# Monitor loop
ALL_COMPLETE=false
CHECK_INTERVAL=30  # Check every 30 seconds
MAX_CHECKS=0  # 0 = unlimited

check_count=0
while [ "$ALL_COMPLETE" = false ]; do
    check_count=$((check_count + 1))
    
    # Check status of each task
    RUNNING_GPU0=false
    RUNNING_GPU1=false
    RUNNING_GPU3=false
    
    if is_running "$PID_GPU0"; then
        RUNNING_GPU0=true
    fi
    
    if is_running "$PID_GPU1"; then
        RUNNING_GPU1=true
    fi
    
    if is_running "$PID_GPU3"; then
        RUNNING_GPU3=true
    fi
    
    # Print status
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Status Check #$check_count ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ "$RUNNING_GPU0" = true ]; then
        echo -e "GPU 0: ${YELLOW}RUNNING${NC} (PID: $PID_GPU0)"
        echo "  Last log lines:"
        show_log_tail "$LOG_GPU0" 3
    else
        EXIT_CODE=$(get_exit_code 0)
        if [ -n "$EXIT_CODE" ]; then
            if [ "$EXIT_CODE" = "0" ]; then
                echo -e "GPU 0: ${GREEN}COMPLETED${NC} (Exit code: $EXIT_CODE)"
            else
                echo -e "GPU 0: ${RED}FAILED${NC} (Exit code: $EXIT_CODE)"
            fi
        else
            echo -e "GPU 0: ${CYAN}UNKNOWN${NC}"
        fi
    fi
    
    echo ""
    
    if [ "$RUNNING_GPU1" = true ]; then
        echo -e "GPU 1: ${YELLOW}RUNNING${NC} (PID: $PID_GPU1)"
        echo "  Last log lines:"
        show_log_tail "$LOG_GPU1" 3
    else
        EXIT_CODE=$(get_exit_code 1)
        if [ -n "$EXIT_CODE" ]; then
            if [ "$EXIT_CODE" = "0" ]; then
                echo -e "GPU 1: ${GREEN}COMPLETED${NC} (Exit code: $EXIT_CODE)"
            else
                echo -e "GPU 1: ${RED}FAILED${NC} (Exit code: $EXIT_CODE)"
            fi
        else
            echo -e "GPU 1: ${CYAN}UNKNOWN${NC}"
        fi
    fi
    
    echo ""
    
    if [ "$RUNNING_GPU3" = true ]; then
        echo -e "GPU 3: ${YELLOW}RUNNING${NC} (PID: $PID_GPU3)"
        echo "  Last log lines:"
        show_log_tail "$LOG_GPU3" 3
    else
        EXIT_CODE=$(get_exit_code 3)
        if [ -n "$EXIT_CODE" ]; then
            if [ "$EXIT_CODE" = "0" ]; then
                echo -e "GPU 3: ${GREEN}COMPLETED${NC} (Exit code: $EXIT_CODE)"
            else
                echo -e "GPU 3: ${RED}FAILED${NC} (Exit code: $EXIT_CODE)"
            fi
        else
            echo -e "GPU 3: ${CYAN}UNKNOWN${NC}"
        fi
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if all tasks are complete
    if [ "$RUNNING_GPU0" = false ] && [ "$RUNNING_GPU1" = false ] && [ "$RUNNING_GPU3" = false ]; then
        ALL_COMPLETE=true
        log_success "All tasks completed!"
        break
    fi
    
    # Check if we should stop monitoring
    if [ $MAX_CHECKS -gt 0 ] && [ $check_count -ge $MAX_CHECKS ]; then
        log_warning "Reached maximum check count. Stopping monitoring."
        log_info "Tasks are still running. Check logs manually:"
        log_info "  tail -f $LOG_GPU0"
        log_info "  tail -f $LOG_GPU1"
        log_info "  tail -f $LOG_GPU3"
        break
    fi
    
    # Wait before next check
    if [ "$ALL_COMPLETE" = false ]; then
        echo ""
        log_info "Next check in ${CHECK_INTERVAL} seconds... (Press Ctrl+C to stop monitoring but keep tasks running)"
        sleep "$CHECK_INTERVAL"
    fi
done

################################################################################
# Final Summary
################################################################################

log_phase "Final Summary"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task Completion Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check final exit codes
for gpu in 0 1 3; do
    EXIT_CODE=$(get_exit_code "$gpu")
    if [ -n "$EXIT_CODE" ]; then
        if [ "$EXIT_CODE" = "0" ]; then
            echo -e "GPU $gpu: ${GREEN}✓ COMPLETED${NC} (Exit code: 0)"
        else
            echo -e "GPU $gpu: ${RED}✗ FAILED${NC} (Exit code: $EXIT_CODE)"
        fi
    else
        echo -e "GPU $gpu: ${CYAN}? UNKNOWN${NC} (Check log file)"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

log_info "Log files:"
log_info "  GPU 0: $LOG_GPU0"
log_info "  GPU 1: $LOG_GPU1"
log_info "  GPU 3: $LOG_GPU3"
echo ""

log_info "To view logs in real-time:"
log_info "  tail -f $LOG_GPU0"
log_info "  tail -f $LOG_GPU1"
log_info "  tail -f $LOG_GPU3"
echo ""

log_info "To view all logs simultaneously:"
log_info "  tail -f $LOG_GPU0 $LOG_GPU1 $LOG_GPU3"
echo ""

log_success "Parallel evaluation runner completed!"
echo ""

################################################################################
# Cleanup
################################################################################

# Optionally remove PID file (comment out if you want to keep it)
# rm -f "$PID_FILE"

exit 0

