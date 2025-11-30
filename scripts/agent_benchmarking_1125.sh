#!/bin/bash
################################################################################
# Agent Benchmarking Script - 1125
# Run all agent modes on GSM8K (500) and MATH-500 datasets
# Distributed across 4 GPUs for parallel execution
################################################################################

# Configuration
MODEL="Qwen2.5-Math-1.5B"
ROUND_NAME="benchmark_1125"
COUNT_GSM8K=500
COUNT_MATH500=500
DATASET_GSM8K="gsm8k"
DATASET_MATH="math500"

# Multi-round/voting settings (set to 5 as requested)
MAX_ITERATIONS=5
NUM_RUNS=5
MAX_SUBPROBLEMS=5

# Save interval
SAVE_INTERVAL=25

# Total number of GPUs
NUM_GPUS=8

# All 9 agent modes
AGENTS=(
    "solver_checker_stateless"
    "solver_checker_chat"
    "solver_checker_trivial_chat"
    "solver_checker_with_tools"
    "solver_checker_summarizer"
    "solver_checker_summarizer_chat"
    "agent_with_python_tools"
    "majority_vote"
    "plan_and_reflection"
)

# GPU assignment (distribute 9 agents across 8 GPUs)
# GPU 0-5: 1 agent each (solver_checker variants)
# GPU 6: majority_vote + agent_with_python_tools (both faster single-pass style)
# GPU 7: plan_and_reflection (complex multi-phase)
declare -A GPU_AGENTS
GPU_AGENTS[0]="solver_checker_stateless"
GPU_AGENTS[1]="solver_checker_chat"
GPU_AGENTS[2]="solver_checker_trivial_chat"
GPU_AGENTS[3]="solver_checker_with_tools"
GPU_AGENTS[4]="solver_checker_summarizer"
GPU_AGENTS[5]="solver_checker_summarizer_chat"
GPU_AGENTS[6]="majority_vote agent_with_python_tools"
GPU_AGENTS[7]="plan_and_reflection"

# Function to build command for agent
build_agent_cmd() {
    local agent=$1
    local dataset=$2
    local count=$3
    
    local cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '${ROUND_NAME}' --dataset '$dataset' --count $count --detailed 'false' --save_interval $SAVE_INTERVAL"
    
    case "$agent" in
        "majority_vote")
            cmd="$cmd --num_runs $NUM_RUNS"
            ;;
        "plan_and_reflection")
            cmd="$cmd --max_iterations $MAX_ITERATIONS --max_subproblems $MAX_SUBPROBLEMS"
            ;;
        "agent_with_python_tools")
            # Single-shot agent, no iterations needed
            ;;
        *)
            # All solver_checker variants
            cmd="$cmd --max_iterations $MAX_ITERATIONS"
            ;;
    esac
    
    echo "$cmd"
}

# Function to get number of tasks for a GPU
count_tasks() {
    local gpu=$1
    local agents_str="${GPU_AGENTS[$gpu]}"
    local count=0
    for agent in $agents_str; do
        count=$((count + 2))  # Each agent runs on 2 datasets
    done
    echo $count
}

# Print header
echo ""
echo "================================================================================"
echo "                    AGENT BENCHMARKING - November 25, 2025"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Model:           $MODEL"
echo "  Round Name:      $ROUND_NAME"
echo "  GSM8K Count:     $COUNT_GSM8K"
echo "  MATH-500 Count:  $COUNT_MATH500"
echo "  Max Iterations:  $MAX_ITERATIONS (solver_checker variants)"
echo "  Num Runs:        $NUM_RUNS (majority_vote)"
echo "  Max Subproblems: $MAX_SUBPROBLEMS (plan_and_reflection)"
echo ""
echo "Agents (${#AGENTS[@]} total):"
for agent in "${AGENTS[@]}"; do
    echo "  - $agent"
done
echo ""
echo "GPU Distribution:"
for gpu in $(seq 0 $((NUM_GPUS-1))); do
    echo "  GPU $gpu: ${GPU_AGENTS[$gpu]}"
done
echo ""
echo "Total tasks: $((${#AGENTS[@]} * 2)) (${#AGENTS[@]} agents x 2 datasets)"
echo ""
echo "================================================================================"
echo ""

# Check if tmux sessions already exist and kill them
echo "Checking for existing sessions..."
for gpu in $(seq 0 $((NUM_GPUS-1))); do
    session_name="benchmark_gpu${gpu}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "  Killing existing session: $session_name"
        tmux kill-session -t "$session_name"
    fi
done
echo ""

# Launch tmux sessions for each GPU
echo "Launching GPU workers..."
echo ""

for gpu in $(seq 0 $((NUM_GPUS-1))); do
    session_name="benchmark_gpu${gpu}"
    agents_str="${GPU_AGENTS[$gpu]}"
    task_count=$(count_tasks $gpu)
    
    echo "Starting GPU $gpu ($task_count tasks): $agents_str"
    
    # Create new session
    tmux new-session -d -s "$session_name"
    
    # Setup environment
    tmux send-keys -t "$session_name" "cd /root/autodl-tmp/SLM-Math" C-m
    tmux send-keys -t "$session_name" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
    
    # Track progress within each session
    task_num=0
    total_tasks=$task_count
    
    # Run each agent assigned to this GPU
    for agent in $agents_str; do
        # GSM8K evaluation
        task_num=$((task_num + 1))
        cmd=$(build_agent_cmd "$agent" "$DATASET_GSM8K" "$COUNT_GSM8K")
        
        tmux send-keys -t "$session_name" "echo ''" C-m
        tmux send-keys -t "$session_name" "echo '================================================================================'" C-m
        tmux send-keys -t "$session_name" "echo '[GPU $gpu] Task $task_num/$total_tasks: $agent on $DATASET_GSM8K'" C-m
        tmux send-keys -t "$session_name" "echo 'Started at: '\$(date '+%Y-%m-%d %H:%M:%S')" C-m
        tmux send-keys -t "$session_name" "echo '================================================================================'" C-m
        tmux send-keys -t "$session_name" "$cmd" C-m
        tmux send-keys -t "$session_name" "echo '[GPU $gpu] Completed: $agent on $DATASET_GSM8K at '\$(date '+%Y-%m-%d %H:%M:%S')" C-m
        
        # MATH-500 evaluation
        task_num=$((task_num + 1))
        cmd=$(build_agent_cmd "$agent" "$DATASET_MATH" "$COUNT_MATH500")
        
        tmux send-keys -t "$session_name" "echo ''" C-m
        tmux send-keys -t "$session_name" "echo '================================================================================'" C-m
        tmux send-keys -t "$session_name" "echo '[GPU $gpu] Task $task_num/$total_tasks: $agent on $DATASET_MATH'" C-m
        tmux send-keys -t "$session_name" "echo 'Started at: '\$(date '+%Y-%m-%d %H:%M:%S')" C-m
        tmux send-keys -t "$session_name" "echo '================================================================================'" C-m
        tmux send-keys -t "$session_name" "$cmd" C-m
        tmux send-keys -t "$session_name" "echo '[GPU $gpu] Completed: $agent on $DATASET_MATH at '\$(date '+%Y-%m-%d %H:%M:%S')" C-m
    done
    
    # Final completion message for this GPU
    tmux send-keys -t "$session_name" "echo ''" C-m
    tmux send-keys -t "$session_name" "echo '================================================================================'" C-m
    tmux send-keys -t "$session_name" "echo '[GPU $gpu] ALL TASKS COMPLETED at '\$(date '+%Y-%m-%d %H:%M:%S')" C-m
    tmux send-keys -t "$session_name" "echo '================================================================================'" C-m
done

echo ""
echo "================================================================================"
echo "                        ALL GPU WORKERS LAUNCHED"
echo "================================================================================"
echo ""
echo "Monitor progress:"
echo "  View all sessions:     tmux ls"
echo "  Attach to GPU 0:       tmux attach -t benchmark_gpu0"
echo "  Attach to GPU 1:       tmux attach -t benchmark_gpu1"
echo "  Attach to GPU 2:       tmux attach -t benchmark_gpu2"
echo "  Attach to GPU 3:       tmux attach -t benchmark_gpu3"
echo "  Attach to GPU 4:       tmux attach -t benchmark_gpu4"
echo "  Attach to GPU 5:       tmux attach -t benchmark_gpu5"
echo "  Attach to GPU 6:       tmux attach -t benchmark_gpu6"
echo "  Attach to GPU 7:       tmux attach -t benchmark_gpu7"
echo ""
echo "Detach from session:     Ctrl+b then d"
echo ""
echo "Check results directory: ls -la results/${ROUND_NAME}*"
echo ""
echo "================================================================================"
echo "Expected completion time: Several hours depending on model speed"
echo "================================================================================"
echo ""

# Create a progress monitoring script
cat > /root/autodl-tmp/SLM-Math/scripts/check_benchmark_progress.sh << 'EOF'
#!/bin/bash
# Check benchmark progress across all GPUs

echo ""
echo "================================================================================"
echo "                    BENCHMARK PROGRESS CHECK"
echo "                    $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
echo ""

# Check each GPU session
for gpu in 0 1 2 3 4 5 6 7; do
    session_name="benchmark_gpu${gpu}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        # Get last few lines from each session
        echo "GPU $gpu ($session_name): RUNNING"
        # Count completed tasks
        completed=$(tmux capture-pane -t "$session_name" -p | grep -c "Completed:")
        echo "  Completed tasks: $completed"
    else
        echo "GPU $gpu ($session_name): NOT RUNNING"
    fi
done

echo ""
echo "Results directories:"
ls -la /root/autodl-tmp/SLM-Math/results/benchmark_1125* 2>/dev/null | head -20 || echo "  No results yet"
echo ""
echo "================================================================================"
EOF

chmod +x /root/autodl-tmp/SLM-Math/scripts/check_benchmark_progress.sh

echo "Progress check script created: ./scripts/check_benchmark_progress.sh"
echo ""

