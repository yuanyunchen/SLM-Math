#!/bin/bash
# Quick status check for all running evaluations

RESULTS_DIR="results"

echo "=========================================="
echo "Agent Evaluation Status"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Check for running processes
running=$(ps aux | grep "eval_agent" | grep -v grep | wc -l)
echo "Running processes: $running"
echo ""

# Check GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{printf "  GPU %s: %s%% utilization, %sMB / %sMB memory\n", $1, $3, $4, $5}'
echo ""

# Check latest results for each agent
echo "Latest Results:"
echo "----------------------------------------"
for agent_dir in $(find $RESULTS_DIR -maxdepth 1 -type d -name "full_eval_*" -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2- | head -16); do
    if [ -f "$agent_dir/summary.txt" ]; then
        dir_name=$(basename "$agent_dir")
        echo ""
        echo "[$dir_name]"
        
        # Extract key metrics
        if grep -q "Accuracy" "$agent_dir/summary.txt"; then
            accuracy=$(grep "Accuracy" "$agent_dir/summary.txt" | head -1)
            echo "  $accuracy"
        fi
        
        if grep -q "First Try" "$agent_dir/summary.txt"; then
            first_try=$(grep "First Try" "$agent_dir/summary.txt" | head -1)
            echo "  $first_try"
        fi
        
        if grep -q "Total Cases" "$agent_dir/summary.txt"; then
            total=$(grep "Total Cases" "$agent_dir/summary.txt" | head -1)
            echo "  $total"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Tmux Sessions:"
echo "=========================================="
tmux ls 2>/dev/null | grep -E "(agent_with_python_tools|majority_vote|plan_and_reflection|solver_checker)" || echo "  No agent sessions found"
echo ""
echo "To attach to specific agent:"
echo "  tmux attach -t <agent_name>"
echo "Example:"
echo "  tmux attach -t agent_with_python_tools"
echo "=========================================="

