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
