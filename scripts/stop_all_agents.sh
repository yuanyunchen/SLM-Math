#!/bin/bash
# Stop all agent tmux sessions

echo "Stopping all agent evaluations..."
echo ""

# List of all agent sessions
agents=(
    "agent_with_python_tools"
    "majority_vote"
    "plan_and_reflection"
    "solver_checker_chat"
    "solver_checker_stateless"
    "solver_checker_summarizer"
    "solver_checker_summarizer_chat"
    "solver_checker_with_tools"
    "solver_checker"
)

# Kill each session
for agent in "${agents[@]}"; do
    tmux has-session -t "$agent" 2>/dev/null
    if [ $? == 0 ]; then
        echo "  Killing session: $agent"
        tmux kill-session -t "$agent"
    else
        echo "  Session not found: $agent"
    fi
done

echo ""
echo "All agent sessions stopped."
echo ""

# Also kill any remaining eval_agent processes
remaining=$(ps aux | grep "eval_agent" | grep -v grep | wc -l)
if [ $remaining -gt 0 ]; then
    echo "Killing $remaining remaining eval_agent processes..."
    pkill -f eval_agent
    sleep 2
    echo "Done."
else
    echo "No remaining eval_agent processes."
fi

echo ""
echo "=========================================="
echo "All agents stopped successfully"
echo "=========================================="

