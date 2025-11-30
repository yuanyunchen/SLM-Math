#!/bin/bash
# List all agent tmux sessions with quick overview

echo "=========================================="
echo "Agent Tmux Sessions Overview"
echo "=========================================="
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

# GPU mapping
declare -A gpu_map=(
    ["agent_with_python_tools"]=0
    ["majority_vote"]=1
    ["plan_and_reflection"]=2
    ["solver_checker_chat"]=3
    ["solver_checker_stateless"]=4
    ["solver_checker_summarizer"]=5
    ["solver_checker_summarizer_chat"]=6
    ["solver_checker_with_tools"]=7
    ["solver_checker"]="0*"
)

echo "Active Sessions:"
echo ""
active_count=0

for agent in "${agents[@]}"; do
    tmux has-session -t "$agent" 2>/dev/null
    if [ $? == 0 ]; then
        gpu=${gpu_map[$agent]}
        status="✓ RUNNING"
        ((active_count++))
    else
        gpu=${gpu_map[$agent]}
        status="✗ NOT RUNNING"
    fi
    
    # Handle special GPU labels like "0*"
    printf "  GPU %s: %-35s %s\n" "$gpu" "$agent" "$status"
done

echo ""
echo "Summary: $active_count / ${#agents[@]} agents running"
echo ""
echo "Note: GPU 0* means sharing GPU 0 with agent_with_python_tools"
echo ""
echo "Commands:"
echo "  Attach to session:  tmux attach -t <agent_name>"
echo "  List all sessions:  tmux ls"
echo "  Stop all agents:    ./scripts/stop_all_agents.sh"
echo "=========================================="

