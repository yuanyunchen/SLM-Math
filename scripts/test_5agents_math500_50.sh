#!/bin/bash
################################################################################
# Test 5 Solver-Checker Agents on MATH500 (50 samples)
# GPUs: 4-7 (4 GPUs for 5 agents, 2 on GPU 4)
# Model: Qwen2.5-Math-1.5B
################################################################################

MODEL="Qwen2.5-Math-1.5B"
ROUND="test_5agents_math500"
DATASET="math500"
COUNT=50
MAX_ITERATIONS=3

# Define 5 agents and their GPU assignments
# GPU 4: solver_checker_stateless + solver_checker_with_tools (both ~3-4GB, fits on 24GB)
# GPU 5: solver_checker_summarizer_chat
# GPU 6: solver_checker_summarizer
# GPU 7: solver_checker_trivial_chat

declare -A AGENT_GPU=(
    ["solver_checker_stateless"]=4
    ["solver_checker_with_tools"]=4
    ["solver_checker_summarizer_chat"]=5
    ["solver_checker_summarizer"]=6
    ["solver_checker_trivial_chat"]=7
)

echo "=========================================="
echo "Testing 5 Solver-Checker Agents on MATH500"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Count: $COUNT samples"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Round: $ROUND"
echo "GPUs: 4-7 (2 agents on GPU 4)"
echo "=========================================="
echo ""

# Launch all 5 agents
for agent in "${!AGENT_GPU[@]}"; do
    gpu=${AGENT_GPU[$agent]}
    session_name="${agent}_test"
    
    echo "Launching $agent on GPU $gpu (session: $session_name)"
    
    # Kill existing session if exists
    tmux has-session -t $session_name 2>/dev/null
    if [ $? == 0 ]; then
        echo "  Killing existing session $session_name..."
        tmux kill-session -t $session_name
    fi
    
    # Create new session
    tmux new-session -d -s $session_name
    
    # Send commands
    tmux send-keys -t $session_name "cd /root/autodl-tmp/SLM-Math" C-m
    tmux send-keys -t $session_name "export CUDA_VISIBLE_DEVICES=$gpu" C-m
    
    # Build command
    cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '$ROUND' --dataset '$DATASET' --count $COUNT --max_iterations $MAX_ITERATIONS --detailed 'true' --save_interval 10"
    
    tmux send-keys -t $session_name "echo 'Starting $agent on GPU $gpu'" C-m
    tmux send-keys -t $session_name "$cmd" C-m
    tmux send-keys -t $session_name "echo 'COMPLETED: $agent on GPU $gpu'" C-m
    
    # Small delay between launches to avoid race conditions
    sleep 2
done

echo ""
echo "=========================================="
echo "All 5 agents launched in parallel!"
echo "=========================================="
echo ""
echo "GPU assignments:"
echo "  GPU 4: solver_checker_stateless, solver_checker_with_tools"
echo "  GPU 5: solver_checker_summarizer_chat"
echo "  GPU 6: solver_checker_summarizer"
echo "  GPU 7: solver_checker_trivial_chat"
echo ""
echo "Active sessions:"
tmux ls 2>/dev/null | grep "_test" || echo "  No test sessions found"
echo ""
echo "Monitor sessions:"
echo "  tmux attach -t solver_checker_stateless_test"
echo "  tmux attach -t solver_checker_summarizer_chat_test"
echo "  tmux attach -t solver_checker_summarizer_test"
echo "  tmux attach -t solver_checker_trivial_chat_test"
echo "  tmux attach -t solver_checker_with_tools_test"
echo ""
echo "List all sessions: tmux ls"
echo "Detach: Ctrl+b + d"
echo ""

