#!/bin/bash
# Test 4 agents on 4 GPUs in parallel - MATH500 with 50 samples
# Agents: agent_with_python_tools, solver_checker_chat, majority_vote, plan_and_reflection

MODEL="Qwen2.5-Math-1.5B"
ROUND="test_4agents"
DATASET="math500"
COUNT=50
MAX_ITERATIONS=3
NUM_RUNS=3  # For majority_vote
MAX_SUBPROBLEMS=3  # For plan_and_reflection

# Agent-to-GPU mapping (4 agents, 4 GPUs)
declare -A AGENT_GPU_MAP=(
    ["agent_with_python_tools"]=0
    ["solver_checker_chat"]=1
    ["majority_vote"]=2
    ["plan_and_reflection"]=3
)

echo "=========================================="
echo "Testing 4 Agents on 4 GPUs - MATH500"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Count: $COUNT samples"
echo "Round: $ROUND"
echo "=========================================="
echo ""

# Launch each agent in its own tmux session
for agent in "${!AGENT_GPU_MAP[@]}"; do
    gpu=${AGENT_GPU_MAP[$agent]}
    session_name="${agent}_test"
    
    echo "Launching $agent on GPU $gpu (session: $session_name)"
    
    # Kill existing session if it exists
    tmux has-session -t $session_name 2>/dev/null
    if [ $? == 0 ]; then
        echo "  Warning: Session $session_name already exists, killing it..."
        tmux kill-session -t $session_name
    fi
    
    # Create new session for this agent
    tmux new-session -d -s $session_name
    
    # Send commands to session
    tmux send-keys -t $session_name "cd /root/autodl-tmp/SLM-Math" C-m
    tmux send-keys -t $session_name "export CUDA_VISIBLE_DEVICES=$gpu" C-m
    
    # Build command based on agent type
    if [ "$agent" = "majority_vote" ]; then
        # majority_vote uses --num_runs instead of --max_iterations
        cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '$ROUND' --dataset '$DATASET' --count $COUNT --num_runs $NUM_RUNS --detailed 'false' --save_interval 10"
    elif [ "$agent" = "plan_and_reflection" ]; then
        # plan_and_reflection uses both --max_iterations and --max_subproblems
        cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '$ROUND' --dataset '$DATASET' --count $COUNT --max_iterations $MAX_ITERATIONS --max_subproblems $MAX_SUBPROBLEMS --detailed 'false' --save_interval 10"
    elif [ "$agent" = "agent_with_python_tools" ]; then
        # agent_with_python_tools is single-shot, no max_iterations needed
        cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '$ROUND' --dataset '$DATASET' --count $COUNT --detailed 'false' --save_interval 10"
    else
        # solver_checker_chat uses standard --max_iterations
        cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '$ROUND' --dataset '$DATASET' --count $COUNT --max_iterations $MAX_ITERATIONS --detailed 'false' --save_interval 10"
    fi
    
    # Run the evaluation
    tmux send-keys -t $session_name "echo '========================================'" C-m
    tmux send-keys -t $session_name "echo 'Starting $agent on GPU $gpu'" C-m
    tmux send-keys -t $session_name "echo '========================================'" C-m
    tmux send-keys -t $session_name "$cmd" C-m
    
    # Notify completion
    tmux send-keys -t $session_name "echo '========================================'" C-m
    tmux send-keys -t $session_name "echo 'COMPLETED: $agent on GPU $gpu'" C-m
    tmux send-keys -t $session_name "echo '========================================'" C-m
done

echo ""
echo "=========================================="
echo "All 4 agents launched in separate tmux sessions"
echo "=========================================="
echo ""
echo "Agent-GPU-Session mapping:"
for agent in "${!AGENT_GPU_MAP[@]}"; do
    gpu=${AGENT_GPU_MAP[$agent]}
    echo "  GPU $gpu: $agent (tmux session: ${agent}_test)"
done | sort
echo ""
echo "To monitor specific agent:"
echo "  tmux attach -t <agent_name>_test"
echo ""
echo "Examples:"
echo "  tmux attach -t agent_with_python_tools_test"
echo "  tmux attach -t solver_checker_chat_test"
echo "  tmux attach -t majority_vote_test"
echo "  tmux attach -t plan_and_reflection_test"
echo ""
echo "List all sessions:"
echo "  tmux ls"
echo ""
echo "Detach from session:"
echo "  Ctrl+b + d"
echo ""

