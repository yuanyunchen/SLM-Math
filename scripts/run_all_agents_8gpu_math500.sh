#!/bin/bash
# Master script to run 8 agents on 8 GPUs in tmux - MATH500 ONLY
# Each agent runs in its own tmux session

MODEL="Qwen2.5-Math-1.5B"
ROUND="full_eval"
DATASET="math500"  # Only math500 dataset
COUNT=500
MAX_ITERATIONS=3
NUM_RUNS=3  # For majority_vote
MAX_SUBPROBLEMS=3  # For plan_and_reflection

# Agent-to-GPU mapping (8 agents, 8 GPUs)
declare -A AGENT_GPU_MAP=(
    ["agent_with_python_tools"]=0
    ["majority_vote"]=1
    ["plan_and_reflection"]=2
    ["solver_checker_chat"]=3
    ["solver_checker_stateless"]=4
    ["solver_checker_summarizer"]=5
    ["solver_checker_summarizer_chat"]=6
    ["solver_checker_with_tools"]=7
)

echo "=========================================="
echo "Starting MATH500 Evaluation on 8 GPUs"
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
    session_name="$agent"
    
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
    tmux send-keys -t $session_name "conda activate slm_math" C-m
    tmux send-keys -t $session_name "export CUDA_VISIBLE_DEVICES=$gpu" C-m
    
    # Build command based on agent type
    if [ "$agent" = "majority_vote" ]; then
        # majority_vote uses --num_runs instead of --max_iterations
        math500_cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '$ROUND' --dataset '$DATASET' --count $COUNT --num_runs $NUM_RUNS --detailed 'false' --save_interval 50"
    elif [ "$agent" = "plan_and_reflection" ]; then
        # plan_and_reflection uses both --max_iterations and --max_subproblems
        math500_cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '$ROUND' --dataset '$DATASET' --count $COUNT --max_iterations $MAX_ITERATIONS --max_subproblems $MAX_SUBPROBLEMS --detailed 'false' --save_interval 50"
    else
        # Other agents use standard --max_iterations
        math500_cmd="python -m evaluation.eval_agent --model '$MODEL' --agent '$agent' --round '$ROUND' --dataset '$DATASET' --count $COUNT --max_iterations $MAX_ITERATIONS --detailed 'false' --save_interval 50"
    fi
    
    # Run math500
    tmux send-keys -t $session_name "echo '========================================'" C-m
    tmux send-keys -t $session_name "echo 'Starting $agent on GPU $gpu - MATH500'" C-m
    tmux send-keys -t $session_name "echo '========================================'" C-m
    tmux send-keys -t $session_name "$math500_cmd" C-m
    
    # Notify completion
    tmux send-keys -t $session_name "echo '========================================'" C-m
    tmux send-keys -t $session_name "echo 'COMPLETED: $agent on GPU $gpu'" C-m
    tmux send-keys -t $session_name "echo '========================================'" C-m
done

echo ""
echo "=========================================="
echo "All agents launched in separate tmux sessions"
echo "=========================================="
echo ""
echo "To monitor specific agent:"
echo "  tmux attach -t <agent_name>"
echo ""
echo "Examples:"
echo "  tmux attach -t agent_with_python_tools"
echo "  tmux attach -t majority_vote"
echo "  tmux attach -t solver_checker_chat"
echo ""
echo "List all sessions:"
echo "  tmux ls"
echo ""
echo "Detach from session:"
echo "  Ctrl+b + d"
echo ""
echo "Agent-GPU-Session mapping:"
for agent in "${!AGENT_GPU_MAP[@]}"; do
    gpu=${AGENT_GPU_MAP[$agent]}
    echo "  GPU $gpu: $agent (tmux session: $agent)"
done | sort
echo ""
echo "Parameters:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Round: $ROUND"
echo "  Count: $COUNT samples"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Num Runs (majority_vote): $NUM_RUNS"
echo "  Max Subproblems (plan_and_reflection): $MAX_SUBPROBLEMS"
echo ""
echo "To check status:"
echo "  ./scripts/list_agent_sessions.sh"
echo "  ./scripts/check_agent_status.sh"
echo ""


