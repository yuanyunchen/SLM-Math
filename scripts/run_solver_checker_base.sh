#!/bin/bash
# Run solver_checker_base agent separately
# This agent was excluded from the main 8-GPU run

MODEL="Qwen2.5-Math-1.5B"
ROUND="full_eval"
COUNT=500
MAX_ITERATIONS=3
AGENT="solver_checker"  # Base version uses "solver_checker" not "solver_checker_base"
SESSION_NAME="solver_checker"  # Changed to match agent name

# You can choose which GPU to use (default: GPU 0, will share with agent_with_python_tools)
# Or use a free GPU if available
GPU=${1:-0}  # Default to GPU 0, can override with: ./run_solver_checker_base.sh 1

echo "=========================================="
echo "Launching solver_checker (base version)"
echo "=========================================="
echo "GPU: $GPU"
echo "Model: $MODEL"
echo "Round: $ROUND"
echo "Count: $COUNT per dataset"
echo "Max Iterations: $MAX_ITERATIONS"
echo "=========================================="
echo ""

# Kill existing session if it exists
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Warning: Session $SESSION_NAME already exists, killing it..."
    tmux kill-session -t $SESSION_NAME
fi

# Create new session for this agent
tmux new-session -d -s $SESSION_NAME

# Send commands to session
tmux send-keys -t $SESSION_NAME "cd /root/autodl-tmp/SLM-Math" C-m
tmux send-keys -t $SESSION_NAME "conda activate slm_math" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=$GPU" C-m

# Run gsm8k first
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting $AGENT on GPU $GPU - GSM8K'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "python -m evaluation.eval_agent --model '$MODEL' --agent '$AGENT' --round '$ROUND' --dataset 'gsm8k' --count $COUNT --max_iterations $MAX_ITERATIONS --detailed 'false' --save_interval 50" C-m

# Then run math
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting $AGENT on GPU $GPU - MATH'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "python -m evaluation.eval_agent --model '$MODEL' --agent '$AGENT' --round '$ROUND' --dataset 'math' --count $COUNT --max_iterations $MAX_ITERATIONS --detailed 'false' --save_interval 50" C-m

# Notify completion
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'COMPLETED: $AGENT on GPU $GPU'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m

echo ""
echo "✓ solver_checker (base version) launched in tmux session: $SESSION_NAME"
echo ""
echo "To monitor:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach:"
echo "  Ctrl+b then d"
echo ""

