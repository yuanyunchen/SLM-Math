#!/bin/bash

################################################################################
# Evaluate All Agents Script
# 
# Usage: ./scripts/run_eval_all_agents.sh [options]
# 
# This script evaluates all available agents and generates a comparison report.
# You can customize the evaluation by setting environment variables or editing
# the configuration section below.
################################################################################

set -e

# Force GPU selection before anything else
export CUDA_VISIBLE_DEVICES=2

################################################################################
# Configuration Settings
################################################################################

# Model to evaluate
MODEL="${MODEL:-Qwen2.5-Math-1.5B}"

# Test round name (used for organizing results)
ROUND_NAME="${ROUND:-multi_agent}"

# Dataset to use (options: gsm8k, math)
DATASET="${DATASET:-gsm8k}"

# Number of test cases to run (set to 0 to run the entire dataset)
COUNT="${COUNT:-10}"

# Maximum iterations for iterative agents (reduced for speed)
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"  # Reduced from 5 to 3 for 40% faster execution

# Maximum sub-problems for plan_and_reflection
MAX_SUBPROBLEMS="${MAX_SUBPROBLEMS:-5}"

# Majority vote parameters
NUM_RUNS="${NUM_RUNS:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

# Detailed output mode
DETAILED="${DETAILED:-false}"

# Save interval
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# Inference settings
BATCH_SIZE="${BATCH_SIZE:-1}"  # Set to 1 for vLLM (vLLM handles batching internally)
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"  # 'vllm' for maximum speed, 'transformers' for compatibility

# Execution mode: sequential (one agent at a time) or parallel (all agents at once)
MODE="${MODE:-sequential}"  # Sequential is REQUIRED for vLLM

################################################################################
# Agent Configuration
################################################################################

# List of agents to evaluate
# Comment out any agents you don't want to evaluate
AGENTS=(
    "solver_checker"
    "solver_checker_chat"
    "solver_checker_trivial_chat"
    "solver_checker_with_tools"
    "solver_checker_summarizer"
    "solver_checker_summarizer_chat"
    "agent_with_python_tools"
    "majority_vote"
    "plan_and_reflection"
)

################################################################################
# Display Configuration
################################################################################

echo "=========================================================================="
echo "Multi-Agent Evaluation"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  GPU:            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Model:          $MODEL"
echo "  Dataset:        $DATASET"
echo "  Sample Count:   $COUNT"
echo "  Round Name:     $ROUND_NAME"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Backend:        $INFERENCE_BACKEND"
echo "  Detailed:       $DETAILED"
echo ""
echo "Agents to evaluate (${#AGENTS[@]} total):"
for i in "${!AGENTS[@]}"; do
    printf "  %2d. %s\n" $((i+1)) "${AGENTS[$i]}"
done
echo "=========================================================================="
echo ""

# Verify model exists
MODEL_PATH="pretrained_models/${MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please ensure the model exists or update the MODEL variable."
    exit 1
fi

echo "Model path verified: $MODEL_PATH"
echo ""

################################################################################
# Run Evaluations (PARALLEL - All agents run concurrently on same GPU)
################################################################################

RESULT_DIRS=()
FAILED_AGENTS=()
START_TIME=$(date +%s)

# Create temporary directory for tracking PIDs and logs
TMP_DIR=$(mktemp -d)
LOG_DIR="logs/parallel_$(date +%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting PARALLEL evaluations at $(date)"
echo "All agents will run concurrently on GPU $CUDA_VISIBLE_DEVICES"
echo "Logs: $LOG_DIR/"
echo ""

# Launch all agents in parallel
PIDS=()
declare -A AGENT_PID_MAP

for agent in "${AGENTS[@]}"; do
    echo "[$(date +%H:%M:%S)] Launching: $agent (background)"
    
    # Build base command
    CMD="python -m evaluation.eval_agent"
    CMD="$CMD --agent ${agent}"
    CMD="$CMD --model ${MODEL}"
    CMD="$CMD --dataset ${DATASET}"
    CMD="$CMD --round ${ROUND_NAME}"
    CMD="$CMD --count ${COUNT}"
    CMD="$CMD --detailed ${DETAILED}"
    CMD="$CMD --save_interval ${SAVE_INTERVAL}"
    CMD="$CMD --batch_size ${BATCH_SIZE}"
    CMD="$CMD --inference_backend ${INFERENCE_BACKEND}"
    
    # Add agent-specific parameters
    case "$agent" in
        solver_checker|solver_checker_chat|solver_checker_trivial_chat|solver_checker_with_tools|solver_checker_summarizer|solver_checker_summarizer_chat)
            CMD="$CMD --max_iterations ${MAX_ITERATIONS}"
            
            if [ "$agent" == "solver_checker_with_tools" ]; then
                CMD="$CMD --enable_solver_tools true --enable_checker_tools true"
            fi
            ;;
        
        majority_vote)
            CMD="$CMD --num_runs ${NUM_RUNS} --temperature ${TEMPERATURE} --top_p ${TOP_P}"
            ;;
        
        plan_and_reflection)
            CMD="$CMD --max_iterations ${MAX_ITERATIONS} --max_subproblems ${MAX_SUBPROBLEMS}"
            ;;
    esac
    
    # Run in background and capture PID
    LOG_FILE="$LOG_DIR/${agent}.log"
    eval $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=($PID)
    AGENT_PID_MAP[$PID]=$agent
    echo "  PID: $PID, Log: $LOG_FILE"
done

echo ""
echo "All ${#AGENTS[@]} agents launched in parallel!"
echo "Waiting for completion..."
echo ""

# Monitor progress
COMPLETED=0
TOTAL=${#AGENTS[@]}

while [ $COMPLETED -lt $TOTAL ]; do
    sleep 5
    COMPLETED=0
    for PID in "${PIDS[@]}"; do
        if ! kill -0 $PID 2>/dev/null; then
            COMPLETED=$((COMPLETED + 1))
        fi
    done
    echo "[$(date +%H:%M:%S)] Progress: $COMPLETED/$TOTAL agents completed"
done

echo ""
echo "All agents finished! Collecting results..."
echo ""

# Wait for all processes and check exit codes
for PID in "${PIDS[@]}"; do
    agent="${AGENT_PID_MAP[$PID]}"
    
    if wait $PID; then
        echo "✓ $agent (PID $PID) - SUCCESS"
    else
        echo "✗ $agent (PID $PID) - FAILED"
        FAILED_AGENTS+=("$agent")
    fi
done

echo ""

# Collect result directories (find all matching directories created during this run)
echo "Collecting result directories..."
sleep 2  # Brief pause to ensure all files are written

# Find all result directories created for this round
for result_dir in results/${ROUND_NAME}_${MODEL}_${DATASET}_*/; do
    if [ -d "$result_dir" ]; then
        # Check if this directory has valid results
        if [ -f "$result_dir/metrics.csv" ]; then
            RESULT_DIRS+=("${result_dir%/}")
            agent_name=$(grep -m1 "agent" "$result_dir/metrics.csv" | cut -d',' -f2 2>/dev/null || echo "unknown")
            echo "  Found: $result_dir (agent: $agent_name)"
        fi
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "Parallel execution logs saved in: $LOG_DIR/"

################################################################################
# Generate Comparison Report
################################################################################

echo "=========================================================================="
echo "Generating Comparison Analysis"
echo "=========================================================================="
echo ""

COMPARISON_DIR="results/${ROUND_NAME}_comparison_$(date +%m%d_%H%M)"

if [ ${#RESULT_DIRS[@]} -gt 0 ]; then
    echo "Comparing ${#RESULT_DIRS[@]} agent results..."
    echo "Output: $COMPARISON_DIR"
    echo ""
    
    # Run comparison script
    python -m scripts.compare_agents \
        --result_dirs "${RESULT_DIRS[@]}" \
        --output_dir "$COMPARISON_DIR" \
        --model "$MODEL" \
        --dataset "$DATASET"
    
    echo ""
    echo "Comparison analysis complete!"
    echo "View results in: $COMPARISON_DIR"
else
    echo "ERROR: No result directories found to compare"
fi

################################################################################
# Summary
################################################################################

echo ""
echo "=========================================================================="
echo "EVALUATION SUMMARY"
echo "=========================================================================="
echo ""
echo "Total Time: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"
echo ""
echo "Agents Evaluated:"
echo "  Total:      ${#AGENTS[@]}"
echo "  Successful: $((${#AGENTS[@]} - ${#FAILED_AGENTS[@]}))"
echo "  Failed:     ${#FAILED_AGENTS[@]}"
echo ""

if [ ${#FAILED_AGENTS[@]} -gt 0 ]; then
    echo "Failed Agents:"
    for failed in "${FAILED_AGENTS[@]}"; do
        echo "  - $failed"
    done
    echo ""
fi

if [ ${#RESULT_DIRS[@]} -gt 0 ]; then
    echo "Result Directories:"
    for dir in "${RESULT_DIRS[@]}"; do
        echo "  - $dir"
    done
    echo ""
    
    echo "Comparison Report:"
    echo "  - $COMPARISON_DIR"
    echo ""
    
    # Display quick results if comparison was successful
    if [ -f "$COMPARISON_DIR/agent_comparison.csv" ]; then
        echo "Quick Results (Accuracy):"
        echo "------------------------------------------------------------------------"
        # Use awk to format the CSV nicely
        awk -F',' 'NR==1 {printf "%-35s %10s\n", $1, $2; print "------------------------------------------------------------------------"} NR>1 {printf "%-35s %10.2f%%\n", $1, $2}' "$COMPARISON_DIR/agent_comparison.csv" | head -n 12
        echo ""
    fi
fi

echo "=========================================================================="
echo "Complete! $(date)"
echo "=========================================================================="
echo ""

