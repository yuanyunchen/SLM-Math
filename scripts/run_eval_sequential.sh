#!/bin/bash
################################################################################
# Sequential Multi-Agent Evaluation Script
# 
# Runs all agents ONE AT A TIME with vLLM for maximum GPU utilization
# This is the RECOMMENDED way to run with vLLM backend
################################################################################

set -e

# Force GPU selection
export CUDA_VISIBLE_DEVICES=2

################################################################################
# Configuration
################################################################################

MODEL="${MODEL:-Qwen2.5-Math-1.5B}"
ROUND_NAME="${ROUND:-multi_agent}"
DATASET="${DATASET:-gsm8k}"
COUNT="${COUNT:-10}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
MAX_SUBPROBLEMS="${MAX_SUBPROBLEMS:-5}"
NUM_RUNS="${NUM_RUNS:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
DETAILED="${DETAILED:-false}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# vLLM settings (vLLM handles batching internally)
BATCH_SIZE="${BATCH_SIZE:-1}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"

# List of agents to evaluate
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
echo "SEQUENTIAL MULTI-AGENT EVALUATION"
echo "=========================================================================="
echo "Configuration:"
echo "  GPU:            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Model:          $MODEL"
echo "  Dataset:        $DATASET"
echo "  Sample Count:   $COUNT"
echo "  Round Name:     $ROUND_NAME"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Backend:        $INFERENCE_BACKEND"
echo "  Agents:         ${#AGENTS[@]}"
echo "=========================================================================="
echo ""

################################################################################
# Run Evaluations Sequentially
################################################################################

RESULT_DIRS=()
FAILED_AGENTS=()
START_TIME=$(date +%s)

echo "Starting sequential evaluation at $(date)"
echo ""

for agent in "${AGENTS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] Running: $agent"
    echo "----------------------------------------------------------------------"
    
    AGENT_START=$(date +%s)
    
    # Build command
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
    
    echo "Command: $CMD"
    echo ""
    
    # Run synchronously (NOT in background)
    if eval $CMD; then
        AGENT_END=$(date +%s)
        AGENT_DURATION=$((AGENT_END - AGENT_START))
        
        echo ""
        echo "✓ SUCCESS: $agent completed in ${AGENT_DURATION}s"
        
        # Find the most recent result directory
        LATEST_DIR=$(find results -type d -name "${ROUND_NAME}_${MODEL}_${DATASET}_*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [ -n "$LATEST_DIR" ] && [ -d "$LATEST_DIR" ]; then
            RESULT_DIRS+=("$LATEST_DIR")
            echo "Results: $LATEST_DIR"
        else
            echo "WARNING: Could not find result directory for $agent"
        fi
    else
        echo ""
        echo "✗ FAILED: $agent evaluation failed"
        FAILED_AGENTS+=("$agent")
    fi
    
    echo ""
done

################################################################################
# Summary
################################################################################

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================================================="
echo "Evaluation Complete!"
echo "=========================================================================="
echo "Total Time: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s)"
echo "Successful: $((${#AGENTS[@]} - ${#FAILED_AGENTS[@]}))/${#AGENTS[@]}"

if [ ${#FAILED_AGENTS[@]} -gt 0 ]; then
    echo "Failed agents: ${FAILED_AGENTS[@]}"
fi

echo ""
echo "Result directories found: ${#RESULT_DIRS[@]}"
for dir in "${RESULT_DIRS[@]}"; do
    echo "  - $dir"
done

echo ""
echo "=========================================================================="

# Run comparison analysis if we have results
if [ ${#RESULT_DIRS[@]} -gt 1 ]; then
    echo "Running comparison analysis..."
    python scripts/compare_agents.py "${RESULT_DIRS[@]}" || echo "Comparison script not available"
fi

echo "Done!"

