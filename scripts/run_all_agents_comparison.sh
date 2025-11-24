#!/bin/bash

################################################################################
# Multi-Agent Evaluation and Comparison Script
#
# This script evaluates all available agents and generates a comprehensive
# comparison analysis showing improved/degraded performance across methods.
################################################################################

set -e

# Force GPU selection before anything else
export CUDA_VISIBLE_DEVICES=2

################################################################################
# Configuration
################################################################################

# Model to evaluate
MODEL="${MODEL:-Qwen2.5-Math-1.5B}"

# Test round name (used for organizing results)
ROUND_NAME="${ROUND:-multi_agent_comparison}"

# Dataset to use (options: gsm8k, math)
DATASET="${DATASET:-gsm8k}"

# Number of test cases to run (set to 0 to run the entire dataset)
COUNT="${COUNT:-10}"

# Maximum iterations for iterative agents
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"

# Detailed output mode
DETAILED="${DETAILED:-false}"

# Save interval
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

################################################################################
# Agent List
################################################################################

# All available agents
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

# Display selected agents
echo "=========================================================================="
echo "Multi-Agent Evaluation Script"
echo "=========================================================================="
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Count: $COUNT"
echo "Round: $ROUND_NAME"
echo "Max Iterations: $MAX_ITERATIONS"
echo ""
echo "Agents to evaluate:"
for agent in "${AGENTS[@]}"; do
    echo "  - $agent"
done
echo "=========================================================================="
echo ""

# Check model path
MODEL_PATH="pretrained_models/${MODEL}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: Model not found at $MODEL_PATH"
    echo "Please ensure the model exists or update the MODEL variable."
    exit 1
fi

################################################################################
# Run Evaluations
################################################################################

RESULT_DIRS=()
FAILED_AGENTS=()

echo "Starting evaluations..."
echo ""

for agent in "${AGENTS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Evaluating agent: $agent"
    echo "----------------------------------------------------------------------"
    
    AGENT_START_TIME=$(date +%s)
    
    # Build command with agent-specific parameters
    CMD="python -m evaluation.eval_agent \
        --agent \"${agent}\" \
        --model \"${MODEL}\" \
        --dataset \"${DATASET}\" \
        --round \"${ROUND_NAME}\" \
        --count ${COUNT} \
        --max_iterations ${MAX_ITERATIONS} \
        --detailed \"${DETAILED}\" \
        --save_interval ${SAVE_INTERVAL}"
    
    # Add agent-specific parameters
    if [ "$agent" == "majority_vote" ]; then
        CMD="$CMD --num_runs 5 --temperature 0.7 --top_p 0.95"
    elif [ "$agent" == "plan_and_reflection" ]; then
        CMD="$CMD --max_subproblems 5"
    elif [ "$agent" == "solver_checker_with_tools" ]; then
        CMD="$CMD --enable_solver_tools true --enable_checker_tools true"
    fi
    
    echo "Running: $CMD"
    echo ""
    
    # Run evaluation and capture result directory
    if eval $CMD; then
        AGENT_END_TIME=$(date +%s)
        AGENT_DURATION=$((AGENT_END_TIME - AGENT_START_TIME))
        echo ""
        echo "Agent '$agent' completed successfully in ${AGENT_DURATION}s"
        
        # Find the most recent result directory for this agent
        RESULT_DIR=$(find results -type d -name "${ROUND_NAME}_${MODEL}_${DATASET}_*" | sort -r | head -1)
        if [ -n "$RESULT_DIR" ]; then
            RESULT_DIRS+=("$RESULT_DIR")
            echo "Results saved to: $RESULT_DIR"
        fi
    else
        echo ""
        echo "ERROR: Agent '$agent' failed!"
        FAILED_AGENTS+=("$agent")
    fi
    
    echo ""
done

################################################################################
# Generate Comparison Analysis
################################################################################

echo "=========================================================================="
echo "Generating Comparison Analysis"
echo "=========================================================================="
echo ""

# Create comparison script
COMPARISON_SCRIPT="scripts/compare_agents.py"

if [ ${#RESULT_DIRS[@]} -gt 0 ]; then
    echo "Found ${#RESULT_DIRS[@]} result directories to compare"
    echo ""
    
    # Call comparison analysis script
    python -m scripts.compare_agents \
        --result_dirs "${RESULT_DIRS[@]}" \
        --output_dir "results/${ROUND_NAME}_comparison_$(date +%m%d_%H%M)" \
        --model "$MODEL" \
        --dataset "$DATASET"
else
    echo "No result directories found. Skipping comparison."
fi

################################################################################
# Final Summary
################################################################################

echo ""
echo "=========================================================================="
echo "Evaluation Complete"
echo "=========================================================================="
echo ""
echo "Successfully evaluated: $((${#AGENTS[@]} - ${#FAILED_AGENTS[@]}))/${#AGENTS[@]} agents"

if [ ${#FAILED_AGENTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed agents:"
    for failed in "${FAILED_AGENTS[@]}"; do
        echo "  - $failed"
    done
fi

echo ""
echo "Result directories:"
for dir in "${RESULT_DIRS[@]}"; do
    echo "  - $dir"
done

echo ""
echo "=========================================================================="

