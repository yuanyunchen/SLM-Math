#!/bin/bash

################################################################################
# Evaluation Script Template (with Multi-Worker Support)
# 通用评估脚本模板 - 支持多进程并行评估
################################################################################

set -e  # Exit on error

################################################################################
# Configuration Variables (需要修改)

# Model to evaluate
MODEL="Qwen2.5-Math-1.5B"

# Checkpoint path (optional)
# Set to empty string "" to use base pretrained model
# For LoRA: "checkpoints/lora_r16_lora_r16_20251125_164902/final_model"
# For SFT: "checkpoints/sft_lr5e5_sft_20251125_190347/final_model"
CHECKPOINT=""

# Test round name
ROUND_NAME="my_evaluation"

# Dataset: "gsm8k", "math500"
DATASET="gsm8k"

# Number of test cases (0 = full dataset)
COUNT=500

# Evaluation mode: "standard"
MODE="standard"

# Streaming console output (per-sample print)
DETAILED="false"

# Log full sample details to file (question, response, answer)
LOG_SAMPLES="true"

# Batch size (use 1 to avoid continuous batching issues)
BATCH_SIZE=1

# Use greedy decoding (default: true)
GREEDY="true"

# Apply chat template to prompts (default: false)
# Enable for chat-tuned models (e.g., Qwen2.5-Math, Qwen3)
APPLY_CHAT_TEMPLATE="false"

################################################################################
# Multi-Worker Settings (多进程并行)

# Number of workers (0 = single process, no parallelism)
# Workers will be distributed evenly across available GPUs
WORKERS=0

# Available GPUs (comma-separated, e.g., "0,1,2,3")
GPUS="0"

################################################################################
# Advanced Settings (可选)

# Resume from existing results directory (leave empty to start fresh)
RESUME_DIR=""

# Save interval: save intermediate results every N samples
SAVE_INTERVAL=10

################################################################################
# Helper Functions

run_single_worker() {
    local gpu_id=$1
    local start_idx=$2
    local worker_count=$3
    local worker_id=$4
    
    echo "[Worker $worker_id] GPU=$gpu_id, start=$start_idx, count=$worker_count"
    
    # Build command
    local CMD="CUDA_VISIBLE_DEVICES=$gpu_id python -m evaluation.eval \
        --model \"$MODEL\" \
        --round \"${ROUND_NAME}_worker${worker_id}\" \
        --dataset \"$DATASET\" \
        --count $worker_count \
        --start $start_idx \
        --mode \"$MODE\" \
        --batch_size $BATCH_SIZE \
        --detailed \"$DETAILED\" \
        --greedy \"$GREEDY\" \
        --apply_chat_template \"$APPLY_CHAT_TEMPLATE\" \
        --save_interval $SAVE_INTERVAL"
    
    # Add checkpoint parameter if specified
    if [ -n "$CHECKPOINT" ]; then
        CMD="$CMD --checkpoint \"$CHECKPOINT\""
    fi
    
    # Execute
    eval $CMD
}

################################################################################
# Pre-flight Checks

echo "=========================================="
echo "Evaluation Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
fi
echo "  Round: $ROUND_NAME"
echo "  Dataset: $DATASET"
echo "  Count: $COUNT (0=full)"
echo "  Mode: $MODE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Greedy: $GREEDY"
echo "  Apply Chat Template: $APPLY_CHAT_TEMPLATE"
echo "  Workers: $WORKERS"
echo "  GPUs: $GPUS"
echo "=========================================="
echo ""

################################################################################
# Run Evaluation

if [ "$WORKERS" -le 1 ]; then
    #---------------------------------------------------------------------------
    # Single Process Mode
    #---------------------------------------------------------------------------
    echo "Running in single-process mode..."
    
    # Use first GPU
    GPU_ARRAY=$(echo "$GPUS" | tr ',' ' ')
    set -- $GPU_ARRAY
    export CUDA_VISIBLE_DEVICES="$1"
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
    echo ""
    
    # Build command
    CMD="python -m evaluation.eval \
        --model \"$MODEL\" \
        --round \"$ROUND_NAME\" \
        --dataset \"$DATASET\" \
        --count $COUNT \
        --mode \"$MODE\" \
        --batch_size $BATCH_SIZE \
        --detailed \"$DETAILED\" \
        --log_samples \"$LOG_SAMPLES\" \
        --greedy \"$GREEDY\" \
        --apply_chat_template \"$APPLY_CHAT_TEMPLATE\" \
        --save_interval $SAVE_INTERVAL"
    
    # Add checkpoint parameter if specified
    if [ -n "$CHECKPOINT" ]; then
        CMD="$CMD --checkpoint \"$CHECKPOINT\""
    fi
    
    # Add resume parameter if specified
    if [ -n "$RESUME_DIR" ]; then
        CMD="$CMD --resume \"$RESUME_DIR\""
    fi
    
    # Execute
    eval $CMD

else
    #---------------------------------------------------------------------------
    # Multi-Worker Mode
    #---------------------------------------------------------------------------
    echo "Running in multi-worker mode with $WORKERS workers..."
    
    # Parse GPUs
    GPU_ARRAY=$(echo "$GPUS" | tr ',' ' ')
    set -- $GPU_ARRAY
    NUM_GPUS=$#
    
    # Store GPUs in a simple way
    GPU_LIST="$GPUS"
    
    # Get total samples
    if [ "$COUNT" -eq 0 ]; then
        if [ "$DATASET" = "gsm8k" ]; then
            TOTAL_SAMPLES=1319
        elif [ "$DATASET" = "math500" ]; then
            TOTAL_SAMPLES=500
        else
            echo "Error: Unknown dataset $DATASET, please set COUNT manually"
            exit 1
        fi
    else
        TOTAL_SAMPLES=$COUNT
    fi
    
    # Calculate samples per worker
    SAMPLES_PER_WORKER=$((TOTAL_SAMPLES / WORKERS))
    REMAINDER=$((TOTAL_SAMPLES % WORKERS))
    
    echo "Total samples: $TOTAL_SAMPLES"
    echo "Samples per worker: ~$SAMPLES_PER_WORKER"
    echo "GPUs available: $NUM_GPUS"
    echo ""
    
    # Launch workers in parallel
    PIDS=""
    START_IDX=0
    
    i=0
    while [ $i -lt $WORKERS ]; do
        # Calculate worker's sample count (distribute remainder)
        if [ $i -lt $REMAINDER ]; then
            WORKER_COUNT=$((SAMPLES_PER_WORKER + 1))
        else
            WORKER_COUNT=$SAMPLES_PER_WORKER
        fi
        
        # Assign GPU (round-robin)
        GPU_IDX=$((i % NUM_GPUS))
        GPU_ID=$(echo "$GPU_LIST" | cut -d',' -f$((GPU_IDX + 1)))
        
        # Launch worker in background
        echo "Launching worker $i on GPU $GPU_ID: samples [$START_IDX, $((START_IDX + WORKER_COUNT)))"
        
        if [ -n "$CHECKPOINT" ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python -m evaluation.eval \
                --model "$MODEL" \
                --round "${ROUND_NAME}_worker${i}" \
                --dataset "$DATASET" \
                --count $WORKER_COUNT \
                --start $START_IDX \
                --mode "$MODE" \
                --batch_size $BATCH_SIZE \
                --detailed "false" \
                --log_samples "$LOG_SAMPLES" \
                --greedy "$GREEDY" \
                --apply_chat_template "$APPLY_CHAT_TEMPLATE" \
                --save_interval $SAVE_INTERVAL \
                --checkpoint "$CHECKPOINT" &
        else
            CUDA_VISIBLE_DEVICES=$GPU_ID python -m evaluation.eval \
                --model "$MODEL" \
                --round "${ROUND_NAME}_worker${i}" \
                --dataset "$DATASET" \
                --count $WORKER_COUNT \
                --start $START_IDX \
                --mode "$MODE" \
                --batch_size $BATCH_SIZE \
                --detailed "false" \
                --log_samples "$LOG_SAMPLES" \
                --greedy "$GREEDY" \
                --apply_chat_template "$APPLY_CHAT_TEMPLATE" \
                --save_interval $SAVE_INTERVAL &
        fi
        
        PIDS="$PIDS $!"
        START_IDX=$((START_IDX + WORKER_COUNT))
        i=$((i + 1))
    done
    
    echo ""
    echo "All $WORKERS workers launched. Waiting for completion..."
    echo "PIDs:$PIDS"
    echo ""
    
    # Wait for all workers
    FAILED=0
    for pid in $PIDS; do
        if ! wait $pid; then
            echo "Worker (PID $pid) failed!"
            FAILED=$((FAILED + 1))
        fi
    done
    
    if [ $FAILED -gt 0 ]; then
        echo "Warning: $FAILED worker(s) failed!"
    fi
    
    echo ""
    echo "All workers completed!"
    echo "Results saved to: results/${ROUND_NAME}_worker*/"
    echo ""
    echo "To merge results, check the individual worker output directories."
fi

################################################################################
# Complete

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
