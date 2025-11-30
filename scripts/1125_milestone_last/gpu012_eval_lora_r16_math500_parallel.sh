#!/bin/bash

################################################################################
# Parallel Evaluation: LoRA r16 checkpoint-298 on MATH500
# GPUs: 0, 1, 2
# Workers: 8 per GPU = 24 total
# Checkpoint: checkpoints/lora_r16_lora_r16_20251125_164902/checkpoint-298
################################################################################

set -e  # Exit on error

################################################################################
# Configuration

# Model
MODEL="Qwen2.5-Math-1.5B"
CHECKPOINT="checkpoints/lora_r16_lora_r16_20251125_164902/checkpoint-298"

# Dataset
DATASET="math500"
TOTAL_SAMPLES=500

# Round name
ROUND_NAME="1125_milestone_lora_r16_ckpt298_math500"

# GPU configuration
GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}

# Worker configuration
WORKERS_PER_GPU=8
TOTAL_WORKERS=$((WORKERS_PER_GPU * NUM_GPUS))

# Evaluation settings
BATCH_SIZE=1
MODE="standard"
LOG_SAMPLES="true"
SAVE_INTERVAL=5

################################################################################
# Pre-flight

echo "=========================================="
echo "Parallel Evaluation Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: $DATASET ($TOTAL_SAMPLES samples)"
echo "  Round: $ROUND_NAME"
echo ""
echo "Parallelization:"
echo "  GPUs: ${GPUS[*]}"
echo "  Workers per GPU: $WORKERS_PER_GPU"
echo "  Total workers: $TOTAL_WORKERS"
echo ""

# Calculate samples per worker
SAMPLES_PER_WORKER=$((TOTAL_SAMPLES / TOTAL_WORKERS))
REMAINDER=$((TOTAL_SAMPLES % TOTAL_WORKERS))

echo "Distribution:"
echo "  Samples per worker: ~$SAMPLES_PER_WORKER"
echo "  Remainder (extra samples for first $REMAINDER workers): $REMAINDER"
echo "=========================================="
echo ""

################################################################################
# Launch workers

echo "Launching $TOTAL_WORKERS workers..."
echo ""

PIDS=()
START_IDX=0
WORKER_ID=0

for gpu_idx in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$gpu_idx]}
    
    echo "--- GPU $GPU_ID: Launching $WORKERS_PER_GPU workers ---"
    
    for ((w=0; w<WORKERS_PER_GPU; w++)); do
        # Calculate this worker's sample count
        if [ $WORKER_ID -lt $REMAINDER ]; then
            WORKER_SAMPLES=$((SAMPLES_PER_WORKER + 1))
        else
            WORKER_SAMPLES=$SAMPLES_PER_WORKER
        fi
        
        # Skip if no samples left
        if [ $WORKER_SAMPLES -le 0 ]; then
            echo "  [Worker $WORKER_ID] Skipped (no samples)"
            continue
        fi
        
        END_IDX=$((START_IDX + WORKER_SAMPLES - 1))
        echo "  [Worker $WORKER_ID] GPU=$GPU_ID, samples=[$START_IDX-$END_IDX] ($WORKER_SAMPLES)"
        
        # Launch worker in background
        (
            CUDA_VISIBLE_DEVICES=$GPU_ID python -m evaluation.eval \
                --model "$MODEL" \
                --checkpoint "$CHECKPOINT" \
                --round "${ROUND_NAME}_w${WORKER_ID}" \
                --dataset "$DATASET" \
                --count $WORKER_SAMPLES \
                --start $START_IDX \
                --mode "$MODE" \
                --batch_size $BATCH_SIZE \
                --detailed "false" \
                --log_samples "$LOG_SAMPLES" \
                --greedy "true" \
                --save_interval $SAVE_INTERVAL \
                2>&1 | while read line; do echo "[W$WORKER_ID] $line"; done
        ) &
        
        PIDS+=($!)
        START_IDX=$((START_IDX + WORKER_SAMPLES))
        WORKER_ID=$((WORKER_ID + 1))
    done
    echo ""
done

echo "=========================================="
echo "All $WORKER_ID workers launched!"
echo "PIDs: ${PIDS[*]}"
echo "=========================================="
echo ""
echo "Waiting for all workers to complete..."
echo "(This may take a while)"
echo ""

################################################################################
# Wait for workers

FAILED=0
COMPLETED=0

for pid in "${PIDS[@]}"; do
    if wait $pid; then
        COMPLETED=$((COMPLETED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "[ERROR] Worker (PID $pid) failed!"
    fi
done

################################################################################
# Summary

echo ""
echo "=========================================="
echo "Parallel Evaluation Complete!"
echo "=========================================="
echo "Workers: $COMPLETED completed, $FAILED failed"
echo ""
echo "Results saved to:"
echo "  results/${ROUND_NAME}_w*/answers/*.json"
echo ""
echo "Log files:"
echo "  results/${ROUND_NAME}_w*/log/*_samples.log"
echo "=========================================="

# Aggregate results
echo ""
echo "Aggregating results..."
echo ""

TOTAL_CORRECT=0
TOTAL_TESTED=0

for ((i=0; i<WORKER_ID; i++)); do
    RESULT_DIR="results/${ROUND_NAME}_w${i}_${MODEL}_${DATASET}_"*
    METRICS_FILE=$(ls -t ${RESULT_DIR}/metrics.csv 2>/dev/null | head -1)
    
    if [ -f "$METRICS_FILE" ]; then
        # Extract correct and total from CSV (skip header)
        CORRECT=$(tail -1 "$METRICS_FILE" | cut -d',' -f5)
        TOTAL=$(tail -1 "$METRICS_FILE" | cut -d',' -f4)
        ACCURACY=$(tail -1 "$METRICS_FILE" | cut -d',' -f6)
        
        echo "  Worker $i: $CORRECT/$TOTAL ($(echo "scale=1; $ACCURACY * 100" | bc)%)"
        
        TOTAL_CORRECT=$((TOTAL_CORRECT + CORRECT))
        TOTAL_TESTED=$((TOTAL_TESTED + TOTAL))
    else
        echo "  Worker $i: [No results found]"
    fi
done

echo ""
echo "=========================================="
if [ $TOTAL_TESTED -gt 0 ]; then
    FINAL_ACCURACY=$(echo "scale=4; $TOTAL_CORRECT / $TOTAL_TESTED * 100" | bc)
    echo "FINAL RESULTS:"
    echo "  Total: $TOTAL_CORRECT / $TOTAL_TESTED"
    echo "  Accuracy: ${FINAL_ACCURACY}%"
else
    echo "No results to aggregate!"
fi
echo "=========================================="


