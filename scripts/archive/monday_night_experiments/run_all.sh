#!/bin/bash

################################################################################
# Run All Experiments
# Launches all three experiments on GPUs 0, 1, 2 in parallel
################################################################################

set -e

# Change to project root
cd /root/autodl-tmp/SLM-Math

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Starting All Experiments"
echo "=========================================="
echo "GPU 0: Full SFT Training"
echo "GPU 1: RL Baseline Training"
echo "GPU 2: LoRA -> RL Pipeline"
echo ""
echo "W&B Project: slm_math_experiments"
echo "=========================================="

# Start GPU 0: Full SFT
echo "[$(date '+%H:%M:%S')] Starting GPU 0: Full SFT..."
nohup bash scripts/monday_night_experiments/exp_gpu0_sft_full.sh > logs/exp_gpu0_sft_full.log 2>&1 &
PID_GPU0=$!
echo "  PID: $PID_GPU0"
echo "  Log: logs/exp_gpu0_sft_full.log"

# Start GPU 1: RL Baseline
echo "[$(date '+%H:%M:%S')] Starting GPU 1: RL Baseline..."
nohup bash scripts/monday_night_experiments/exp_gpu1_rl_baseline.sh > logs/exp_gpu1_rl_baseline.log 2>&1 &
PID_GPU1=$!
echo "  PID: $PID_GPU1"
echo "  Log: logs/exp_gpu1_rl_baseline.log"

# Start GPU 2: LoRA -> RL
echo "[$(date '+%H:%M:%S')] Starting GPU 2: LoRA -> RL..."
nohup bash scripts/monday_night_experiments/exp_gpu2_lora_then_rl.sh > logs/exp_gpu2_lora_rl.log 2>&1 &
PID_GPU2=$!
echo "  PID: $PID_GPU2"
echo "  Log: logs/exp_gpu2_lora_rl.log"

echo ""
echo "=========================================="
echo "All Experiments Started!"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  GPU 0 (SFT Full): $PID_GPU0"
echo "  GPU 1 (RL Base):  $PID_GPU1"
echo "  GPU 2 (LoRA+RL):  $PID_GPU2"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/exp_gpu0_sft_full.log"
echo "  tail -f logs/exp_gpu1_rl_baseline.log"
echo "  tail -f logs/exp_gpu2_lora_rl.log"
echo ""
echo "Monitor GPUs:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Kill all:"
echo "  kill $PID_GPU0 $PID_GPU1 $PID_GPU2"
echo ""
echo "W&B Dashboard: https://wandb.ai/slm_math_experiments"
echo "=========================================="

# Save PIDs to file for later reference
echo "$PID_GPU0 $PID_GPU1 $PID_GPU2" > logs/experiment_pids.txt
echo "PIDs saved to logs/experiment_pids.txt"

