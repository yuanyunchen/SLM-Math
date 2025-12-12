#!/bin/bash

# Train with GRPO (Group Relative Policy Optimization)
# Configuration matches report: lr 5e-6, KL 0.05, batch 64, 1 epoch
# Usage: bash scripts/train_rl_grpo.sh

python models/train_rl_grpo.py \
    --sft_checkpoint checkpoints/sft_lora_final \
    --config configs/rl_grpo_config.yaml \
    --epochs 1 \
    --learning_rate 5e-6 \
    --kl_coef 0.05 \
    --gpus 0

