#!/bin/bash

# Multi-agent RL training for solver-verifier system
# Usage: bash scripts/train_rl_solver_verifier_multi.sh

python models/train_rl_solver_verifier_multi.py \
    --solver_path pretrained_models/Qwen2.5-Math-1.5B \
    --verifier_path pretrained_models/Qwen2.5-Math-1.5B \
    --shared_model \
    --output results/rl_multi_agent \
    --max_turns 3 \
    --epochs 1 \
    --batch_size 5 \
    --learning_rate 5e-6 \
    --seed 42

