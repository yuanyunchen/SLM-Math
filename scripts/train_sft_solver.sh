#!/bin/bash

# Train solver model with code generation using SFT
# Usage: bash scripts/train_sft_solver.sh

python models/train_sft_solver.py \
    --model pretrained_models/Qwen2.5-Math-1.5B \
    --data data/sft_solver_training/run_1209_1011/reasoning_code_v2_gsm8k_train_math_train_1209_1011_code_box_eq_gt.jsonl \
    --mode lora \
    --lora_rank 16 \
    --epochs 2 \
    --batch_size 16 \
    --gradient_accumulation 8 \
    --learning_rate 1e-4 \
    --max_seq_length 2048 \
    --gpus 0

