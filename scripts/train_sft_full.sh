#!/bin/bash

# Train with Full SFT (all parameters)
# Configuration matches report: lr 5e-5, batch 128, 2 epochs
# Usage: bash scripts/train_sft_full.sh

python models/train_sft_full.py \
    --model pretrained_models/Qwen2.5-Math-1.5B \
    --data data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json \
    --epochs 2 \
    --batch_size 16 \
    --gradient_accumulation 8 \
    --learning_rate 5e-5 \
    --gpus 0,1

