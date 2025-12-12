#!/bin/bash

# Train with SFT using LoRA (Low-Rank Adaptation)
# Configuration matches report: rank 16, lr 1e-4, batch 128, 2 epochs
# Usage: bash scripts/train_sft_lora.sh

python models/train_sft_lora.py \
    --model pretrained_models/Qwen2.5-Math-1.5B \
    --data data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json \
    --lora_rank 16 \
    --epochs 2 \
    --batch_size 16 \
    --gradient_accumulation 8 \
    --learning_rate 1e-4 \
    --gpus 0

