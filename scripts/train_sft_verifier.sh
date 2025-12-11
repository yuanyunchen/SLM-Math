#!/bin/bash

# Train verifier model with SFT
# Usage: bash scripts/train_sft_verifier.sh

python models/train_sft_verifier.py \
    --model pretrained_models/Qwen2.5-Math-1.5B \
    --data data/sft_verifier_training/sft_verifier_training_formatted.jsonl \
    --mode lora \
    --lora_rank 16 \
    --epochs 3 \
    --batch_size 8 \
    --gradient_accumulation 4 \
    --learning_rate 1e-4 \
    --max_seq_length 2048 \
    --gpus 0

