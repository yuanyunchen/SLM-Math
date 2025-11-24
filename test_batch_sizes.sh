#!/bin/bash

# Test SFT batch size on GPU 0
echo "Testing SFT batch_size=32 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python models/train_sft_baseline.py \
    --mode sft \
    --model_name pretrained_models/Qwen2.5-Math-1.5B \
    --data_path data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json \
    --num_epochs 1 \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gpus 0 2>&1 | grep -E "Model loaded|Dataset size|OOM|OutOfMemoryError|error" &

SFT_PID=$!
sleep 30
if ps -p $SFT_PID > /dev/null; then
    echo "✓ SFT batch_size=32 works on GPU 0"
    kill $SFT_PID 2>/dev/null
    SFT_BATCH=32
else
    echo "✗ SFT batch_size=32 failed, trying 24..."
    CUDA_VISIBLE_DEVICES=0 python models/train_sft_baseline.py \
        --mode sft \
        --model_name pretrained_models/Qwen2.5-Math-1.5B \
        --data_path data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json \
        --num_epochs 1 \
        --batch_size 24 \
        --gradient_accumulation_steps 1 \
        --gpus 0 2>&1 | grep -E "Model loaded|Dataset size|OOM" &
    SFT_PID=$!
    sleep 30
    if ps -p $SFT_PID > /dev/null; then
        echo "✓ SFT batch_size=24 works on GPU 0"
        kill $SFT_PID 2>/dev/null
        SFT_BATCH=24
    else
        echo "✗ SFT batch_size=24 also failed"
        SFT_BATCH=8
    fi
fi

pkill -f "train_sft_baseline.py" 2>/dev/null
sleep 3

# Test LoRA batch size on GPU 1
echo "Testing LoRA batch_size=64 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python models/train_sft_baseline.py \
    --mode lora \
    --lora_rank 16 \
    --model_name pretrained_models/Qwen2.5-Math-1.5B \
    --data_path data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json \
    --num_epochs 1 \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --gpus 1 2>&1 | grep -E "Model loaded|Dataset size|OOM" &

LORA_PID=$!
sleep 30
if ps -p $LORA_PID > /dev/null; then
    echo "✓ LoRA batch_size=64 works on GPU 1"
    kill $LORA_PID 2>/dev/null
    LORA_BATCH=64
else
    echo "✗ LoRA batch_size=64 failed, trying 48..."
    CUDA_VISIBLE_DEVICES=1 python models/train_sft_baseline.py \
        --mode lora \
        --lora_rank 16 \
        --model_name pretrained_models/Qwen2.5-Math-1.5B \
        --data_path data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json \
        --num_epochs 1 \
        --batch_size 48 \
        --gradient_accumulation_steps 1 \
        --gpus 1 2>&1 | grep -E "Model loaded|Dataset size|OOM" &
    LORA_PID=$!
    sleep 30
    if ps -p $LORA_PID > /dev/null; then
        echo "✓ LoRA batch_size=48 works on GPU 1"
        kill $LORA_PID 2>/dev/null
        LORA_BATCH=48
    else
        echo "✗ LoRA batch_size=48 also failed"
        LORA_BATCH=16
    fi
fi

pkill -f "train_sft_baseline.py" 2>/dev/null
sleep 2

echo ""
echo "=========================================="
echo "Final Results:"
echo "  SFT batch size: $SFT_BATCH (GPU 0)"
echo "  LoRA batch size: $LORA_BATCH (GPU 1)"
echo "=========================================="

