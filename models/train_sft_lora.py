"""
SFT Training with LoRA (Low-Rank Adaptation)
Wrapper for train_sft_baseline.py with LoRA mode

This script trains Qwen2.5-Math-1.5B using LoRA adapters:
- LoRA rank: 16
- Learning rate: 1e-4
- Batch size: 128 (effective)
- Epochs: 2

Usage:
    python models/train_sft_lora.py --gpus 0
    python models/train_sft_lora.py --lora_rank 32 --gpus 0,1
"""

import sys
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the baseline trainer
from models.train_sft_baseline import main as baseline_main

if __name__ == "__main__":
    # Override sys.argv to force LoRA mode
    parser = argparse.ArgumentParser(description="SFT training with LoRA")
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Per-device batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='pretrained_models/Qwen2.5-Math-1.5B', help='Model path')
    parser.add_argument('--data', type=str, 
                        default='data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json',
                        help='Training data path')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Build sys.argv for baseline script
    sys.argv = [
        'train_sft_baseline.py',
        '--mode', 'lora',
        '--lora_rank', str(args.lora_rank),
        '--gpus', args.gpus,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--gradient_accumulation', str(args.gradient_accumulation),
        '--learning_rate', str(args.learning_rate),
        '--model', args.model,
        '--data', args.data,
    ]
    
    if args.output:
        sys.argv.extend(['--output', args.output])
    
    # Run baseline main
    baseline_main()
