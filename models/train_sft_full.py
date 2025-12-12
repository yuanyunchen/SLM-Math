"""
SFT Training with Full Fine-Tuning (All Parameters)
Wrapper for train_sft_baseline.py with full SFT mode

This script trains all parameters of Qwen2.5-Math-1.5B:
- Learning rate: 5e-5 (lower to prevent catastrophic forgetting)
- Batch size: 128 (effective)
- Epochs: 2

Usage:
    python models/train_sft_full.py --gpus 0,1
    python models/train_sft_full.py --learning_rate 1e-5 --gpus 0,1,2,3
"""

import sys
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the baseline trainer
from models.train_sft_baseline import main as baseline_main

if __name__ == "__main__":
    # Override sys.argv to force full SFT mode
    parser = argparse.ArgumentParser(description="Full SFT training (all parameters)")
    parser.add_argument('--gpus', type=str, default='0,1', help='GPU IDs')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Per-device batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (lower for full SFT)')
    parser.add_argument('--model', type=str, default='pretrained_models/Qwen2.5-Math-1.5B', help='Model path')
    parser.add_argument('--data', type=str, 
                        default='data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json',
                        help='Training data path')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Build sys.argv for baseline script
    sys.argv = [
        'train_sft_baseline.py',
        '--mode', 'sft',  # Full SFT mode
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

