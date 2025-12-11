"""
GRPO (Group Relative Policy Optimization) Reinforcement Learning Training
Wrapper for train_rl_base.py with standard GRPO configuration

This script trains with GRPO after SFT initialization:
- Learning rate: 5e-6
- KL coefficient: 0.05
- Batch size: 16 prompts, K=2 responses (32 total)
- Gradient accumulation: 4 (effective batch 64)
- Epochs: 1

Usage:
    python models/train_rl_grpo.py --sft_checkpoint checkpoints/sft_lora --gpus 0
    python models/train_rl_grpo.py --config configs/rl_grpo_config.yaml
"""

import sys
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the base RL trainer
from models.train_rl_base import main as rl_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO RL training")
    parser.add_argument('--sft_checkpoint', type=str, 
                        default='checkpoints/sft_lora_final',
                        help='SFT checkpoint to initialize from')
    parser.add_argument('--config', type=str, 
                        default='configs/rl_grpo_config.yaml',
                        help='GRPO config file')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--kl_coef', type=float, default=0.05, help='KL coefficient')
    
    args = parser.parse_args()
    
    # Build sys.argv for RL base script
    sys.argv = [
        'train_rl_base.py',
        '--config', args.config,
        '--model_path', args.sft_checkpoint,
        '--gpus', args.gpus,
        '--epochs', str(args.epochs),
        '--learning_rate', str(args.learning_rate),
        '--kl_coef', str(args.kl_coef),
    ]
    
    if args.output:
        sys.argv.extend(['--output_dir', args.output])
    
    # Run RL main
    rl_main()

