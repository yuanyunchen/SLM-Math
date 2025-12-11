"""
RL Training for Agent with Code Feedback
Wrapper for train_rl_agent_with_code_feedback.py with cleaner interface

This trains the Code Feedback agent with GRPO:
- Two-step generation: reasoning+code → execute → final answer
- Code execution feedback integration
- Results: GSM8K 84.6%, MATH 67.8% (SFT+RL)

Usage:
    python models/train_rl_code_feedback.py --sft_checkpoint checkpoints/sft_lora
    python models/train_rl_code_feedback.py --config configs/rl_grpo_config.yaml
"""

import sys
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the full implementation
from models.train_rl_agent_with_code_feedback import main as code_feedback_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL training for Code Feedback agent")
    
    parser.add_argument('--sft_checkpoint', type=str,
                        default='checkpoints/sft_lora_final',
                        help='SFT checkpoint to initialize from')
    parser.add_argument('--config', type=str,
                        default='configs/rl_grpo_config.yaml',
                        help='GRPO config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                        help='Learning rate')
    parser.add_argument('--kl_coef', type=float, default=0.05,
                        help='KL coefficient')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs')
    
    args = parser.parse_args()
    
    # Build sys.argv for the full implementation
    sys.argv = [
        'train_rl_agent_with_code_feedback.py',
        '--config', args.config,
        '--model_path', args.sft_checkpoint,
        '--num_epochs', str(args.epochs),
        '--learning_rate', str(args.learning_rate),
        '--kl_coef', str(args.kl_coef),
    ]
    
    if args.output:
        sys.argv.extend(['--output_dir', args.output])
    
    # Set GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # Run main
    code_feedback_main()

