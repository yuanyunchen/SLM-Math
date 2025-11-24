"""
Supervised Fine-Tuning (SFT) baseline training script
Based on train_sft.ipynb with multi-GPU support and configurable LoRA

Usage:
    python models/train_sft_baseline.py --mode sft --gpus 0,1
    python models/train_sft_baseline.py --mode lora --lora_rank 16 --gpus 0,1
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# System prompt and formatting function
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve problems using step-by-step reasoning with clear explanations. "
    "Always provide your final answer in \\boxed{} format."
)


class TrainingMetricsCallback(TrainerCallback):
    """Callback to log training metrics to CSV file."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.metrics_history = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs:
            # Record timestamp and step
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'step': state.global_step,
                'epoch': round(state.epoch, 2) if state.epoch else 0,
            }
            
            # Add all logged metrics
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    log_entry[key] = value
            
            self.metrics_history.append(log_entry)
            
            # Save to CSV (append mode)
            df = pd.DataFrame([log_entry])
            if not self.csv_path.exists():
                df.to_csv(self.csv_path, index=False, mode='w')
            else:
                df.to_csv(self.csv_path, index=False, mode='a', header=False)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        logger.info(f"Epoch {state.epoch} completed at step {state.global_step}")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        logger.info(f"Training metrics saved to: {self.csv_path}")
        
        # Also save a summary
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            summary_path = self.csv_path.parent / "metrics_summary.txt"
            
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("Training Metrics Summary\n")
                f.write("=" * 80 + "\n\n")
                
                # Overall statistics
                if 'loss' in df.columns:
                    f.write(f"Final Training Loss: {df['loss'].iloc[-1]:.6f}\n")
                    f.write(f"Min Training Loss: {df['loss'].min():.6f}\n")
                    f.write(f"Avg Training Loss: {df['loss'].mean():.6f}\n\n")
                
                if 'learning_rate' in df.columns:
                    f.write(f"Final Learning Rate: {df['learning_rate'].iloc[-1]:.2e}\n\n")
                
                f.write(f"Total Steps: {df['step'].max()}\n")
                f.write(f"Total Epochs: {df['epoch'].max():.2f}\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Metrics summary saved to: {summary_path}")


def formatting_prompts(example, tokenizer):
    """Format training examples into chat template."""
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                example["question"]
                + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            ),
        },
        {
            "role": "assistant",
            "content": example["raw_output"],
        },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def load_data(data_path: str) -> Dataset:
    """Load and prepare training data."""
    logger.info(f"Loading data from: {data_path}")
    
    # Load JSON data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter for correct answers only
    if "correct" in df.columns:
        df = df[df["correct"] == True].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} correct samples")
    
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Sample columns: {df.columns.tolist()}")
    
    # Show sample
    if len(df) > 0:
        logger.info(f"Sample question: {df['question'].iloc[0][:100]}...")
    
    # Convert to HF Dataset (no train-val split as requested)
    dataset = Dataset.from_pandas(df)
    
    # Shuffle data
    dataset = dataset.shuffle(seed=42)
    
    logger.info(f"Training dataset size: {len(dataset)}")
    
    return dataset


def setup_model_and_tokenizer(model_name: str, use_lora: bool = False, lora_rank: int = 16):
    """Load model and tokenizer with optional LoRA configuration."""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Model loaded on: {model.device}")
    
    # Setup LoRA if requested
    peft_config = None
    if use_lora:
        logger.info(f"Configuring LoRA with rank={lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none"
        )
    
    return model, tokenizer, peft_config


def train(args):
    """Main training function."""
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = f"lora_r{args.lora_rank}" if args.mode == "lora" else "sft"
    output_dir = Path("checkpoints") / f"{mode_str}_{timestamp}"
    log_dir = Path("logs") / f"{mode_str}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Logs: {log_dir}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    if args.mode == "lora":
        logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info("=" * 80)
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # Setup wandb
    os.environ["WANDB_PROJECT"] = "slm_math_sft"
    run_name = f"{mode_str}_{timestamp}"
    
    # Load data
    train_dataset = load_data(args.data_path)
    
    # Load model and tokenizer
    use_lora = (args.mode == "lora")
    model, tokenizer, peft_config = setup_model_and_tokenizer(
        args.model_name,
        use_lora=use_lora,
        lora_rank=args.lora_rank
    )
    
    # Create formatting function closure
    def format_func(example):
        return formatting_prompts(example, tokenizer)
    
    # Calculate steps for saving checkpoints
    num_gpus = len(args.gpus.split(','))
    total_batch_size = args.batch_size * args.gradient_accumulation_steps * num_gpus
    steps_per_epoch = len(train_dataset) // total_batch_size
    if len(train_dataset) % total_batch_size != 0:
        steps_per_epoch += 1
    save_steps = steps_per_epoch * args.save_every_n_epochs
    
    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Save checkpoint every {save_steps} steps ({args.save_every_n_epochs} epochs)")
    
    # Setup CSV logging callback
    csv_path = log_dir / "training_metrics.csv"
    metrics_callback = TrainingMetricsCallback(csv_path)
    logger.info(f"Training metrics will be saved to: {csv_path}")
    
    # Training arguments based on notebook
    if args.mode == "sft":
        # Full SFT settings from notebook
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=5e-6,
            warmup_ratio=0.03,
            logging_steps=20,
            logging_first_step=True,
            # Save checkpoint every 2 epochs
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=None,  # Keep all checkpoints
            bf16=True,
            gradient_checkpointing=False,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            report_to=["wandb"] if args.use_wandb else [],
            run_name=run_name,
            dataloader_num_workers=4,
            # Multi-GPU settings
            ddp_find_unused_parameters=False,
        )
    else:  # lora
        # LoRA settings from notebook
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=1e-4,
            warmup_ratio=0.03,
            logging_steps=20,
            logging_first_step=True,
            # Save checkpoint every 2 epochs
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=None,  # Keep all checkpoints
            bf16=True,
            gradient_checkpointing=False,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            report_to=["wandb"] if args.use_wandb else [],
            run_name=run_name,
            dataloader_num_workers=4,
            # Multi-GPU settings
            ddp_find_unused_parameters=False,
        )
    
    # Create trainer with metrics callback
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=format_func,
        args=training_args,
        peft_config=peft_config,
        callbacks=[metrics_callback],
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    final_dir = output_dir / "final_model"
    logger.info(f"Saving final model to: {final_dir}")
    
    if args.mode == "lora":
        trainer.model.save_pretrained(final_dir)
    else:
        trainer.save_model(final_dir)
    
    tokenizer.save_pretrained(final_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    metrics_file = log_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Final model saved to: {final_dir}")
    logger.info(f"Training metrics: {metrics}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SFT Baseline Training")
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sft", "lora"],
        help="Training mode: full SFT or LoRA"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pretrained_models/Qwen2.5-Math-1.5B",
        help="Model name or path"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json",
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (only used in lora mode)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU IDs to use"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=2,
        help="Save checkpoint every N epochs"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Setup W&B login at the very beginning if requested
    if args.use_wandb:
        try:
            import wandb
            print("\n" + "=" * 80)
            print("Weights & Biases (W&B) Login Required")
            print("=" * 80)
            
            # Try to check if already logged in
            try:
                api = wandb.Api()
                if api.api_key:
                    print(f"✓ Already logged in to W&B as: {api.default_entity if hasattr(api, 'default_entity') else 'user'}")
                else:
                    raise ValueError("Not logged in")
            except:
                # Not logged in, prompt for login
                print("\nPlease log in to Weights & Biases")
                print("You can find your API key at: https://wandb.ai/authorize")
                print("\nOption 1: Enter your API key when prompted")
                print("Option 2: Set WANDB_API_KEY environment variable")
                print("Option 3: Run 'wandb login' before running this script\n")
                
                # Try login
                if not wandb.login(relogin=True):
                    raise Exception("W&B login failed")
            
            print("✓ W&B login successful!")
            print("=" * 80 + "\n")
            
        except ImportError:
            print("WARNING: wandb is not installed. Installing now...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb"])
            import wandb
            print("\nPlease log in to Weights & Biases")
            print("You can find your API key at: https://wandb.ai/authorize\n")
            if not wandb.login():
                raise Exception("W&B login failed")
                
        except Exception as e:
            print(f"\n✗ ERROR: Failed to setup W&B: {e}")
            print("Please choose one of the following:")
            print("  1. Run 'wandb login' in terminal before running this script")
            print("  2. Set WANDB_API_KEY environment variable")
            print("  3. Disable W&B by removing --use_wandb flag\n")
            sys.exit(1)
    
    train(args)

