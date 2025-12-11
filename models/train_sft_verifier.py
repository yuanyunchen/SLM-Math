"""
Supervised Fine-Tuning for Verifier Model
Trains a model to verify mathematical solutions and provide verdicts (CORRECT/INCORRECT/UNCLEAR)

Usage:
    python models/train_sft_verifier.py --mode lora --lora_rank 16 --gpus 0
    python models/train_sft_verifier.py --mode sft --gpus 0,1
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Verifier system prompt
VERIFIER_SYSTEM_PROMPT = (
    "You are a mathematical solution verifier. "
    "Your task is to carefully check if a given solution to a math problem is correct. "
    "Analyze the reasoning steps, calculations, and final answer. "
    "Provide one of three verdicts: CORRECT, INCORRECT, or UNCLEAR. "
    "If INCORRECT or UNCLEAR, explain what is wrong or unclear."
)


def load_verifier_data(data_path: str) -> Dataset:
    """Load verifier training data from JSONL file"""
    logger.info(f"Loading verifier data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read JSONL file
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    
    logger.info(f"Loaded {len(data_list)} verifier training samples")
    
    # Convert to dataset
    dataset = Dataset.from_list(data_list)
    
    # Log statistics
    if 'category' in dataset.column_names:
        categories = pd.Series([item['category'] for item in data_list])
        logger.info(f"Category distribution:\n{categories.value_counts()}")
    
    if 'verdict' in dataset.column_names:
        verdicts = pd.Series([item['verdict'] for item in data_list])
        logger.info(f"Verdict distribution:\n{verdicts.value_counts()}")
    
    return dataset


def format_verifier_prompt(example: Dict) -> str:
    """Format verifier training example as prompt"""
    # Input contains: Question + Solution
    input_text = example['input']
    
    # Output is the verdict (CORRECT/INCORRECT/UNCLEAR)
    output_text = example['output']
    
    # Format as conversation
    prompt = f"""<|im_start|>system
{VERIFIER_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""
    
    return prompt


def formatting_func(examples: Dict) -> List[str]:
    """Format batch of examples for training"""
    return [format_verifier_prompt(ex) for ex in examples]


class VerifierMetricsCallback(TrainerCallback):
    """Callback to log verifier-specific metrics"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, 'training_metrics.csv')
        self.metrics = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics_entry = {
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            }
            self.metrics.append(metrics_entry)
            
            # Save metrics periodically
            if state.global_step % 50 == 0:
                df = pd.DataFrame(self.metrics)
                df.to_csv(self.metrics_file, index=False)
                logger.info(f"Metrics saved to {self.metrics_file}")


def train_verifier(
    model_path: str,
    data_path: str,
    output_dir: str,
    mode: str = "lora",
    lora_rank: int = 16,
    num_epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    max_seq_length: int = 2048,
    gradient_checkpointing: bool = True,
    gpus: str = "0",
):
    """Train verifier model"""
    
    # Set GPU devices
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device_count = len(gpus.split(','))
    logger.info(f"Using {device_count} GPU(s): {gpus}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    dataset = load_verifier_data(data_path)
    
    # Split into train/eval (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto' if device_count == 1 else None,
    )
    
    # Configure LoRA if needed
    if mode == "lora":
        logger.info(f"Configuring LoRA with rank={lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        logger.info("Training in full SFT mode (all parameters)")
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Precision
        bf16=True,
        
        # Logging
        logging_dir=log_dir,
        logging_steps=10,
        logging_strategy="steps",
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Other
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=42,
        
        # SFT specific
        max_seq_length=max_seq_length,
        packing=False,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        formatting_func=lambda examples: [format_verifier_prompt(ex) for ex in examples],
        callbacks=[VerifierMetricsCallback(output_dir)],
    )
    
    # Train
    logger.info("=" * 80)
    logger.info("Starting verifier training...")
    logger.info("=" * 80)
    
    train_result = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    # Save training metrics
    metrics_summary = {
        'train_runtime': train_result.metrics.get('train_runtime'),
        'train_samples_per_second': train_result.metrics.get('train_samples_per_second'),
        'train_loss': train_result.metrics.get('train_loss'),
        'epoch': train_result.metrics.get('epoch'),
    }
    
    metrics_file = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Training metrics: {metrics_summary}")
    logger.info("=" * 80)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train verifier model for mathematical solution verification")
    
    # Model and data
    parser.add_argument('--model', type=str, default='pretrained_models/Qwen2.5-Math-1.5B',
                        help='Path to base model')
    parser.add_argument('--data', type=str, 
                        default='data/sft_verifier_training/sft_verifier_training_formatted.jsonl',
                        help='Path to verifier training data (JSONL format)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for checkpoints')
    
    # Training mode
    parser.add_argument('--mode', type=str, choices=['sft', 'lora'], default='lora',
                        help='Training mode: sft (full) or lora')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank (only used if mode=lora)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-device batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length')
    
    # System
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU IDs (e.g., "0,1,2")')
    parser.add_argument('--no_gradient_checkpointing', action='store_true',
                        help='Disable gradient checkpointing')
    
    args = parser.parse_args()
    
    # Generate output directory name if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"checkpoints/verifier_{args.mode}_r{args.lora_rank}_{timestamp}"
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("Verifier Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Mode: {args.mode}")
    if args.mode == 'lora':
        logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation * len(args.gpus.split(','))}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info("=" * 80)
    
    # Train
    train_verifier(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        mode=args.mode,
        lora_rank=args.lora_rank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()

