"""
Agent SFT Training Script for Interactive Code Execution

Trains models on:
1. rstar_100k_clean.csv - Pure correct reasoning with code execution
2. correction_gsm8k_*.json - Self-correction data (error -> reflection -> correction)
3. correction_math_*.json - Self-correction data (error -> reflection -> correction)

Key Design Decisions:
- Train on FULL sequences including error + correction process
- This teaches the model to: identify errors, reflect, and self-correct
- All correction samples are "success" cases (final answer is correct)

Usage:
    python models/train_agent_sft.py --mode sft --data_mix all --gpus 0
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import random

import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_utils import extract_answer, check_answer

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# System prompt for interactive code execution
SYSTEM_PROMPT_AGENT = (
    "You are a mathematical reasoning assistant that uses Python code to solve problems. "
    "Write code to help with calculations. When you see an error or unexpected result, "
    "analyze what went wrong, explain your reasoning, and correct the code. "
    "Always provide your final answer in \\boxed{} format."
)


def load_rstar_data(data_path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load rstar clean data from CSV."""
    logger.info(f"Loading rstar data from: {data_path}")
    
    if not data_path.exists():
        logger.warning(f"Rstar data not found: {data_path}")
        return []
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from rstar")
    
    # Convert to list of dicts with standard format
    samples = []
    for _, row in df.iterrows():
        samples.append({
            "query": row["query"],
            "response": row["response"],
            "source": "rstar",
            "error_type": None  # No error in rstar data
        })
    
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
        logger.info(f"Sampled {max_samples} from rstar data")
    
    return samples


def load_correction_data(data_path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load correction data from JSON."""
    logger.info(f"Loading correction data from: {data_path}")
    
    if not data_path.exists():
        logger.warning(f"Correction data not found: {data_path}")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples from correction data")
    
    # Filter to only success cases (should already be filtered, but double-check)
    samples = []
    for item in data:
        if item.get("status") == "success":
            samples.append({
                "query": item["query"],
                "response": item["response"],
                "source": item.get("source", "correction"),
                "error_type": item.get("error_type", "unknown"),
                "ground_truth": item.get("ground_truth")
            })
    
    logger.info(f"Filtered to {len(samples)} success samples")
    
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
        logger.info(f"Sampled {max_samples} from correction data")
    
    return samples


def load_all_data(args) -> Dataset:
    """Load and combine all training data based on data_mix setting."""
    base_path = Path(__file__).parent.parent / "data"
    all_samples = []
    
    # Data paths
    rstar_path = base_path / "rstar_sft_interactive" / "rstar_100k_clean.csv"
    correction_gsm8k_path = base_path / "agent_sft" / "correction_gsm8k_1206_1448" / "correction_data_1206_1503.json"
    correction_math_path = base_path / "agent_sft" / "correction_math_1206_1447" / "correction_data_1206_1528.json"
    
    # Load based on data_mix setting
    if args.data_mix in ["all", "rstar"]:
        rstar_samples = load_rstar_data(rstar_path, args.max_rstar_samples)
        all_samples.extend(rstar_samples)
        logger.info(f"Added {len(rstar_samples)} rstar samples")
    
    if args.data_mix in ["all", "correction"]:
        # Use specific limits if provided, otherwise fall back to general limit
        gsm8k_limit = args.max_gsm8k_correction if args.max_gsm8k_correction is not None else args.max_correction_samples
        math_limit = args.max_math_correction if args.max_math_correction is not None else args.max_correction_samples
        
        gsm8k_samples = load_correction_data(correction_gsm8k_path, gsm8k_limit)
        all_samples.extend(gsm8k_samples)
        logger.info(f"Added {len(gsm8k_samples)} GSM8K correction samples (limit: {gsm8k_limit or 'all'})")
        
        math_samples = load_correction_data(correction_math_path, math_limit)
        all_samples.extend(math_samples)
        logger.info(f"Added {len(math_samples)} MATH correction samples (limit: {math_limit or 'all'})")
    
    if args.data_mix == "correction_only":
        gsm8k_limit = args.max_gsm8k_correction if args.max_gsm8k_correction is not None else args.max_correction_samples
        math_limit = args.max_math_correction if args.max_math_correction is not None else args.max_correction_samples
        
        gsm8k_samples = load_correction_data(correction_gsm8k_path, gsm8k_limit)
        all_samples.extend(gsm8k_samples)
        math_samples = load_correction_data(correction_math_path, math_limit)
        all_samples.extend(math_samples)
    
    # Log statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Data Loading Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(all_samples)}")
    
    # Count by source
    source_counts = {}
    error_type_counts = {}
    for sample in all_samples:
        source = sample.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        
        error_type = sample.get("error_type")
        if error_type:
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    
    logger.info(f"\nBy source:")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source}: {count}")
    
    if error_type_counts:
        logger.info(f"\nBy error type (correction data only):")
        for error_type, count in sorted(error_type_counts.items()):
            logger.info(f"  {error_type}: {count}")
    
    logger.info(f"{'='*60}\n")
    
    # Convert to HF Dataset
    df = pd.DataFrame(all_samples)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=42)
    
    return dataset


def format_prompt_plain(query: str) -> str:
    """Format prompt in plain text format (no chat template).
    
    This matches the format used in previous experiments and eval.
    """
    prompt = f"""{query}
Please reason step by step, and put your final answer within \\boxed{{}}.

You may use Python code to help with calculations. Show your reasoning step by step."""
    return prompt


def formatting_prompts_agent(examples, tokenizer, apply_chat_template: bool = False):
    """Format training examples for agent-style training.
    
    Args:
        examples: Training examples with 'query' and 'response' fields
        tokenizer: Tokenizer for chat template (if used)
        apply_chat_template: If True, use chat template format.
                           If False, use plain text format (default for consistency).
    
    Handles both single examples and batched examples.
    """
    # Handle batched input
    if isinstance(examples.get("query"), list):
        texts = []
        for query, response in zip(examples["query"], examples["response"]):
            if apply_chat_template:
                # Use chat template format
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_AGENT,
                    },
                    {
                        "role": "user",
                        "content": (
                            query
                            + "\nPlease solve this step by step using Python code. "
                            "Put your final answer within \\boxed{}."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": response,
                    },
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                # Use plain text format (consistent with previous experiments)
                prompt = format_prompt_plain(query)
                text = prompt + "\n\n" + response
            texts.append(text)
        return texts
    else:
        # Single example
        if apply_chat_template:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_AGENT,
                },
                {
                    "role": "user",
                    "content": (
                        examples["query"]
                        + "\nPlease solve this step by step using Python code. "
                        "Put your final answer within \\boxed{}."
                    ),
                },
                {
                    "role": "assistant",
                    "content": examples["response"],
                },
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Use plain text format (consistent with previous experiments)
            prompt = format_prompt_plain(examples["query"])
            text = prompt + "\n\n" + examples["response"]
        return text


class TrainingMetricsCallback(TrainerCallback):
    """Callback to log training metrics to CSV file."""
    
    def __init__(self, csv_path: Path, use_wandb: bool = False):
        self.csv_path = csv_path
        self.metrics_history = []
        self.use_wandb = use_wandb
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs:
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'step': state.global_step,
                'epoch': round(state.epoch, 2) if state.epoch else 0,
            }
            
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    log_entry[key] = value
            
            self.metrics_history.append(log_entry)
            
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
        
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            summary_path = self.csv_path.parent / "metrics_summary.txt"
            
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("Training Metrics Summary\n")
                f.write("=" * 80 + "\n\n")
                
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


def setup_model_and_tokenizer(model_name: str, use_lora: bool = False, lora_rank: int = 16):
    """Load model and tokenizer with optional LoRA configuration."""
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Model loaded on: {model.device}")
    
    peft_config = None
    if use_lora:
        logger.info(f"Configuring LoRA with rank={lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
    
    return model, tokenizer, peft_config


def train(args):
    """Main training function."""
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = f"lora_r{args.lora_rank}" if args.mode == "lora" else "sft"
    
    if args.round_name:
        dir_name = f"{args.round_name}_{mode_str}_{timestamp}"
    else:
        dir_name = f"agent_{mode_str}_{timestamp}"
    
    output_dir = Path("checkpoints") / dir_name
    log_dir = Path("logs") / dir_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("Agent SFT Training Configuration")
    logger.info("=" * 80)
    if args.round_name:
        logger.info(f"Round Name: {args.round_name}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data Mix: {args.data_mix}")
    logger.info(f"Apply Chat Template: {args.apply_chat_template}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Logs: {log_dir}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"Gradient checkpointing: {args.gradient_checkpointing}")
    if args.mode == "lora":
        logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info("=" * 80)
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # Load data
    train_dataset = load_all_data(args)
    
    # Load model and tokenizer
    use_lora = (args.mode == "lora")
    model, tokenizer, peft_config = setup_model_and_tokenizer(
        args.model_name,
        use_lora=use_lora,
        lora_rank=args.lora_rank
    )
    
    # Create formatting function closure
    def format_func(example):
        return formatting_prompts_agent(example, tokenizer, apply_chat_template=args.apply_chat_template)
    
    # Calculate steps for saving
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
    metrics_callback = TrainingMetricsCallback(csv_path, use_wandb=args.use_wandb)
    logger.info(f"Training metrics will be saved to: {csv_path}")
    
    # Determine learning rate
    if args.learning_rate is not None:
        lr = args.learning_rate
    else:
        lr = 1e-5 if args.mode == "sft" else 1e-4
    
    logger.info(f"Using learning rate: {lr}")
    
    # Setup wandb
    run_name = None
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        elif args.round_name:
            run_name = f"{args.round_name}_{mode_str}_{timestamp}"
        else:
            run_name = f"agent_{mode_str}_{timestamp}"
    
    # Training arguments - compatible with trl 0.9.x
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=lr,
        warmup_ratio=0.03,
        logging_steps=20,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to=["wandb"] if args.use_wandb else [],
        run_name=run_name,
        dataloader_num_workers=0,  # Avoid memory issues with forked processes
        ddp_find_unused_parameters=False,
    )
    
    # Create trainer - max_seq_length passed to SFTTrainer for trl 0.9.x compatibility
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=format_func,
        args=training_args,
        peft_config=peft_config,
        callbacks=[metrics_callback],
        max_seq_length=args.max_seq_length,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    if not args.skip_save:
        final_dir = output_dir / "final_model"
        logger.info(f"Saving final model to: {final_dir}")
        
        if args.mode == "lora":
            trainer.model.save_pretrained(final_dir)
        else:
            trainer.save_model(final_dir)
        
        tokenizer.save_pretrained(final_dir)
    else:
        logger.info("Skipping model save (--skip_save enabled)")
    
    # Save training config
    config_path = log_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Save training metrics
    metrics = train_result.metrics
    metrics_file = log_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed!")
    if not args.skip_save:
        logger.info(f"Final model saved to: {output_dir / 'final_model'}")
    logger.info(f"Training metrics: {metrics}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Agent SFT Training")
    
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
        "--data_mix",
        type=str,
        default="all",
        choices=["all", "rstar", "correction", "correction_only"],
        help="Data mix: all (rstar+correction), rstar only, correction only"
    )
    parser.add_argument(
        "--max_rstar_samples",
        type=int,
        default=None,
        help="Maximum number of rstar samples to use (default: all)"
    )
    parser.add_argument(
        "--max_correction_samples",
        type=int,
        default=None,
        help="Maximum correction samples per dataset (default: all, overridden by specific args)"
    )
    parser.add_argument(
        "--max_gsm8k_correction",
        type=int,
        default=None,
        help="Maximum GSM8K correction samples (overrides --max_correction_samples for GSM8K)"
    )
    parser.add_argument(
        "--max_math_correction",
        type=int,
        default=None,
        help="Maximum MATH correction samples (overrides --max_correction_samples for MATH)"
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
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: 1e-5 for SFT, 1e-4 for LoRA)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length for training"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--round_name",
        type=str,
        default=None,
        help="Custom name for this training round"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="slm_math_agent_sft",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="Skip saving the final model"
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Use chat template format (default: False, use plain text for consistency)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Setup W&B login if requested
    if args.use_wandb:
        try:
            import wandb
            try:
                api = wandb.Api()
                if api.api_key:
                    logger.info("Already logged in to W&B")
                else:
                    raise ValueError("Not logged in")
            except:
                if not wandb.login(relogin=True):
                    raise Exception("W&B login failed")
            logger.info("W&B login successful!")
        except ImportError:
            logger.warning("wandb not installed, disabling W&B logging")
            args.use_wandb = False
        except Exception as e:
            logger.error(f"Failed to setup W&B: {e}")
            logger.warning("Disabling W&B logging")
            args.use_wandb = False
    
    train(args)

