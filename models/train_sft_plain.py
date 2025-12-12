"""
Supervised Fine-Tuning (SFT) script without chat templates.
Prompts are formatted as plain text instead of using tokenizer.apply_chat_template.
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
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------
def load_eval_dataset(dataset_name: str, max_samples: int = 500):
    """Load evaluation dataset"""
    base_path = Path(__file__).parent.parent
    dataset_path = base_path / "data" / dataset_name

    if not dataset_path.exists():
        logger.warning(f"Dataset {dataset_name} not found at {dataset_path}, skipping evaluation")
        return None

    try:
        dataset = load_from_disk(str(dataset_path))
        # Use test split if available
        if "test" in dataset:
            eval_data = dataset["test"]
        elif hasattr(dataset, "keys") and len(dataset.keys()) > 0:
            split_name = list(dataset.keys())[0]
            eval_data = dataset[split_name]
            logger.info(f"Using split '{split_name}' for {dataset_name}")
        else:
            eval_data = dataset

        # Limit to max_samples
        if len(eval_data) > max_samples:
            eval_data = eval_data.select(range(max_samples))

        logger.info(f"Loaded {len(eval_data)} samples from {dataset_name}")
        return eval_data
    except Exception as e:
        logger.warning(f"Failed to load {dataset_name}: {e}")
        return None


def extract_ground_truth(example, dataset_name: str):
    """Extract ground truth answer from dataset example"""
    if dataset_name == "gsm8k":
        if "answer" in example:
            return example["answer"].split("####")[-1].strip()
    elif dataset_name in ["math", "math500"]:
        if "answer" in example:
            return example["answer"].strip()
        if "solution" in example:
            solution = example["solution"]
            idx = solution.find("\\boxed{")
            if idx != -1:
                start = idx + len("\\boxed{")
                depth = 1
                end = start
                while end < len(solution) and depth > 0:
                    if solution[end] == "{":
                        depth += 1
                    elif solution[end] == "}":
                        depth -= 1
                    end += 1
                if depth == 0:
                    return solution[start : end - 1].strip()
    return None


def build_plain_prompt(question: str, dataset_name: str) -> str:
    """Build plain-text prompt without chat template using standard formatting."""
    return format_prompt_standard(question, dataset_name)


def evaluate_model_on_dataset(model, tokenizer, dataset, dataset_name: str, device, log_dir: Path = None):
    """Evaluate model on a dataset using plain-text prompts."""
    if dataset is None:
        return None

    model.eval()
    correct = 0
    total = 0

    import random

    sample_indices = set(random.sample(range(len(dataset)), min(10, len(dataset))))
    sample_logs = []

    from tqdm import tqdm

    progress_bar = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc=f"Evaluating {dataset_name}",
        unit="sample",
    )

    for idx, example in progress_bar:
        if dataset_name == "gsm8k":
            question = example.get("question", "")
        elif dataset_name in ["math", "math500"]:
            question = example.get("problem", "")
        else:
            continue

        ground_truth = extract_ground_truth(example, dataset_name)
        if not ground_truth:
            continue

        prompt = build_plain_prompt(question, dataset_name)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        predicted = extract_answer(response)
        is_correct = check_answer(predicted, ground_truth)

        if is_correct:
            correct += 1
        total += 1

        if idx in sample_indices:
            sample_logs.append(
                {
                    "index": idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "response": response,
                    "correct": is_correct,
                }
            )

        current_accuracy = correct / total if total > 0 else 0.0
        progress_bar.set_postfix({"Accuracy": f"{current_accuracy*100:.1f}%", "Correct": f"{correct}/{total}"})

    progress_bar.close()

    if log_dir and sample_logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_log_file = log_dir / f"eval_samples_{dataset_name}_{timestamp}.txt"
        with open(sample_log_file, "w", encoding="utf-8") as f:
            accuracy = correct / total if total > 0 else 0.0
            f.write("=" * 80 + "\n")
            f.write(f"Evaluation Sample Log - {dataset_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write("=" * 80 + "\n\n")

            for i, sample in enumerate(sample_logs, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"Sample {i}/{len(sample_logs)} (Index: {sample['index']})\n")
                f.write(f"{'='*80}\n")
                f.write(f"Question:\n{sample['question']}\n\n")
                f.write(f"Ground Truth: {sample['ground_truth']}\n")
                f.write(f"Predicted: {sample['predicted']}\n")
                f.write(f"Correct: {'✓' if sample['correct'] else '✗'}\n\n")
                f.write(f"Full Response:\n{'-'*80}\n{sample['response']}\n{'-'*80}\n")

        logger.info(f"Saved {len(sample_logs)} sample logs to {sample_log_file}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


class TrainingMetricsCallbackPlain:
    """Callback to log training metrics to CSV file with plain-text evaluation."""

    def __init__(self, csv_path: Path, eval_datasets: dict = None, eval_every_n_epochs: int = 1, use_wandb: bool = False):
        from transformers import TrainerCallback

        class _InnerCallback(TrainerCallback):
            def __init__(self, outer):
                self.outer = outer

            def on_log(self, args, state, control, logs=None, **kwargs):
                return outer_on_log(args, state, control, logs, **kwargs)

            def on_epoch_end(self, args, state, control, **kwargs):
                return outer_on_epoch_end(args, state, control, **kwargs)

            def on_train_end(self, args, state, control, **kwargs):
                return outer_on_train_end(args, state, control, **kwargs)

        self._inner_class = _InnerCallback
        self.csv_path = csv_path
        self.metrics_history = []
        self.eval_datasets = eval_datasets or {}
        self.eval_every_n_epochs = eval_every_n_epochs
        self.last_eval_epoch = -1
        self.use_wandb = use_wandb
        self.tokenizer = None

        def outer_on_log(args, state, control, logs=None, **kwargs):
            if logs:
                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step": state.global_step,
                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                }
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        log_entry[key] = value
                self.metrics_history.append(log_entry)
                df = pd.DataFrame([log_entry])
                if not self.csv_path.exists():
                    df.to_csv(self.csv_path, index=False, mode="w")
                else:
                    df.to_csv(self.csv_path, index=False, mode="a", header=False)

        def outer_on_epoch_end(args, state, control, **kwargs):
            logger.info(f"Epoch {state.epoch} completed at step {state.global_step}")
            current_epoch = int(state.epoch)
            if (
                self.eval_datasets
                and current_epoch > self.last_eval_epoch
                and current_epoch % self.eval_every_n_epochs == 0
            ):
                self.last_eval_epoch = current_epoch
                logger.info(f"\n{'='*80}")
                logger.info(f"Running evaluation at epoch {current_epoch}")
                logger.info(f"{'='*80}")

                model = kwargs.get("model")
                tokenizer = self.tokenizer

                if model and tokenizer:
                    device = next(model.parameters()).device
                    eval_results = {}

                    for dataset_name, dataset in self.eval_datasets.items():
                        logger.info(f"Evaluating on {dataset_name}...")
                        accuracy = evaluate_model_on_dataset(
                            model,
                            tokenizer,
                            dataset,
                            dataset_name,
                            device,
                            log_dir=self.csv_path.parent,
                        )
                        if accuracy is not None:
                            eval_results[f"eval_{dataset_name}_accuracy"] = accuracy
                            logger.info(f"  {dataset_name} accuracy: {accuracy:.4f}")

                    if eval_results:
                        log_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "step": state.global_step,
                            "epoch": round(state.epoch, 2),
                            **eval_results,
                        }
                        self.metrics_history.append(log_entry)
                        df = pd.DataFrame([log_entry])
                        if not self.csv_path.exists():
                            df.to_csv(self.csv_path, index=False, mode="w")
                        else:
                            df.to_csv(self.csv_path, index=False, mode="a", header=False)

                        if self.use_wandb:
                            try:
                                import wandb

                                wandb.log({**eval_results, "epoch": current_epoch}, step=state.global_step)
                                logger.info("Logged evaluation results to W&B")
                            except Exception as e:
                                logger.warning(f"Failed to log to W&B: {e}")

                logger.info(f"{'='*80}\n")

        def outer_on_train_end(args, state, control, **kwargs):
            logger.info(f"Training metrics saved to: {self.csv_path}")
            if self.metrics_history:
                df = pd.DataFrame(self.metrics_history)
                summary_path = self.csv_path.parent / "metrics_summary.txt"
                with open(summary_path, "w") as f:
                    f.write("=" * 80 + "\n")
                    f.write("Training Metrics Summary\n")
                    f.write("=" * 80 + "\n\n")
                    if "loss" in df.columns:
                        f.write(f"Final Training Loss: {df['loss'].iloc[-1]:.6f}\n")
                        f.write(f"Min Training Loss: {df['loss'].min():.6f}\n")
                        f.write(f"Avg Training Loss: {df['loss'].mean():.6f}\n\n")
                    if "learning_rate" in df.columns:
                        f.write(f"Final Learning Rate: {df['learning_rate'].iloc[-1]:.2e}\n\n")
                    f.write(f"Total Steps: {df['step'].max()}\n")
                    f.write(f"Total Epochs: {df['epoch'].max():.2f}\n\n")
                    eval_cols = [col for col in df.columns if col.startswith("eval_")]
                    if eval_cols:
                        f.write("=" * 80 + "\n")
                        f.write("Evaluation Results\n")
                        f.write("=" * 80 + "\n\n")
                        for col in eval_cols:
                            dataset_name = col.replace("eval_", "").replace("_accuracy", "")
                            if col in df.columns:
                                eval_df = df[df[col].notna()]
                                if len(eval_df) > 0:
                                    f.write(f"{dataset_name.upper()}:\n")
                                    f.write(f"  Final Accuracy: {eval_df[col].iloc[-1]:.4f}\n")
                                    f.write(f"  Best Accuracy: {eval_df[col].max():.4f}\n")
                                    f.write(f"  All Results: {list(eval_df[col].values)}\n\n")
                    f.write("=" * 80 + "\n")
                logger.info(f"Metrics summary saved to: {summary_path}")

        self._on_log = outer_on_log
        self._on_epoch_end = outer_on_epoch_end
        self._on_train_end = outer_on_train_end

    def __call__(self):
        return self._inner_class(self)


def formatting_prompts_plain(example, tokenizer):
    """Format training examples as plain text (no chat template)."""
    prompt = build_plain_prompt(example["question"], "train")
    return prompt + "\n" + example["raw_output"]


def load_data(data_path: str) -> Dataset:
    """Load and prepare training data."""
    logger.info(f"Loading data from: {data_path}")
    path_obj = Path(data_path)
    suffix = path_obj.suffix.lower()

    # Support both JSON array files (.json) and line-delimited JSON (.jsonl)
    with open(data_path, "r") as f:
        if suffix == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    df = pd.DataFrame(data)
    if "correct" in df.columns:
        df = df[df["correct"] == True].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} correct samples")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Sample columns: {df.columns.tolist()}")
    if len(df) > 0:
        logger.info(f"Sample question: {df['question'].iloc[0][:100]}...")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=42)
    logger.info(f"Training dataset size: {len(dataset)}")
    return dataset


def setup_model_and_tokenizer(model_name: str, use_lora: bool = False, lora_rank: int = 16):
    """Load model and tokenizer with optional LoRA configuration."""
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
        )
    return model, tokenizer, peft_config


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    """Main training function."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = f"lora_r{args.lora_rank}" if args.mode == "lora" else "sft"
    dir_name = f"{args.round_name}_{mode_str}_{timestamp}" if args.round_name else f"{mode_str}_{timestamp}"
    output_dir = Path("checkpoints") / dir_name
    log_dir = Path("logs") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    if args.round_name:
        logger.info(f"Round Name: {args.round_name}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Logs: {log_dir}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"Gradient checkpointing: {args.gradient_checkpointing}")
    logger.info(f"Enable evaluation: {args.enable_eval}")
    if args.enable_eval:
        logger.info(f"Evaluation frequency: every {args.eval_every_n_epochs} epoch(s)")
    if args.mode == "lora":
        logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info("=" * 80)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        run_name = args.wandb_run_name
    elif args.round_name:
        run_name = f"{args.round_name}_{mode_str}_{timestamp}"
    else:
        run_name = f"{mode_str}_{timestamp}"

    train_dataset = load_data(args.data_path)

    use_lora = args.mode == "lora"
    model, tokenizer, peft_config = setup_model_and_tokenizer(
        args.model_name, use_lora=use_lora, lora_rank=args.lora_rank
    )

    def format_func(example):
        return formatting_prompts_plain(example, tokenizer)

    num_gpus = len(args.gpus.split(","))
    total_batch_size = args.batch_size * args.gradient_accumulation_steps * num_gpus
    steps_per_epoch = len(train_dataset) // total_batch_size
    if len(train_dataset) % total_batch_size != 0:
        steps_per_epoch += 1
    save_steps = steps_per_epoch * args.save_every_n_epochs

    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Save checkpoint every {save_steps} steps ({args.save_every_n_epochs} epochs)")

    eval_datasets = {}
    if args.enable_eval:
        logger.info(f"\nLoading evaluation datasets...")
        logger.info(f"Evaluation will run every {args.eval_every_n_epochs} epoch(s)")
        gsm8k_eval = load_eval_dataset("gsm8k", max_samples=args.eval_samples)
        if gsm8k_eval:
            eval_datasets["gsm8k"] = gsm8k_eval
        math500_eval = load_eval_dataset("math500", max_samples=args.eval_samples)
        if math500_eval:
            eval_datasets["math500"] = math500_eval
        if eval_datasets:
            logger.info(f"Loaded evaluation datasets: {list(eval_datasets.keys())}")
        else:
            logger.warning("No evaluation datasets loaded")

    csv_path = log_dir / "training_metrics.csv"
    metrics_callback_wrapper = TrainingMetricsCallbackPlain(
        csv_path,
        eval_datasets=eval_datasets if args.enable_eval else None,
        eval_every_n_epochs=args.eval_every_n_epochs,
        use_wandb=args.use_wandb,
    )
    metrics_callback = metrics_callback_wrapper()
    metrics_callback.tokenizer = tokenizer
    logger.info(f"Training metrics will be saved to: {csv_path}")

    lr = args.learning_rate if args.learning_rate is not None else (1e-5 if args.mode == "sft" else 1e-4)
    logger.info(f"Using learning rate: {lr}")

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
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=format_func,
        args=training_args,
        peft_config=peft_config,
        callbacks=[metrics_callback],
    )

    logger.info("Starting training...")
    train_result = trainer.train()

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

    metrics = train_result.metrics
    metrics_file = log_dir / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training completed!")
    if not args.skip_save:
        logger.info(f"Final model saved to: {output_dir / 'final_model'}")
    logger.info(f"Training metrics: {metrics}")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training without chat template")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sft", "lora"],
        help="Training mode: full SFT or LoRA",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pretrained_models/Qwen2.5-Math-1.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json",
        help="Path to training data JSON file",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (only used in lora mode)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU IDs to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: 1e-5 for SFT, 1e-4 for LoRA)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=2,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--enable_eval",
        action="store_true",
        help="Enable evaluation during training on GSM8K and MATH500",
    )
    parser.add_argument(
        "--eval_every_n_epochs",
        type=int,
        default=1,
        help="Run evaluation every N epochs (default: 1)",
    )
    parser.add_argument(
        "--round_name",
        type=str,
        default=None,
        help="Custom name for this training round (will be included in output directory name)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="slm_math_sft_plain",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=200,
        help="Number of samples per dataset for evaluation (default: 200)",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="Skip saving the final model (eval only mode)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.use_wandb:
        try:
            import wandb

            print("\n" + "=" * 80)
            print("Weights & Biases (W&B) Login Required")
            print("=" * 80)
            try:
                api = wandb.Api()
                if not api.api_key:
                    raise ValueError("Not logged in")
            except Exception:
                print("\nPlease log in to Weights & Biases")
                print("You can find your API key at: https://wandb.ai/authorize\n")
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

