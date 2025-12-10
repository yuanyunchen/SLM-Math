"""
GRPO training tailored for the Solver-Verifier agent.

Key differences vs. the baseline RL trainer:
- Uses solver-verifier style prompting that explicitly allows Python code.
- Executes generated code and shapes reward with code consistency signals.
- Penalizes repeated answers across multiple samples of the same prompt.
- Handles GSM8K and MATH500 datasets directly via HuggingFace load_from_disk.
"""

import os
import sys
import math
import json
import yaml
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk

# Project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.prompt_utils import (
    extract_answer,
    check_answer,
    extract_question_and_answer,
    normalize_answer,
    format_prompt_standard,
)
from agent.solver_verifier import extract_code_and_execute

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


@dataclass
class SVRLConfig:
    """Solver-Verifier GRPO configuration."""

    # Model
    model_path: str
    trust_remote_code: bool = True

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Data
    datasets: List[str] = field(default_factory=lambda: ["gsm8k", "math"])
    max_samples: int = -1  # total across all datasets
    train_split_ratio: float = 0.95

    # GRPO
    num_return_sequences: int = 2
    reward_correct: float = 1.0
    reward_wrong: float = -1.0
    reward_no_answer: float = -0.5
    reward_code_error: float = -0.2
    reward_code_inconsistent: float = -0.2
    reward_code_consistent: float = 0.1
    reward_duplicate_answer: float = -0.1
    reward_missing_box: float = -0.1
    kl_coef: float = 0.0
    whiten_rewards: bool = True
    clip_range: float = 0.2

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    max_steps: int = -1
    output_dir: str = "results/rl_solver_verifier_checkpoints"
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    eval_samples: int = 0  # disabled by default (no in-training eval on train data)


def load_config_from_yaml(config_path: str) -> SVRLConfig:
    """Load configuration from YAML (reusing rl_grpo_config structure)."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def g(path, default=None):
        cur = cfg
        for p in path:
            if cur is None or p not in cur:
                return default
            cur = cur[p]
        return cur

    model_path = g(["model", "path"]) or g(["model", "name"])
    config = SVRLConfig(
        model_path=model_path,
        trust_remote_code=g(["model", "trust_remote_code"], True),
        use_lora=g(["lora", "enabled"], True),
        lora_r=g(["lora", "r"], 16),
        lora_alpha=g(["lora", "alpha"], 32),
        lora_dropout=g(["lora", "dropout"], 0.05),
        lora_target_modules=g(
            ["lora", "target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        datasets=g(["data", "datasets"], ["gsm8k", "math500"]),
        max_samples=g(["data", "max_samples"], -1),
        train_split_ratio=g(["data", "train_split_ratio"], 0.95),
        num_return_sequences=g(["grpo", "num_return_sequences"], 2),
        reward_correct=g(["grpo", "reward_correct"], 1.0),
        reward_wrong=g(["grpo", "reward_wrong"], -1.0),
        reward_no_answer=g(["grpo", "reward_no_answer"], -0.5),
        reward_code_error=g(["grpo", "reward_code_error"], -0.2),
        reward_code_inconsistent=g(["grpo", "reward_code_inconsistent"], -0.2),
        reward_code_consistent=g(["grpo", "reward_code_consistent"], 0.1),
        reward_duplicate_answer=g(["grpo", "reward_duplicate_answer"], -0.1),
        reward_missing_box=g(["grpo", "reward_missing_box"], -0.1),
        kl_coef=g(["grpo", "kl_coef"], 0.0),
        whiten_rewards=g(["grpo", "whiten_rewards"], True),
        clip_range=g(["grpo", "clip_range"], 0.2),
        max_new_tokens=g(["generation", "max_new_tokens"], 2048),
        temperature=g(["generation", "temperature"], 0.7),
        top_p=g(["generation", "top_p"], 0.9),
        top_k=g(["generation", "top_k"], 50),
        do_sample=g(["generation", "do_sample"], True),
        num_epochs=g(["training", "num_epochs"], 3),
        per_device_train_batch_size=g(
            ["training", "per_device_train_batch_size"], 4
        ),
        gradient_accumulation_steps=g(
            ["training", "gradient_accumulation_steps"], 4
        ),
        learning_rate=g(["training", "learning_rate"], 5e-6),
        max_grad_norm=g(["training", "max_grad_norm"], 1.0),
        warmup_ratio=g(["training", "warmup_ratio"], 0.1),
        logging_steps=g(["training", "logging_steps"], 10),
        eval_steps=g(["training", "eval_steps"], 200),
        save_steps=g(["training", "save_steps"], 200),
        max_steps=g(["training", "max_steps"], -1),
        output_dir=g(["training", "output_dir"], "results/rl_solver_verifier_checkpoints"),
        seed=g(["training", "seed"], 42),
        bf16=g(["training", "bf16"], True),
        gradient_checkpointing=g(["training", "gradient_checkpointing"], True),
        eval_samples=g(["training", "eval_samples"], 0),
    )

    # Normalize dataset names
    config.datasets = [d.strip() for d in config.datasets]
    return config


# =============================================================================
# Data utilities
# =============================================================================


def format_solver_verifier_prompt(question: str, dataset_name: str) -> str:
    """Standard prompt with explicit tool instruction."""
    base = format_prompt_standard(question, dataset_name)
    tool_hint = (
        "\n\nYou may use Python code inside ```python``` blocks to compute intermediate "
        "results. Execute code carefully and keep the final answer in \\boxed{}."
    )
    return base + tool_hint


def load_train_data(datasets: List[str], max_samples: int = -1) -> List[Dict]:
    """Load and merge train splits from provided datasets."""
    base_path = project_root / "data"
    data = []

    per_dataset_cap = None
    if max_samples and max_samples > 0:
        per_dataset_cap = math.ceil(max_samples / max(1, len(datasets)))

    for name in datasets:
        ds_path = base_path / name
        if not ds_path.exists():
            logger.warning(f"Dataset {name} not found at {ds_path}, skipping")
            continue

        try:
            ds = load_from_disk(str(ds_path))
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            continue

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        split = ds[split_name]

        if per_dataset_cap and len(split) > per_dataset_cap:
            split = split.select(range(per_dataset_cap))

        for ex in split:
            try:
                q, gt = extract_question_and_answer(ex, name)
                data.append({"question": q, "ground_truth": gt, "dataset": name})
            except Exception as e:
                logger.warning(f"Skip {name} sample due to parse error: {e}")

    if max_samples > 0 and len(data) > max_samples:
        data = data[:max_samples]

    logger.info(f"Loaded {len(data)} samples from {datasets}")
    return data


# =============================================================================
# Reward utilities
# =============================================================================


def parse_code_result(exec_results: List[Dict]) -> Optional[str]:
    """Extract numeric-looking result from the last successful code execution."""
    if not exec_results:
        return None
    last = exec_results[-1]
    if not last.get("success"):
        return None
    output = (last.get("output") or "").strip()
    if not output:
        return None
    last_line = output.split("\n")[-1].strip()
    nums = [n for n in re.findall(r"[-+]?\d*\.?\d+", last_line)]
    if nums:
        return nums[-1]
    return None


def compute_solver_verifier_reward(
    response: str, ground_truth: str, config: SVRLConfig
) -> Tuple[float, Dict]:
    """
    Reward with code-awareness and consistency checks.

    Returns:
        reward, info
    """
    # Extract final answer
    boxed_answer = extract_answer(response)

    # Run code (without mutating response)
    _, exec_results, used_code = extract_code_and_execute(response, detailed=False)
    code_result = parse_code_result(exec_results)

    main_answer = boxed_answer if boxed_answer else code_result

    reward = 0.0
    info = {
        "boxed_answer": boxed_answer,
        "code_result": code_result,
        "used_code": used_code,
        "has_code_error": any(not r.get("success") for r in exec_results)
        if exec_results
        else False,
        "final_answer": main_answer,
    }

    if main_answer is None:
        reward = config.reward_no_answer
        info["is_correct"] = False
        info["reason"] = "no_answer"
    else:
        is_correct = check_answer(main_answer, ground_truth)
        info["is_correct"] = is_correct
        reward = config.reward_correct if is_correct else config.reward_wrong
        if boxed_answer is None:
            reward += config.reward_missing_box
            info["reason"] = "no_boxed_answer"

    # Code execution signals
    if used_code:
        if info["has_code_error"]:
            reward += config.reward_code_error
        if code_result:
            if boxed_answer:
                if check_answer(code_result, boxed_answer):
                    reward += config.reward_code_consistent
                else:
                    reward += config.reward_code_inconsistent

    info["reward"] = reward
    return reward, info


# =============================================================================
# Trainer
# =============================================================================


class SVGRPOTrainer:
    """GRPO trainer with solver-verifier aware rewards."""

    def __init__(
        self,
        config: SVRLConfig,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer,
        train_data: List[Dict],
        eval_datasets: dict = None,
        use_wandb: bool = False,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.eval_datasets = eval_datasets or {}
        self.use_wandb = use_wandb

        self.device = next(model.parameters()).device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        self.total_steps = (
            len(train_data) // config.per_device_train_batch_size
        ) * config.num_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)

        self.global_step = 0
        self.epoch = 0

        self.metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)

        logger.info(
            f"Trainer initialized: {self.total_steps} steps, {self.warmup_steps} warmup steps"
        )

    # ----------------------------
    # Generation utilities
    # ----------------------------
    def get_lr_multiplier(self):
        if self.global_step < self.warmup_steps:
            return self.global_step / max(1, self.warmup_steps)
        progress = (self.global_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    def update_lr(self):
        lr_mult = self.get_lr_multiplier()
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate * lr_mult

    def generate_responses(self, prompts: List[str]) -> List[List[str]]:
        """Generate multiple responses per prompt."""
        self.model.eval()
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        batch_size = len(prompts)
        n_ret = self.config.num_return_sequences

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=n_ret,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        all_responses = []
        for i in range(batch_size):
            responses = []
            prompt_len = prompt_lengths[i]
            for j in range(n_ret):
                idx = i * n_ret + j
                gen_ids = outputs[idx][prompt_len:]
                responses.append(
                    self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                )
            all_responses.append(responses)
        return all_responses

    def compute_log_probs_batch(
        self, prompts: List[str], responses: List[str], model
    ) -> torch.Tensor:
        """Log probs of responses conditioned on prompts."""
        prompt_lengths = []
        for p in prompts:
            ids = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
            prompt_lengths.append(ids["input_ids"].shape[1])

        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        log_probs_list = []
        for i in range(len(prompts)):
            prompt_len = prompt_lengths[i]
            attn = inputs["attention_mask"][i]
            seq_len = attn.sum().item()

            resp_logits = logits[i, prompt_len - 1 : seq_len - 1, :]
            resp_ids = inputs["input_ids"][i, prompt_len:seq_len]

            if len(resp_ids) == 0:
                log_probs_list.append(torch.tensor(0.0, device=self.device))
                continue

            token_log_probs = F.log_softmax(resp_logits, dim=-1)
            selected = token_log_probs.gather(dim=-1, index=resp_ids.unsqueeze(-1)).squeeze(-1)
            log_probs_list.append(selected.mean())

        return torch.stack(log_probs_list)

    def compute_log_probs(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        return self.compute_log_probs_batch(prompts, responses, self.model)

    def compute_kl_penalty(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        policy_log_probs = self.compute_log_probs(prompts, responses)
        if self.ref_model is None:
            return torch.tensor(0.0, device=self.device)
        self.ref_model.eval()
        ref_log_probs = self.compute_log_probs_batch(prompts, responses, self.ref_model)
        return (policy_log_probs - ref_log_probs).mean()

    # ----------------------------
    # Training step
    # ----------------------------
    def train_step(self, batch: Dict, accumulate: bool = False) -> Dict[str, float]:
        questions = batch["question"]
        truths = batch["ground_truth"]
        datasets = batch["dataset"]

        prompts = [
            format_solver_verifier_prompt(q, ds) for q, ds in zip(questions, datasets)
        ]

        all_responses = self.generate_responses(prompts)

        flat_prompts, flat_responses, flat_truths = [], [], []
        flat_indices_per_prompt = []
        reward_infos = []
        rewards = []

        for i, (p, rs, truth) in enumerate(zip(prompts, all_responses, truths)):
            idxs = []
            for r in rs:
                idx = len(flat_prompts)
                idxs.append(idx)
                flat_prompts.append(p)
                flat_responses.append(r)
                flat_truths.append(truth)
            flat_indices_per_prompt.append(idxs)

        # Base rewards
        for r, truth in zip(flat_responses, flat_truths):
            rew, info = compute_solver_verifier_reward(r, truth, self.config)
            rewards.append(rew)
            reward_infos.append(info)

        # Duplicate answer penalty within each prompt's samples
        for idxs in flat_indices_per_prompt:
            answers = [
                normalize_answer(reward_infos[idx]["final_answer"])
                if reward_infos[idx]["final_answer"]
                else ""
                for idx in idxs
            ]
            counts = Counter(answers)
            for idx, ans in zip(idxs, answers):
                if ans and counts[ans] > 1:
                    rewards[idx] += self.config.reward_duplicate_answer
                    reward_infos[idx]["duplicate_penalty"] = True
                else:
                    reward_infos[idx]["duplicate_penalty"] = False

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        raw_reward_mean = rewards_tensor.mean().item()

        if self.config.whiten_rewards and len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (
                rewards_tensor.std() + 1e-8
            )

        # KL penalty
        kl_penalty = (
            self.compute_kl_penalty(flat_prompts, flat_responses)
            if self.config.kl_coef > 0
            else torch.tensor(0.0, device=self.device)
        )

        self.model.train()

        batch_size = len(flat_prompts)

        prompt_lengths = []
        for p in flat_prompts:
            ids = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
            prompt_lengths.append(ids["input_ids"].shape[1])

        full_texts = [p + r for p, r in zip(flat_prompts, flat_responses)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2560,
            padding=True,
        ).to(self.device)

        labels = inputs["input_ids"].clone()
        for i in range(batch_size):
            labels[i, : prompt_lengths[i]] = -100
            padding_mask = inputs["attention_mask"][i] == 0
            labels[i, padding_mask] = -100

        outputs = self.model(**inputs, labels=labels)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(batch_size, -1)

        valid_tokens = (shift_labels != -100).float()
        per_sample_loss = (per_token_loss * valid_tokens).sum(dim=1) / (
            valid_tokens.sum(dim=1) + 1e-8
        )

        weighted_loss = per_sample_loss * rewards_tensor
        avg_loss = weighted_loss.mean()
        final_loss = avg_loss + self.config.kl_coef * kl_penalty

        if accumulate:
            final_loss = final_loss / self.config.gradient_accumulation_steps

        final_loss.backward()

        if not accumulate:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        accuracy = sum(info["is_correct"] for info in reward_infos) / max(
            1, len(reward_infos)
        )
        code_error_rate = sum(info["has_code_error"] for info in reward_infos) / max(
            1, len(reward_infos)
        )

        metrics = {
            "loss": final_loss.item()
            * (self.config.gradient_accumulation_steps if accumulate else 1),
            "reward": rewards_tensor.mean().item(),  # whitened (≈0 if whitening on)
            "raw_reward": raw_reward_mean,           # pre-whiten mean for visibility
            "accuracy": accuracy,
            "kl_penalty": kl_penalty.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "code_error_rate": code_error_rate,
        }
        return metrics

    # ----------------------------
    # Evaluation
    # ----------------------------
    def run_evaluation(self):
        if not self.eval_datasets:
            return {}
        logger.info(f"\n{'='*80}")
        logger.info(f"Running evaluation at epoch {self.epoch + 1}")
        logger.info(f"{'='*80}")
        eval_results = {}
        log_dir = Path(self.config.output_dir)
        for ds_name, ds in self.eval_datasets.items():
            logger.info(f"Evaluating on {ds_name}...")
            acc = evaluate_solver_verifier_on_dataset(
                self.model,
                self.tokenizer,
                ds,
                ds_name,
                self.device,
                log_dir=log_dir,
                config=self.config,
            )
            if acc is not None:
                eval_results[f"eval_{ds_name}_accuracy"] = acc
                self.eval_metrics[ds_name].append(acc)
                logger.info(f"  {ds_name} accuracy: {acc:.4f}")
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({f"eval/{ds_name}_accuracy": acc, "epoch": self.epoch + 1}, step=self.global_step)
                    except ImportError:
                        pass
        logger.info(f"{'='*80}\n")
        return eval_results

    # ----------------------------
    # Train loop
    # ----------------------------
    def train(self):
        logger.info("=" * 80)
        logger.info("Starting Solver-Verifier GRPO Training")
        logger.info("=" * 80)

        grad_acc = self.config.gradient_accumulation_steps
        logger.info(f"Gradient accumulation steps: {grad_acc}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            import random

            random.shuffle(self.train_data)
            batch_size = self.config.per_device_train_batch_size
            num_batches = len(self.train_data) // batch_size
            epoch_metrics = defaultdict(list)
            accum_metrics = defaultdict(list)

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
            for batch_idx in progress_bar:
                start = batch_idx * batch_size
                end = start + batch_size
                batch_data = self.train_data[start:end]

                batch = {
                    "question": [d["question"] for d in batch_data],
                    "ground_truth": [d["ground_truth"] for d in batch_data],
                    "dataset": [d["dataset"] for d in batch_data],
                }

                self.update_lr()
                metrics = self.train_step(
                    batch,
                    accumulate=(
                        batch_idx % grad_acc != grad_acc - 1
                    ),
                )

                for k, v in metrics.items():
                    accum_metrics[k].append(v)

                if (batch_idx + 1) % grad_acc == 0 or batch_idx == num_batches - 1:
                    for k, v in accum_metrics.items():
                        avg_v = np.mean(v)
                        epoch_metrics[k].append(avg_v)
                        self.metrics[k].append(avg_v)
                    accum_metrics = defaultdict(list)
                    self.global_step += 1

                if (
                    self.global_step % self.config.logging_steps == 0
                    and len(epoch_metrics["loss"]) > 0
                ):
                    avg_metrics = {
                        k: np.mean(v[-self.config.logging_steps :])
                        for k, v in epoch_metrics.items()
                        if v
                    }
                    progress_bar.set_postfix(avg_metrics)
                    # Explicit console log for quick visibility (includes accuracy and other metrics)
                    acc_val = avg_metrics.get("accuracy")
                    log_parts = [
                        f"Step {self.global_step}",
                        f"loss={avg_metrics.get('loss', 0):.4f}",
                        f"reward={avg_metrics.get('reward', 0):.4f}",
                        f"raw_reward={avg_metrics.get('raw_reward', 0):.4f}",
                    ]
                    if acc_val is not None:
                        log_parts.append(f"accuracy={acc_val:.4f}")
                    if "kl_penalty" in avg_metrics:
                        log_parts.append(f"kl={avg_metrics.get('kl_penalty', 0):.4f}")
                    if "code_error_rate" in avg_metrics:
                        log_parts.append(f"code_err={avg_metrics.get('code_error_rate', 0):.4f}")
                    # current LR
                    try:
                        log_parts.append(f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
                    except Exception:
                        pass
                    logger.info(" ".join(log_parts))
                    if self.use_wandb:
                        try:
                            import wandb
                            wandb.log(
                                {f"train/{k}": v for k, v in avg_metrics.items()},
                                step=self.global_step,
                            )
                        except ImportError:
                            pass

                if (
                    self.global_step > 0
                    and self.global_step % self.config.eval_steps == 0
                ):
                    logger.info(f"\nRunning evaluation at step {self.global_step}...")
                    self.run_evaluation()

                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    logger.info(
                        f"\nReached max_steps={self.config.max_steps}, stopping training..."
                    )
                    if self.global_step % self.config.eval_steps != 0:
                        self.run_evaluation()
                    if self.global_step % self.config.save_steps != 0:
                        self.save_checkpoint()
                    return

            avg_epoch = {k: np.mean(v) for k, v in epoch_metrics.items()}
            logger.info(f"Epoch {epoch + 1} Summary:")
            for k, v in avg_epoch.items():
                logger.info(f"  {k}: {v:.4f}")

            if self.global_step % self.config.eval_steps != 0:
                logger.info(f"\nRunning evaluation at end of epoch {epoch + 1}...")
                self.run_evaluation()

            self.save_checkpoint(f"epoch_{epoch + 1}")

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)

    # ----------------------------
    # Checkpoint
    # ----------------------------
    def save_checkpoint(self, name: str = None):
        if name is None:
            name = f"checkpoint-{self.global_step}"
        save_dir = Path(self.config.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "metrics": dict(self.metrics),
            "eval_metrics": dict(self.eval_metrics),
        }
        with open(save_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        if self.eval_metrics:
            eval_summary_path = save_dir / "eval_summary.txt"
            with open(eval_summary_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("Evaluation Results\n")
                f.write("=" * 80 + "\n\n")
                for ds_name, accs in self.eval_metrics.items():
                    f.write(f"{ds_name.upper()}:\n")
                    f.write(f"  Final Accuracy: {accs[-1]:.4f}\n")
                    f.write(f"  Best Accuracy: {max(accs):.4f}\n")
                    f.write(f"  All Results: {accs}\n\n")
                f.write("=" * 80 + "\n")

        logger.info(f"Checkpoint saved: {save_dir}")


# =============================================================================
# Evaluation helper
# =============================================================================


def evaluate_solver_verifier_on_dataset(
    model,
    tokenizer,
    dataset,
    dataset_name: str,
    device,
    log_dir: Path = None,
    config: SVRLConfig = None,
):
    """Lightweight accuracy eval using solver-verifier reward check."""
    if dataset is None:
        return None
    model.eval()
    correct = 0
    total = 0

    sample_logs = []
    import random

    sample_indices = set(random.sample(range(len(dataset)), min(5, len(dataset))))
    progress_bar = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc=f"Evaluating {dataset_name}",
        unit="sample",
    )

    for idx, example in progress_bar:
        question, ground_truth = extract_question_and_answer(example, dataset_name)
        prompt = format_solver_verifier_prompt(question, dataset_name)

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        reward, info = compute_solver_verifier_reward(response, ground_truth, config)
        is_correct = info["is_correct"]
        if is_correct:
            correct += 1
        total += 1

        current_acc = correct / total if total else 0
        progress_bar.set_postfix({"Accuracy": f"{current_acc*100:.1f}%"})

        if idx in sample_indices:
            sample_logs.append(
                {
                    "index": idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response,
                    "predicted": info.get("final_answer"),
                    "boxed": info.get("boxed_answer"),
                    "code_result": info.get("code_result"),
                    "correct": is_correct,
                    "reward": reward,
                }
            )
    progress_bar.close()

    accuracy = correct / total if total else 0.0

    if log_dir and sample_logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_log_file = log_dir / f"eval_samples_{dataset_name}_{timestamp}.txt"
        with open(sample_log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Evaluation Sample Log - {dataset_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write("=" * 80 + "\n\n")
            for i, sample in enumerate(sample_logs, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"Sample {i}/{len(sample_logs)} (Index: {sample['index']})\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Question:\n{sample['question']}\n\n")
                f.write(f"Ground Truth: {sample['ground_truth']}\n")
                f.write(f"Predicted: {sample['predicted']}\n")
                f.write(f"Boxed: {sample['boxed']}\n")
                f.write(f"Code Result: {sample['code_result']}\n")
                f.write(f"Correct: {sample['correct']}\n")
                f.write(f"Reward: {sample['reward']}\n")
                f.write(f"Response:\n{sample['response']}\n")
                f.write(f"{'-'*80}\n")
        logger.info(f"Saved {len(sample_logs)} sample logs to {sample_log_file}")

    return accuracy


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Solver-Verifier GRPO RL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model_path", type=str, help="Override model path")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--datasets", type=str, help="Comma-separated datasets")
    parser.add_argument("--max_samples", type=int, help="Max total samples")
    parser.add_argument("--num_epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Grad accum")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_return_sequences", type=int, help="Generations per prompt")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--kl_coef", type=float, help="KL coefficient")
    parser.add_argument("--eval_steps", type=int, help="Eval steps")
    parser.add_argument("--save_steps", type=int, help="Save steps")
    parser.add_argument("--logging_steps", type=int, help="Logging steps")
    parser.add_argument("--max_steps", type=int, help="Max steps")
    parser.add_argument("--reward_code_error", type=float, help="Penalty for code error")
    parser.add_argument(
        "--reward_code_inconsistent", type=float, help="Penalty for code inconsistency"
    )
    parser.add_argument(
        "--reward_code_consistent", type=float, help="Bonus for code consistency"
    )
    parser.add_argument(
        "--reward_duplicate_answer", type=float, help="Penalty for duplicate answers"
    )
    parser.add_argument("--reward_missing_box", type=float, help="Penalty for no boxed")
    parser.add_argument("--eval_samples", type=int, help="Samples per dataset for eval (0 disables)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="slm_math_rl", help="W&B project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.datasets:
        config.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation_steps:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_return_sequences:
        config.num_return_sequences = args.num_return_sequences
    if args.temperature:
        config.temperature = args.temperature
    if args.kl_coef is not None:
        config.kl_coef = args.kl_coef
    if args.eval_steps:
        config.eval_steps = args.eval_steps
    if args.save_steps:
        config.save_steps = args.save_steps
    if args.logging_steps:
        config.logging_steps = args.logging_steps
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.reward_code_error is not None:
        config.reward_code_error = args.reward_code_error
    if args.reward_code_inconsistent is not None:
        config.reward_code_inconsistent = args.reward_code_inconsistent
    if args.reward_code_consistent is not None:
        config.reward_code_consistent = args.reward_code_consistent
    if args.reward_duplicate_answer is not None:
        config.reward_duplicate_answer = args.reward_duplicate_answer
    if args.reward_missing_box is not None:
        config.reward_missing_box = args.reward_missing_box
    if args.eval_samples is not None:
        config.eval_samples = args.eval_samples

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("Solver-Verifier GRPO Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Datasets: {config.datasets}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"LoRA: {config.use_lora} (r={config.lora_r}, alpha={config.lora_alpha})")
    logger.info(
        f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}"
    )
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Eval samples per dataset: {config.eval_samples}")
    logger.info("=" * 80)

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            run_name = args.wandb_run_name or f"sv_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_path": config.model_path,
                    "datasets": config.datasets,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.per_device_train_batch_size,
                    "num_epochs": config.num_epochs,
                    "num_return_sequences": config.num_return_sequences,
                    "temperature": config.temperature,
                    "kl_coef": config.kl_coef,
                    "reward_code_error": config.reward_code_error,
                    "reward_code_inconsistent": config.reward_code_inconsistent,
                    "reward_code_consistent": config.reward_code_consistent,
                    "reward_duplicate_answer": config.reward_duplicate_answer,
                    "reward_missing_box": config.reward_missing_box,
                },
            )
            logger.info(f"Weights & Biases initialized: project={args.wandb_project}, run={run_name}")
        except ImportError:
            logger.warning("wandb not installed, disabling wandb logging")
            args.use_wandb = False

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For decoder-only models use left padding to avoid generation offset issues
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if config.use_lora:
        logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if config.kl_coef > 0:
        logger.info("Loading reference model for KL penalty...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            device_map="auto",
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
        ref_model = None

    train_data = load_train_data(config.datasets, config.max_samples)

    eval_datasets = {}
    if config.eval_samples and config.eval_samples > 0:
        logger.info("\nLoading evaluation datasets...")
        for name in config.datasets:
            ds_path = project_root / "data" / name
            if not ds_path.exists():
                continue
            try:
                ds = load_from_disk(str(ds_path))
                split = "test" if "test" in ds else list(ds.keys())[0]
                eval_ds = ds[split]
                if config.eval_samples and len(eval_ds) > config.eval_samples:
                    eval_ds = eval_ds.select(range(config.eval_samples))
                eval_datasets[name] = eval_ds
            except Exception as e:
                logger.warning(f"Failed to load eval split for {name}: {e}")
        if eval_datasets:
            logger.info(f"Loaded evaluation datasets: {list(eval_datasets.keys())}")
        else:
            logger.warning("No evaluation datasets loaded")

    trainer = SVGRPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_datasets=eval_datasets,
        use_wandb=args.use_wandb,
    )

    trainer.train()

    if args.use_wandb and wandb_run is not None:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass

    logger.info("Training finished successfully!")


if __name__ == "__main__":
    main()

