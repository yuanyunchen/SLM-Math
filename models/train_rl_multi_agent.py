"""
Multi-Agent GRPO Training: Solver + Verifier Co-Evolution

This script trains two models jointly:
1. Solver: Generates code-based solutions to math problems
2. Verifier: Evaluates solver's code and decides CORRECT/INCORRECT/UNCLEAR

Key differences from single-agent training:
- Two separate models with their own LoRA adapters
- Solver receives feedback from verifier verdicts
- Verifier learns to distinguish correct from incorrect solutions
- Joint optimization with separate reward signals

Based on solver_verifier_multi.py workflow and train_rl_base.py GRPO implementation.

Usage:
    python models/train_rl_multi_agent.py --config configs/rl_multi_agent_config.yaml
    python models/train_rl_multi_agent.py --config configs/rl_multi_agent_config.yaml \
        --solver_model_path <solver_sft_checkpoint> \
        --verifier_model_path <verifier_sft_checkpoint>
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
from utils.python_code_execution import extract_python_code_blocks, execute_python_code

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
class MultiAgentRLConfig:
    """Multi-Agent GRPO configuration for Solver + Verifier."""

    # Solver Model
    solver_model_path: str
    solver_trust_remote_code: bool = True

    # Verifier Model
    verifier_model_path: str
    verifier_trust_remote_code: bool = True

    # LoRA for Solver
    solver_use_lora: bool = True
    solver_lora_r: int = 16
    solver_lora_alpha: int = 32
    solver_lora_dropout: float = 0.05
    solver_lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # LoRA for Verifier
    verifier_use_lora: bool = True
    verifier_lora_r: int = 16
    verifier_lora_alpha: int = 32
    verifier_lora_dropout: float = 0.05
    verifier_lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Data
    datasets: List[str] = field(default_factory=lambda: ["gsm8k", "math500"])
    max_samples: int = -1
    train_split_ratio: float = 0.95

    # GRPO - Solver Rewards
    solver_num_return_sequences: int = 2
    solver_reward_correct: float = 1.0
    solver_reward_wrong: float = -1.0
    solver_reward_no_answer: float = -0.5
    solver_reward_code_error: float = -0.2
    solver_reward_verified: float = 0.3  # Bonus when verifier says CORRECT
    solver_reward_rejected: float = -0.1  # Penalty when verifier says INCORRECT

    # GRPO - Verifier Rewards
    verifier_num_return_sequences: int = 1  # Usually 1 for classifier
    verifier_reward_correct_prediction: float = 1.0  # Verifier correctly predicts
    verifier_reward_wrong_prediction: float = -1.0  # Verifier incorrectly predicts
    verifier_reward_unclear: float = -0.3  # Penalty for UNCLEAR verdict

    # KL and normalization
    kl_coef: float = 0.0
    whiten_rewards: bool = True
    clip_range: float = 0.2

    # Generation - Solver
    solver_max_new_tokens: int = 2048
    solver_temperature: float = 0.7
    solver_top_p: float = 0.9
    solver_top_k: int = 50
    solver_do_sample: bool = True

    # Generation - Verifier
    verifier_max_new_tokens: int = 128
    verifier_temperature: float = 0.3
    verifier_top_p: float = 0.9
    verifier_do_sample: bool = False  # Greedy for classifier

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    solver_learning_rate: float = 5e-6
    verifier_learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    max_steps: int = -1
    output_dir: str = "results/rl_multi_agent_checkpoints"
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    eval_samples: int = 100

    # Multi-agent specific
    train_solver: bool = True  # Whether to train solver
    train_verifier: bool = True  # Whether to train verifier
    verifier_update_frequency: int = 1  # Update verifier every N solver steps


def load_config_from_yaml(config_path: str) -> MultiAgentRLConfig:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def g(path, default=None):
        cur = cfg
        for p in path:
            if cur is None or p not in cur:
                return default
            cur = cur[p]
        return cur

    config = MultiAgentRLConfig(
        # Solver Model
        solver_model_path=g(["solver", "model_path"], "pretrained_models/Qwen2.5-Math-1.5B"),
        solver_trust_remote_code=g(["solver", "trust_remote_code"], True),
        # Verifier Model
        verifier_model_path=g(["verifier", "model_path"], "pretrained_models/Qwen2.5-Math-1.5B"),
        verifier_trust_remote_code=g(["verifier", "trust_remote_code"], True),
        # Solver LoRA
        solver_use_lora=g(["solver", "lora", "enabled"], True),
        solver_lora_r=g(["solver", "lora", "r"], 16),
        solver_lora_alpha=g(["solver", "lora", "alpha"], 32),
        solver_lora_dropout=g(["solver", "lora", "dropout"], 0.05),
        solver_lora_target_modules=g(
            ["solver", "lora", "target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        # Verifier LoRA
        verifier_use_lora=g(["verifier", "lora", "enabled"], True),
        verifier_lora_r=g(["verifier", "lora", "r"], 16),
        verifier_lora_alpha=g(["verifier", "lora", "alpha"], 32),
        verifier_lora_dropout=g(["verifier", "lora", "dropout"], 0.05),
        verifier_lora_target_modules=g(
            ["verifier", "lora", "target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        # Data
        datasets=g(["data", "datasets"], ["gsm8k", "math500"]),
        max_samples=g(["data", "max_samples"], -1),
        train_split_ratio=g(["data", "train_split_ratio"], 0.95),
        # Solver GRPO rewards
        solver_num_return_sequences=g(["solver", "grpo", "num_return_sequences"], 2),
        solver_reward_correct=g(["solver", "grpo", "reward_correct"], 1.0),
        solver_reward_wrong=g(["solver", "grpo", "reward_wrong"], -1.0),
        solver_reward_no_answer=g(["solver", "grpo", "reward_no_answer"], -0.5),
        solver_reward_code_error=g(["solver", "grpo", "reward_code_error"], -0.2),
        solver_reward_verified=g(["solver", "grpo", "reward_verified"], 0.3),
        solver_reward_rejected=g(["solver", "grpo", "reward_rejected"], -0.1),
        # Verifier GRPO rewards
        verifier_num_return_sequences=g(["verifier", "grpo", "num_return_sequences"], 1),
        verifier_reward_correct_prediction=g(["verifier", "grpo", "reward_correct_prediction"], 1.0),
        verifier_reward_wrong_prediction=g(["verifier", "grpo", "reward_wrong_prediction"], -1.0),
        verifier_reward_unclear=g(["verifier", "grpo", "reward_unclear"], -0.3),
        # Common GRPO
        kl_coef=g(["grpo", "kl_coef"], 0.0),
        whiten_rewards=g(["grpo", "whiten_rewards"], True),
        clip_range=g(["grpo", "clip_range"], 0.2),
        # Solver generation
        solver_max_new_tokens=g(["solver", "generation", "max_new_tokens"], 2048),
        solver_temperature=g(["solver", "generation", "temperature"], 0.7),
        solver_top_p=g(["solver", "generation", "top_p"], 0.9),
        solver_top_k=g(["solver", "generation", "top_k"], 50),
        solver_do_sample=g(["solver", "generation", "do_sample"], True),
        # Verifier generation
        verifier_max_new_tokens=g(["verifier", "generation", "max_new_tokens"], 128),
        verifier_temperature=g(["verifier", "generation", "temperature"], 0.3),
        verifier_top_p=g(["verifier", "generation", "top_p"], 0.9),
        verifier_do_sample=g(["verifier", "generation", "do_sample"], False),
        # Training
        num_epochs=g(["training", "num_epochs"], 3),
        per_device_train_batch_size=g(["training", "per_device_train_batch_size"], 2),
        gradient_accumulation_steps=g(["training", "gradient_accumulation_steps"], 8),
        solver_learning_rate=g(["solver", "learning_rate"], 5e-6),
        verifier_learning_rate=g(["verifier", "learning_rate"], 1e-5),
        max_grad_norm=g(["training", "max_grad_norm"], 1.0),
        warmup_ratio=g(["training", "warmup_ratio"], 0.1),
        logging_steps=g(["training", "logging_steps"], 10),
        eval_steps=g(["training", "eval_steps"], 200),
        save_steps=g(["training", "save_steps"], 200),
        max_steps=g(["training", "max_steps"], -1),
        output_dir=g(["training", "output_dir"], "results/rl_multi_agent_checkpoints"),
        seed=g(["training", "seed"], 42),
        bf16=g(["training", "bf16"], True),
        gradient_checkpointing=g(["training", "gradient_checkpointing"], True),
        eval_samples=g(["training", "eval_samples"], 100),
        # Multi-agent
        train_solver=g(["training", "train_solver"], True),
        train_verifier=g(["training", "train_verifier"], True),
        verifier_update_frequency=g(["training", "verifier_update_frequency"], 1),
    )

    config.datasets = [d.strip() for d in config.datasets]
    return config


# =============================================================================
# Prompts
# =============================================================================


def format_solver_prompt(question: str, dataset_name: str) -> str:
    """Format prompt for solver with code instruction."""
    base = format_prompt_standard(question, dataset_name)
    tool_hint = (
        "\n\nYou may use Python code inside ```python``` blocks to compute intermediate "
        "results. Execute code carefully and keep the final answer in \\boxed{}."
    )
    return base + tool_hint


def format_verifier_prompt(question: str, code_snippet: str, code_output: str, candidate_answer: str) -> str:
    """Format prompt for verifier (matches solver_verifier_multi.py)."""
    return f"""You are a verifier. Decide if the given answer is correct.

Problem:
{question}

Solver code (latest):
```python
{code_snippet.strip() if code_snippet else "(no code provided)"}
```

Code execution output (last line is the candidate answer):
{code_output if code_output else "(no execution output)"}

Candidate answer: {candidate_answer}

Label with one of:
- CORRECT (confident the answer matches the problem)
- INCORRECT (confident the answer is wrong)
- UNCLEAR (not enough evidence)

Respond with exactly one label: CORRECT, INCORRECT, or UNCLEAR."""


# =============================================================================
# Data utilities
# =============================================================================


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
# Code execution and parsing
# =============================================================================


def extract_code_and_execute(response: str, detailed: bool = False) -> Tuple[str, List[Dict], bool]:
    """Extract and execute Python code blocks from response."""
    code_blocks = extract_python_code_blocks(response)
    if not code_blocks:
        return response, [], False
    exec_results = []
    output_parts = []
    for i, code in enumerate(code_blocks, 1):
        result = execute_python_code(code, timeout=10)
        exec_results.append(result)
        if result['success']:
            if result['output'].strip():
                output_parts.append(f"Code block {i} output:\n{result['output']}")
        else:
            output_parts.append(f"Code block {i} error:\n{result['error']}")
    if output_parts:
        code_output = "\n".join(output_parts)
        response_with_output = f"{response}\n\n[Code Execution Results]\n{code_output}"
    else:
        response_with_output = response
    return response_with_output, exec_results, len(exec_results) > 0


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


def parse_verifier_label(response: str) -> str:
    """Parse verifier label from text."""
    up = response.upper()
    if "CORRECT" in up and "INCORRECT" not in up:
        return "CORRECT"
    if "INCORRECT" in up:
        return "INCORRECT"
    if "UNCLEAR" in up or "UNSURE" in up or "NOT SURE" in up:
        return "UNCLEAR"
    return "UNCLEAR"


# =============================================================================
# Reward computation
# =============================================================================


def compute_solver_reward(
    response: str,
    ground_truth: str,
    verifier_verdict: str,
    config: MultiAgentRLConfig,
) -> Tuple[float, Dict]:
    """
    Compute reward for solver response.
    
    The reward combines:
    1. Base correctness reward (comparing with ground truth)
    2. Code execution signals
    3. Verifier feedback signals
    """
    boxed_answer = extract_answer(response)
    _, exec_results, used_code = extract_code_and_execute(response, detailed=False)
    code_result = parse_code_result(exec_results)
    
    main_answer = boxed_answer if boxed_answer else code_result
    
    reward = 0.0
    info = {
        "boxed_answer": boxed_answer,
        "code_result": code_result,
        "used_code": used_code,
        "has_code_error": any(not r.get("success") for r in exec_results) if exec_results else False,
        "final_answer": main_answer,
        "verifier_verdict": verifier_verdict,
    }
    
    # Base correctness reward
    if main_answer is None:
        reward = config.solver_reward_no_answer
        info["is_correct"] = False
        info["reason"] = "no_answer"
    else:
        is_correct = check_answer(main_answer, ground_truth)
        info["is_correct"] = is_correct
        reward = config.solver_reward_correct if is_correct else config.solver_reward_wrong
    
    # Code error penalty
    if used_code and info["has_code_error"]:
        reward += config.solver_reward_code_error
    
    # Verifier feedback signal
    if verifier_verdict == "CORRECT":
        reward += config.solver_reward_verified
    elif verifier_verdict == "INCORRECT":
        reward += config.solver_reward_rejected
    # UNCLEAR: no additional reward/penalty
    
    info["reward"] = reward
    return reward, info


def compute_verifier_reward(
    verifier_verdict: str,
    solver_is_correct: bool,
    config: MultiAgentRLConfig,
) -> Tuple[float, Dict]:
    """
    Compute reward for verifier prediction.
    
    The verifier is rewarded for correctly predicting whether the solver's answer is correct.
    """
    info = {
        "verifier_verdict": verifier_verdict,
        "solver_is_correct": solver_is_correct,
    }
    
    if verifier_verdict == "UNCLEAR":
        reward = config.verifier_reward_unclear
        info["prediction_correct"] = False
        info["reason"] = "unclear"
    elif verifier_verdict == "CORRECT":
        if solver_is_correct:
            reward = config.verifier_reward_correct_prediction
            info["prediction_correct"] = True
            info["reason"] = "true_positive"
        else:
            reward = config.verifier_reward_wrong_prediction
            info["prediction_correct"] = False
            info["reason"] = "false_positive"
    else:  # INCORRECT
        if not solver_is_correct:
            reward = config.verifier_reward_correct_prediction
            info["prediction_correct"] = True
            info["reason"] = "true_negative"
        else:
            reward = config.verifier_reward_wrong_prediction
            info["prediction_correct"] = False
            info["reason"] = "false_negative"
    
    info["reward"] = reward
    return reward, info


# =============================================================================
# Multi-Agent Trainer
# =============================================================================


class MultiAgentGRPOTrainer:
    """GRPO trainer for joint Solver + Verifier training."""

    def __init__(
        self,
        config: MultiAgentRLConfig,
        solver_model: AutoModelForCausalLM,
        solver_tokenizer,
        verifier_model: AutoModelForCausalLM,
        verifier_tokenizer,
        solver_ref_model: AutoModelForCausalLM = None,
        verifier_ref_model: AutoModelForCausalLM = None,
        train_data: List[Dict] = None,
        eval_datasets: dict = None,
        use_wandb: bool = False,
    ):
        self.config = config
        self.solver_model = solver_model
        self.solver_tokenizer = solver_tokenizer
        self.verifier_model = verifier_model
        self.verifier_tokenizer = verifier_tokenizer
        self.solver_ref_model = solver_ref_model
        self.verifier_ref_model = verifier_ref_model
        self.train_data = train_data or []
        self.eval_datasets = eval_datasets or {}
        self.use_wandb = use_wandb

        self.solver_device = next(solver_model.parameters()).device
        self.verifier_device = next(verifier_model.parameters()).device

        # Separate optimizers for each model
        if config.train_solver:
            self.solver_optimizer = torch.optim.AdamW(
                solver_model.parameters(),
                lr=config.solver_learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
        else:
            self.solver_optimizer = None
            
        if config.train_verifier:
            self.verifier_optimizer = torch.optim.AdamW(
                verifier_model.parameters(),
                lr=config.verifier_learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
        else:
            self.verifier_optimizer = None

        self.total_steps = (
            len(train_data) // config.per_device_train_batch_size
        ) * config.num_epochs if train_data else 0
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)

        self.global_step = 0
        self.epoch = 0

        self.metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)

        logger.info(
            f"Multi-Agent Trainer initialized: {self.total_steps} steps, {self.warmup_steps} warmup steps"
        )

    def get_lr_multiplier(self):
        if self.global_step < self.warmup_steps:
            return self.global_step / max(1, self.warmup_steps)
        progress = (self.global_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    def update_lr(self):
        lr_mult = self.get_lr_multiplier()
        if self.solver_optimizer:
            for group in self.solver_optimizer.param_groups:
                group["lr"] = self.config.solver_learning_rate * lr_mult
        if self.verifier_optimizer:
            for group in self.verifier_optimizer.param_groups:
                group["lr"] = self.config.verifier_learning_rate * lr_mult

    # ----------------------------
    # Generation utilities
    # ----------------------------
    
    def generate_solver_responses(self, prompts: List[str]) -> List[List[str]]:
        """Generate multiple solver responses per prompt."""
        self.solver_model.eval()
        inputs = self.solver_tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.solver_device)

        batch_size = len(prompts)
        n_ret = self.config.solver_num_return_sequences

        with torch.no_grad():
            outputs = self.solver_model.generate(
                **inputs,
                max_new_tokens=self.config.solver_max_new_tokens,
                num_return_sequences=n_ret,
                temperature=self.config.solver_temperature,
                top_p=self.config.solver_top_p,
                top_k=self.config.solver_top_k,
                do_sample=self.config.solver_do_sample,
                pad_token_id=self.solver_tokenizer.eos_token_id,
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
                    self.solver_tokenizer.decode(gen_ids, skip_special_tokens=True)
                )
            all_responses.append(responses)
        return all_responses

    def generate_verifier_response(self, prompt: str) -> str:
        """Generate a single verifier response."""
        self.verifier_model.eval()
        inputs = self.verifier_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.verifier_device)

        with torch.no_grad():
            outputs = self.verifier_model.generate(
                **inputs,
                max_new_tokens=self.config.verifier_max_new_tokens,
                temperature=self.config.verifier_temperature,
                top_p=self.config.verifier_top_p,
                do_sample=self.config.verifier_do_sample,
                pad_token_id=self.verifier_tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][prompt_len:]
        return self.verifier_tokenizer.decode(gen_ids, skip_special_tokens=True)

    def generate_verifier_responses_batch(self, prompts: List[str]) -> List[str]:
        """Generate verifier responses for a batch of prompts."""
        self.verifier_model.eval()
        inputs = self.verifier_tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(self.verifier_device)

        with torch.no_grad():
            outputs = self.verifier_model.generate(
                **inputs,
                max_new_tokens=self.config.verifier_max_new_tokens,
                temperature=self.config.verifier_temperature,
                top_p=self.config.verifier_top_p,
                do_sample=self.config.verifier_do_sample,
                pad_token_id=self.verifier_tokenizer.eos_token_id,
            )

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        responses = []
        for i in range(len(prompts)):
            prompt_len = prompt_lengths[i]
            gen_ids = outputs[i][prompt_len:]
            responses.append(
                self.verifier_tokenizer.decode(gen_ids, skip_special_tokens=True)
            )
        return responses

    # ----------------------------
    # Policy gradient computation
    # ----------------------------

    def compute_policy_loss(
        self,
        model,
        tokenizer,
        device,
        prompts: List[str],
        responses: List[str],
        rewards: torch.Tensor,
        accumulate: bool = False,
    ) -> torch.Tensor:
        """Compute policy gradient loss for a model."""
        model.train()
        batch_size = len(prompts)

        # Get prompt lengths
        prompt_lengths = []
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
            prompt_lengths.append(ids["input_ids"].shape[1])

        # Tokenize full texts
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2560,
            padding=True,
        ).to(device)

        # Create labels: mask prompt tokens and padding
        labels = inputs["input_ids"].clone()
        for i in range(batch_size):
            labels[i, : prompt_lengths[i]] = -100
            padding_mask = inputs["attention_mask"][i] == 0
            labels[i, padding_mask] = -100

        # Forward pass
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits

        # Compute per-sample loss
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

        # Weight by rewards: REINFORCE
        weighted_loss = per_sample_loss * rewards
        avg_loss = weighted_loss.mean()

        if accumulate:
            avg_loss = avg_loss / self.config.gradient_accumulation_steps

        return avg_loss

    # ----------------------------
    # Training step
    # ----------------------------

    def train_step(self, batch: Dict, accumulate: bool = False) -> Dict[str, float]:
        """Single training step for both solver and verifier."""
        questions = batch["question"]
        truths = batch["ground_truth"]
        datasets = batch["dataset"]

        # ========== SOLVER GENERATION ==========
        solver_prompts = [
            format_solver_prompt(q, ds) for q, ds in zip(questions, datasets)
        ]
        all_solver_responses = self.generate_solver_responses(solver_prompts)

        # Flatten solver outputs
        flat_solver_prompts = []
        flat_solver_responses = []
        flat_truths = []
        flat_questions = []

        for i, (q, prompt, rs, truth) in enumerate(
            zip(questions, solver_prompts, all_solver_responses, truths)
        ):
            for r in rs:
                flat_solver_prompts.append(prompt)
                flat_solver_responses.append(r)
                flat_truths.append(truth)
                flat_questions.append(q)

        # ========== EXTRACT CODE AND RUN ==========
        solver_exec_results = []
        solver_code_blocks = []
        solver_code_outputs = []
        solver_answers = []

        for response in flat_solver_responses:
            _, exec_results, used_code = extract_code_and_execute(response, detailed=False)
            solver_exec_results.append(exec_results)
            
            # Extract code and output for verifier
            code_blocks = extract_python_code_blocks(response)
            latest_code = code_blocks[-1] if code_blocks else ""
            solver_code_blocks.append(latest_code)
            
            code_output = ""
            if exec_results and exec_results[-1].get("success"):
                code_output = exec_results[-1].get("output", "")
            solver_code_outputs.append(code_output)
            
            # Get candidate answer
            boxed = extract_answer(response)
            code_result = parse_code_result(exec_results)
            answer = boxed if boxed else code_result
            solver_answers.append(answer if answer else "")

        # ========== VERIFIER GENERATION ==========
        verifier_prompts = []
        for q, code, output, answer in zip(
            flat_questions, solver_code_blocks, solver_code_outputs, solver_answers
        ):
            v_prompt = format_verifier_prompt(q, code, output, answer)
            verifier_prompts.append(v_prompt)

        verifier_responses = self.generate_verifier_responses_batch(verifier_prompts)
        verifier_verdicts = [parse_verifier_label(r) for r in verifier_responses]

        # ========== COMPUTE REWARDS ==========
        # Solver rewards
        solver_rewards = []
        solver_infos = []
        for response, truth, verdict in zip(
            flat_solver_responses, flat_truths, verifier_verdicts
        ):
            reward, info = compute_solver_reward(response, truth, verdict, self.config)
            solver_rewards.append(reward)
            solver_infos.append(info)

        # Verifier rewards
        verifier_rewards = []
        verifier_infos = []
        for verdict, solver_info in zip(verifier_verdicts, solver_infos):
            solver_is_correct = solver_info["is_correct"]
            reward, info = compute_verifier_reward(verdict, solver_is_correct, self.config)
            verifier_rewards.append(reward)
            verifier_infos.append(info)

        # Convert to tensors
        solver_rewards_tensor = torch.tensor(solver_rewards, dtype=torch.float32).to(self.solver_device)
        verifier_rewards_tensor = torch.tensor(verifier_rewards, dtype=torch.float32).to(self.verifier_device)

        solver_raw_reward = solver_rewards_tensor.mean().item()
        verifier_raw_reward = verifier_rewards_tensor.mean().item()

        # Whiten rewards
        if self.config.whiten_rewards:
            if len(solver_rewards_tensor) > 1:
                solver_rewards_tensor = (solver_rewards_tensor - solver_rewards_tensor.mean()) / (
                    solver_rewards_tensor.std() + 1e-8
                )
            if len(verifier_rewards_tensor) > 1:
                verifier_rewards_tensor = (verifier_rewards_tensor - verifier_rewards_tensor.mean()) / (
                    verifier_rewards_tensor.std() + 1e-8
                )

        # ========== SOLVER TRAINING ==========
        solver_loss = torch.tensor(0.0)
        if self.config.train_solver and self.solver_optimizer:
            solver_loss = self.compute_policy_loss(
                self.solver_model,
                self.solver_tokenizer,
                self.solver_device,
                flat_solver_prompts,
                flat_solver_responses,
                solver_rewards_tensor,
                accumulate=accumulate,
            )
            solver_loss.backward()

            if not accumulate:
                torch.nn.utils.clip_grad_norm_(
                    self.solver_model.parameters(), self.config.max_grad_norm
                )
                self.solver_optimizer.step()
                self.solver_optimizer.zero_grad()

        # ========== VERIFIER TRAINING ==========
        verifier_loss = torch.tensor(0.0)
        if self.config.train_verifier and self.verifier_optimizer:
            if self.global_step % self.config.verifier_update_frequency == 0:
                verifier_loss = self.compute_policy_loss(
                    self.verifier_model,
                    self.verifier_tokenizer,
                    self.verifier_device,
                    verifier_prompts,
                    verifier_responses,
                    verifier_rewards_tensor,
                    accumulate=accumulate,
                )
                verifier_loss.backward()

                if not accumulate:
                    torch.nn.utils.clip_grad_norm_(
                        self.verifier_model.parameters(), self.config.max_grad_norm
                    )
                    self.verifier_optimizer.step()
                    self.verifier_optimizer.zero_grad()

        # ========== METRICS ==========
        solver_accuracy = sum(info["is_correct"] for info in solver_infos) / max(1, len(solver_infos))
        verifier_accuracy = sum(info["prediction_correct"] for info in verifier_infos) / max(1, len(verifier_infos))
        
        verdict_counts = Counter(verifier_verdicts)
        
        metrics = {
            "solver_loss": solver_loss.item() * (self.config.gradient_accumulation_steps if accumulate else 1),
            "verifier_loss": verifier_loss.item() * (self.config.gradient_accumulation_steps if accumulate else 1),
            "solver_reward": solver_raw_reward,
            "verifier_reward": verifier_raw_reward,
            "solver_accuracy": solver_accuracy,
            "verifier_accuracy": verifier_accuracy,
            "verdict_correct_rate": verdict_counts.get("CORRECT", 0) / max(1, len(verifier_verdicts)),
            "verdict_incorrect_rate": verdict_counts.get("INCORRECT", 0) / max(1, len(verifier_verdicts)),
            "verdict_unclear_rate": verdict_counts.get("UNCLEAR", 0) / max(1, len(verifier_verdicts)),
            "solver_lr": self.solver_optimizer.param_groups[0]["lr"] if self.solver_optimizer else 0,
            "verifier_lr": self.verifier_optimizer.param_groups[0]["lr"] if self.verifier_optimizer else 0,
        }

        return metrics

    # ----------------------------
    # Evaluation
    # ----------------------------

    def run_evaluation(self):
        if not self.eval_datasets:
            return {}
        logger.info(f"\n{'='*80}")
        logger.info(f"Running evaluation at epoch {self.epoch + 1}, step {self.global_step}")
        logger.info(f"{'='*80}")
        
        eval_results = {}
        log_dir = Path(self.config.output_dir)
        
        for ds_name, ds in self.eval_datasets.items():
            logger.info(f"Evaluating on {ds_name}...")
            solver_acc, verifier_acc = self.evaluate_multi_agent_on_dataset(ds, ds_name, log_dir)
            
            if solver_acc is not None:
                eval_results[f"eval_{ds_name}_solver_accuracy"] = solver_acc
                self.eval_metrics[f"{ds_name}_solver"].append(solver_acc)
                logger.info(f"  {ds_name} solver accuracy: {solver_acc:.4f}")
            
            if verifier_acc is not None:
                eval_results[f"eval_{ds_name}_verifier_accuracy"] = verifier_acc
                self.eval_metrics[f"{ds_name}_verifier"].append(verifier_acc)
                logger.info(f"  {ds_name} verifier accuracy: {verifier_acc:.4f}")
                
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log(eval_results, step=self.global_step)
                except ImportError:
                    pass
        
        logger.info(f"{'='*80}\n")
        return eval_results

    def evaluate_multi_agent_on_dataset(
        self, dataset, dataset_name: str, log_dir: Path = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """Evaluate both solver and verifier on a dataset."""
        if dataset is None:
            return None, None

        self.solver_model.eval()
        self.verifier_model.eval()
        
        solver_correct = 0
        verifier_correct = 0
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
            
            # Solver generation
            solver_prompt = format_solver_prompt(question, dataset_name)
            inputs = self.solver_tokenizer(
                solver_prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.solver_device)

            with torch.no_grad():
                outputs = self.solver_model.generate(
                    **inputs,
                    max_new_tokens=self.config.solver_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.solver_tokenizer.eos_token_id,
                )

            solver_response = self.solver_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            
            # Extract answer and code
            _, exec_results, _ = extract_code_and_execute(solver_response, detailed=False)
            code_blocks = extract_python_code_blocks(solver_response)
            latest_code = code_blocks[-1] if code_blocks else ""
            code_output = exec_results[-1].get("output", "") if exec_results and exec_results[-1].get("success") else ""
            
            boxed = extract_answer(solver_response)
            code_result = parse_code_result(exec_results)
            solver_answer = boxed if boxed else code_result
            
            solver_is_correct = check_answer(solver_answer, ground_truth) if solver_answer else False
            if solver_is_correct:
                solver_correct += 1
            
            # Verifier generation
            verifier_prompt = format_verifier_prompt(question, latest_code, code_output, solver_answer or "")
            v_inputs = self.verifier_tokenizer(
                verifier_prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.verifier_device)

            with torch.no_grad():
                v_outputs = self.verifier_model.generate(
                    **v_inputs,
                    max_new_tokens=self.config.verifier_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.verifier_tokenizer.eos_token_id,
                )

            verifier_response = self.verifier_tokenizer.decode(
                v_outputs[0][v_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            verifier_verdict = parse_verifier_label(verifier_response)
            
            # Check if verifier is correct
            verifier_is_correct = (
                (verifier_verdict == "CORRECT" and solver_is_correct) or
                (verifier_verdict == "INCORRECT" and not solver_is_correct)
            )
            if verifier_is_correct:
                verifier_correct += 1
            
            total += 1
            
            current_solver_acc = solver_correct / total if total else 0
            current_verifier_acc = verifier_correct / total if total else 0
            progress_bar.set_postfix({
                "Solver": f"{current_solver_acc*100:.1f}%",
                "Verifier": f"{current_verifier_acc*100:.1f}%"
            })
            
            if idx in sample_indices:
                sample_logs.append({
                    "index": idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "solver_response": solver_response[:500],
                    "solver_answer": solver_answer,
                    "solver_correct": solver_is_correct,
                    "verifier_verdict": verifier_verdict,
                    "verifier_correct": verifier_is_correct,
                })

        progress_bar.close()
        
        solver_accuracy = solver_correct / total if total else 0.0
        verifier_accuracy = verifier_correct / total if total else 0.0
        
        # Save sample logs
        if log_dir and sample_logs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_log_file = log_dir / f"eval_samples_{dataset_name}_{timestamp}.txt"
            with open(sample_log_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Multi-Agent Evaluation - {dataset_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Solver Accuracy: {solver_accuracy:.4f}\n")
                f.write(f"Verifier Accuracy: {verifier_accuracy:.4f}\n")
                f.write("=" * 80 + "\n\n")
                for i, sample in enumerate(sample_logs, 1):
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Sample {i}/{len(sample_logs)} (Index: {sample['index']})\n")
                    f.write(f"{'-'*80}\n")
                    f.write(f"Question:\n{sample['question']}\n\n")
                    f.write(f"Ground Truth: {sample['ground_truth']}\n")
                    f.write(f"Solver Answer: {sample['solver_answer']}\n")
                    f.write(f"Solver Correct: {sample['solver_correct']}\n")
                    f.write(f"Verifier Verdict: {sample['verifier_verdict']}\n")
                    f.write(f"Verifier Correct: {sample['verifier_correct']}\n\n")
                    f.write(f"Solver Response:\n{sample['solver_response']}\n")
                    f.write(f"{'-'*80}\n")
            logger.info(f"Saved sample logs to {sample_log_file}")
        
        return solver_accuracy, verifier_accuracy

    # ----------------------------
    # Train loop
    # ----------------------------

    def train(self):
        logger.info("=" * 80)
        logger.info("Starting Multi-Agent GRPO Training")
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
                    accumulate=(batch_idx % grad_acc != grad_acc - 1),
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
                    and len(epoch_metrics["solver_loss"]) > 0
                ):
                    avg_metrics = {
                        k: np.mean(v[-self.config.logging_steps:])
                        for k, v in epoch_metrics.items()
                        if v
                    }
                    progress_bar.set_postfix({
                        "s_acc": f"{avg_metrics.get('solver_accuracy', 0):.3f}",
                        "v_acc": f"{avg_metrics.get('verifier_accuracy', 0):.3f}",
                    })
                    
                    log_parts = [
                        f"Step {self.global_step}",
                        f"s_loss={avg_metrics.get('solver_loss', 0):.4f}",
                        f"v_loss={avg_metrics.get('verifier_loss', 0):.4f}",
                        f"s_rew={avg_metrics.get('solver_reward', 0):.3f}",
                        f"v_rew={avg_metrics.get('verifier_reward', 0):.3f}",
                        f"s_acc={avg_metrics.get('solver_accuracy', 0):.3f}",
                        f"v_acc={avg_metrics.get('verifier_accuracy', 0):.3f}",
                    ]
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

                if self.global_step > 0 and self.global_step % self.config.eval_steps == 0:
                    logger.info(f"\nRunning evaluation at step {self.global_step}...")
                    self.run_evaluation()

                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    logger.info(f"\nReached max_steps={self.config.max_steps}, stopping...")
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

        # Save solver
        solver_dir = save_dir / "solver"
        solver_dir.mkdir(parents=True, exist_ok=True)
        self.solver_model.save_pretrained(solver_dir)
        self.solver_tokenizer.save_pretrained(solver_dir)

        # Save verifier
        verifier_dir = save_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        self.verifier_model.save_pretrained(verifier_dir)
        self.verifier_tokenizer.save_pretrained(verifier_dir)

        # Save training state
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
                f.write("Multi-Agent Evaluation Results\n")
                f.write("=" * 80 + "\n\n")
                for ds_name, accs in self.eval_metrics.items():
                    f.write(f"{ds_name.upper()}:\n")
                    f.write(f"  Final: {accs[-1]:.4f}\n")
                    f.write(f"  Best: {max(accs):.4f}\n")
                    f.write(f"  All: {accs}\n\n")
                f.write("=" * 80 + "\n")

        logger.info(f"Checkpoint saved: {save_dir}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent GRPO RL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--solver_model_path", type=str, help="Override solver model path")
    parser.add_argument("--verifier_model_path", type=str, help="Override verifier model path")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--datasets", type=str, help="Comma-separated datasets")
    parser.add_argument("--max_samples", type=int, help="Max total samples")
    parser.add_argument("--num_epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Grad accum")
    parser.add_argument("--solver_learning_rate", type=float, help="Solver learning rate")
    parser.add_argument("--verifier_learning_rate", type=float, help="Verifier learning rate")
    parser.add_argument("--solver_num_return_sequences", type=int, help="Solver generations per prompt")
    parser.add_argument("--solver_temperature", type=float, help="Solver sampling temperature")
    parser.add_argument("--eval_steps", type=int, help="Eval steps")
    parser.add_argument("--save_steps", type=int, help="Save steps")
    parser.add_argument("--logging_steps", type=int, help="Logging steps")
    parser.add_argument("--max_steps", type=int, help="Max steps")
    parser.add_argument("--eval_samples", type=int, help="Samples per dataset for eval")
    parser.add_argument("--train_solver", action="store_true", default=None, help="Train solver")
    parser.add_argument("--train_verifier", action="store_true", default=None, help="Train verifier")
    parser.add_argument("--no_train_solver", action="store_true", help="Don't train solver")
    parser.add_argument("--no_train_verifier", action="store_true", help="Don't train verifier")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="slm_math_multi_agent", help="W&B project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    
    # Apply overrides
    if args.solver_model_path:
        config.solver_model_path = args.solver_model_path
    if args.verifier_model_path:
        config.verifier_model_path = args.verifier_model_path
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
    if args.solver_learning_rate:
        config.solver_learning_rate = args.solver_learning_rate
    if args.verifier_learning_rate:
        config.verifier_learning_rate = args.verifier_learning_rate
    if args.solver_num_return_sequences:
        config.solver_num_return_sequences = args.solver_num_return_sequences
    if args.solver_temperature:
        config.solver_temperature = args.solver_temperature
    if args.eval_steps:
        config.eval_steps = args.eval_steps
    if args.save_steps:
        config.save_steps = args.save_steps
    if args.logging_steps:
        config.logging_steps = args.logging_steps
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.eval_samples is not None:
        config.eval_samples = args.eval_samples
    if args.no_train_solver:
        config.train_solver = False
    if args.no_train_verifier:
        config.train_verifier = False

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("Multi-Agent GRPO Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Solver Model: {config.solver_model_path}")
    logger.info(f"Verifier Model: {config.verifier_model_path}")
    logger.info(f"Datasets: {config.datasets}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Train Solver: {config.train_solver}")
    logger.info(f"Train Verifier: {config.train_verifier}")
    logger.info(f"Solver LoRA: r={config.solver_lora_r}, alpha={config.solver_lora_alpha}")
    logger.info(f"Verifier LoRA: r={config.verifier_lora_r}, alpha={config.verifier_lora_alpha}")
    logger.info(f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    logger.info(f"Solver LR: {config.solver_learning_rate}")
    logger.info(f"Verifier LR: {config.verifier_learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info("=" * 80)

    # Initialize wandb
    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"multi_agent_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "solver_model_path": config.solver_model_path,
                    "verifier_model_path": config.verifier_model_path,
                    "datasets": config.datasets,
                    "solver_learning_rate": config.solver_learning_rate,
                    "verifier_learning_rate": config.verifier_learning_rate,
                    "batch_size": config.per_device_train_batch_size,
                    "num_epochs": config.num_epochs,
                },
            )
            logger.info(f"W&B initialized: project={args.wandb_project}, run={run_name}")
        except ImportError:
            logger.warning("wandb not installed, disabling")
            args.use_wandb = False

    # Load solver model and tokenizer
    logger.info("Loading solver model...")
    solver_tokenizer = AutoTokenizer.from_pretrained(
        config.solver_model_path, trust_remote_code=config.solver_trust_remote_code
    )
    if solver_tokenizer.pad_token is None:
        solver_tokenizer.pad_token = solver_tokenizer.eos_token
    solver_tokenizer.padding_side = "left"

    solver_model = AutoModelForCausalLM.from_pretrained(
        config.solver_model_path,
        trust_remote_code=config.solver_trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )

    if config.gradient_checkpointing:
        solver_model.gradient_checkpointing_enable()

    if config.solver_use_lora and config.train_solver:
        logger.info("Applying LoRA to solver...")
        solver_lora_config = LoraConfig(
            r=config.solver_lora_r,
            lora_alpha=config.solver_lora_alpha,
            target_modules=config.solver_lora_target_modules,
            lora_dropout=config.solver_lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        solver_model = get_peft_model(solver_model, solver_lora_config)
        solver_model.print_trainable_parameters()

    # Load verifier model and tokenizer
    logger.info("Loading verifier model...")
    verifier_tokenizer = AutoTokenizer.from_pretrained(
        config.verifier_model_path, trust_remote_code=config.verifier_trust_remote_code
    )
    if verifier_tokenizer.pad_token is None:
        verifier_tokenizer.pad_token = verifier_tokenizer.eos_token
    verifier_tokenizer.padding_side = "left"

    verifier_model = AutoModelForCausalLM.from_pretrained(
        config.verifier_model_path,
        trust_remote_code=config.verifier_trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )

    if config.gradient_checkpointing:
        verifier_model.gradient_checkpointing_enable()

    if config.verifier_use_lora and config.train_verifier:
        logger.info("Applying LoRA to verifier...")
        verifier_lora_config = LoraConfig(
            r=config.verifier_lora_r,
            lora_alpha=config.verifier_lora_alpha,
            target_modules=config.verifier_lora_target_modules,
            lora_dropout=config.verifier_lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        verifier_model = get_peft_model(verifier_model, verifier_lora_config)
        verifier_model.print_trainable_parameters()

    # Load training data
    train_data = load_train_data(config.datasets, config.max_samples)

    # Load evaluation datasets
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

    # Create trainer
    trainer = MultiAgentGRPOTrainer(
        config=config,
        solver_model=solver_model,
        solver_tokenizer=solver_tokenizer,
        verifier_model=verifier_model,
        verifier_tokenizer=verifier_tokenizer,
        train_data=train_data,
        eval_datasets=eval_datasets,
        use_wandb=args.use_wandb,
    )

    # Train
    trainer.train()

    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass

    logger.info("Training finished successfully!")


if __name__ == "__main__":
    main()

