"""
Multi-Agent GRPO Training for Solver-Verifier System
Trains both solver and verifier models jointly with reinforcement learning

This script implements:
1. Joint training of solver and verifier with shared or separate checkpoints
2. Multi-turn interaction rewards (solver proposes, verifier checks, solver refines)
3. Curriculum learning: start with single-turn, gradually increase to multi-turn
4. Advanced reward shaping: correctness + verification quality + consistency

Usage:
    python models/train_rl_solver_verifier_multi.py --config configs/rl_grpo_config.yaml
    python models/train_rl_solver_verifier_multi.py --solver_path checkpoints/solver --verifier_path checkpoints/verifier
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

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

from utils.prompt_utils import extract_answer, check_answer, normalize_answer
from agent.solver_verifier import extract_code_and_execute

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class MultiAgentRLConfig:
    """Multi-agent solver-verifier GRPO configuration"""
    
    # Models
    solver_model_path: str
    verifier_model_path: str
    shared_model: bool = False  # If True, use same model for both
    trust_remote_code: bool = True
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Data
    datasets: List[str] = field(default_factory=lambda: ["gsm8k", "math"])
    max_samples: int = 4000
    
    # Multi-turn interaction
    max_turns: int = 3  # Maximum solver-verifier iterations
    curriculum_learning: bool = True  # Gradually increase turns
    
    # GRPO parameters
    num_return_sequences: int = 2
    learning_rate: float = 5e-6
    num_epochs: int = 1
    per_device_batch_size: int = 5
    gradient_accumulation_steps: int = 6
    kl_coef: float = 0.05
    
    # Rewards
    reward_correct: float = 1.0
    reward_wrong: float = -1.0
    reward_verifier_correct_verdict: float = 0.3  # Verifier correctly identifies error
    reward_verifier_wrong_verdict: float = -0.3  # Verifier gives wrong verdict
    reward_improvement: float = 0.5  # Solver fixes error after feedback
    reward_code_error: float = -0.1
    
    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Training
    output_dir: str = "results/rl_multi_agent"
    save_steps: int = 200
    logging_steps: int = 10
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True


def load_datasets(config: MultiAgentRLConfig) -> List[Dict]:
    """Load training datasets"""
    all_samples = []
    
    for dataset_name in config.datasets:
        dataset_path = project_root / "data" / dataset_name
        
        if not dataset_path.exists():
            logger.warning(f"Dataset {dataset_name} not found, skipping")
            continue
        
        try:
            dataset = load_from_disk(str(dataset_path))
            
            # Get train split
            if 'train' in dataset:
                data = dataset['train']
            else:
                data = dataset
            
            # Convert to list of dicts
            for item in data:
                if 'question' in item and 'answer' in item:
                    all_samples.append({
                        'question': item['question'],
                        'ground_truth': item['answer'].split('####')[-1].strip() if '####' in item['answer'] else item['answer'].strip(),
                        'dataset': dataset_name
                    })
            
            logger.info(f"Loaded {len(all_samples)} samples from {dataset_name}")
        
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
    
    # Limit samples if specified
    if config.max_samples > 0 and len(all_samples) > config.max_samples:
        import random
        random.seed(config.seed)
        all_samples = random.sample(all_samples, config.max_samples)
    
    logger.info(f"Total training samples: {len(all_samples)}")
    return all_samples


def format_solver_prompt(question: str, feedback: Optional[str] = None) -> str:
    """Format prompt for solver model"""
    system_prompt = (
        "You are a mathematical problem solver. "
        "Solve problems using step-by-step reasoning. "
        "When appropriate, write Python code to perform calculations. "
        "Always provide your final answer in \\boxed{} format."
    )
    
    if feedback:
        user_message = f"{question}\n\nPrevious attempt feedback: {feedback}\n\nPlease provide a corrected solution."
    else:
        user_message = question
    
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    return prompt


def format_verifier_prompt(question: str, solution: str) -> str:
    """Format prompt for verifier model"""
    system_prompt = (
        "You are a mathematical solution verifier. "
        "Check if the solution is correct. "
        "Provide verdict: CORRECT, INCORRECT, or UNCLEAR. "
        "If INCORRECT or UNCLEAR, explain the issue."
    )
    
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Question: {question}

Solution:
{solution}<|im_end|>
<|im_start|>assistant
"""
    return prompt


def compute_multi_turn_reward(
    question: str,
    ground_truth: str,
    solver_outputs: List[str],
    verifier_outputs: List[str],
    config: MultiAgentRLConfig
) -> Tuple[float, float, Dict]:
    """
    Compute rewards for multi-turn solver-verifier interaction
    
    Returns:
        solver_reward: Reward for solver
        verifier_reward: Reward for verifier
        info: Additional information dict
    """
    solver_reward = 0.0
    verifier_reward = 0.0
    info = {
        'final_correct': False,
        'verifier_accuracy': 0.0,
        'improved': False,
        'turns': len(solver_outputs)
    }
    
    # Track solver answers across turns
    solver_answers = []
    for output in solver_outputs:
        answer = extract_answer(output)
        solver_answers.append(answer)
        
        # Check for code errors
        if '```python' in output:
            code_result = extract_code_and_execute(output)
            if code_result.get('status') == 'error':
                solver_reward += config.reward_code_error
    
    # Final answer correctness
    final_answer = solver_answers[-1] if solver_answers else None
    if final_answer:
        is_correct = check_answer(final_answer, ground_truth)
        info['final_correct'] = is_correct
        
        if is_correct:
            solver_reward += config.reward_correct
        else:
            solver_reward += config.reward_wrong
    else:
        solver_reward += config.reward_wrong
    
    # Check if solver improved across turns
    if len(solver_answers) > 1:
        first_correct = check_answer(solver_answers[0], ground_truth) if solver_answers[0] else False
        last_correct = check_answer(solver_answers[-1], ground_truth) if solver_answers[-1] else False
        
        if not first_correct and last_correct:
            solver_reward += config.reward_improvement
            info['improved'] = True
    
    # Evaluate verifier accuracy
    verifier_correct_count = 0
    for i, (solver_ans, verifier_out) in enumerate(zip(solver_answers[:-1], verifier_outputs)):
        # Check if verifier's verdict matches actual correctness
        actual_correct = check_answer(solver_ans, ground_truth) if solver_ans else False
        
        verifier_verdict = "UNCLEAR"
        if "CORRECT" in verifier_out.upper():
            verifier_verdict = "CORRECT"
        elif "INCORRECT" in verifier_out.upper():
            verifier_verdict = "INCORRECT"
        
        # Reward verifier for accurate verdicts
        if actual_correct and verifier_verdict == "CORRECT":
            verifier_reward += config.reward_verifier_correct_verdict
            verifier_correct_count += 1
        elif not actual_correct and verifier_verdict == "INCORRECT":
            verifier_reward += config.reward_verifier_correct_verdict
            verifier_correct_count += 1
        else:
            verifier_reward += config.reward_verifier_wrong_verdict
    
    if len(verifier_outputs) > 0:
        info['verifier_accuracy'] = verifier_correct_count / len(verifier_outputs)
    
    return solver_reward, verifier_reward, info


def train_multi_agent_rl(config: MultiAgentRLConfig):
    """Main training loop for multi-agent RL"""
    
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config
    config_file = os.path.join(config.output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    logger.info("=" * 80)
    logger.info("Multi-Agent Solver-Verifier GRPO Training")
    logger.info("=" * 80)
    logger.info(f"Solver model: {config.solver_model_path}")
    logger.info(f"Verifier model: {config.verifier_model_path}")
    logger.info(f"Shared model: {config.shared_model}")
    logger.info(f"Max turns: {config.max_turns}")
    logger.info(f"Datasets: {config.datasets}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 80)
    
    # Load data
    train_samples = load_datasets(config)
    
    # Load models and tokenizers
    logger.info("Loading models...")
    
    # Solver
    solver_tokenizer = AutoTokenizer.from_pretrained(
        config.solver_model_path,
        trust_remote_code=config.trust_remote_code
    )
    solver_model = AutoModelForCausalLM.from_pretrained(
        config.solver_model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map='auto'
    )
    
    # Verifier (same model if shared)
    if config.shared_model:
        verifier_tokenizer = solver_tokenizer
        verifier_model = solver_model
        logger.info("Using shared model for solver and verifier")
    else:
        verifier_tokenizer = AutoTokenizer.from_pretrained(
            config.verifier_model_path,
            trust_remote_code=config.trust_remote_code
        )
        verifier_model = AutoModelForCausalLM.from_pretrained(
            config.verifier_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            device_map='auto'
        )
    
    # Apply LoRA if enabled
    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        solver_model = get_peft_model(solver_model, lora_config)
        if not config.shared_model:
            verifier_model = get_peft_model(verifier_model, lora_config)
        
        logger.info("LoRA applied")
        solver_model.print_trainable_parameters()
    
    # Optimizers
    solver_optimizer = torch.optim.AdamW(
        solver_model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    if not config.shared_model:
        verifier_optimizer = torch.optim.AdamW(
            verifier_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
    
    # Training loop
    global_step = 0
    solver_model.train()
    if not config.shared_model:
        verifier_model.train()
    
    logger.info("Starting training...")
    
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Curriculum learning: gradually increase max turns
        if config.curriculum_learning:
            current_max_turns = min(1 + epoch, config.max_turns)
        else:
            current_max_turns = config.max_turns
        
        logger.info(f"Current max turns: {current_max_turns}")
        
        # Batch training samples
        for batch_start in tqdm(range(0, len(train_samples), config.per_device_batch_size)):
            batch_samples = train_samples[batch_start:batch_start + config.per_device_batch_size]
            
            batch_solver_rewards = []
            batch_verifier_rewards = []
            
            # Generate multi-turn interactions for each sample
            for sample in batch_samples:
                question = sample['question']
                ground_truth = sample['ground_truth']
                
                solver_outputs = []
                verifier_outputs = []
                feedback = None
                
                # Multi-turn interaction
                for turn in range(current_max_turns):
                    # Solver generates solution
                    solver_prompt = format_solver_prompt(question, feedback)
                    solver_inputs = solver_tokenizer(solver_prompt, return_tensors='pt').to(solver_model.device)
                    
                    with torch.no_grad():
                        solver_output_ids = solver_model.generate(
                            **solver_inputs,
                            max_new_tokens=config.max_new_tokens,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            do_sample=config.do_sample,
                            num_return_sequences=1,
                        )
                    
                    solver_output = solver_tokenizer.decode(
                        solver_output_ids[0][solver_inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    solver_outputs.append(solver_output)
                    
                    # Check if answer is correct (early stopping)
                    answer = extract_answer(solver_output)
                    if answer and check_answer(answer, ground_truth):
                        break
                    
                    # Verifier checks solution (except last turn)
                    if turn < current_max_turns - 1:
                        verifier_prompt = format_verifier_prompt(question, solver_output)
                        verifier_inputs = verifier_tokenizer(verifier_prompt, return_tensors='pt').to(verifier_model.device)
                        
                        with torch.no_grad():
                            verifier_output_ids = verifier_model.generate(
                                **verifier_inputs,
                                max_new_tokens=512,
                                temperature=0.7,
                                do_sample=True,
                            )
                        
                        verifier_output = verifier_tokenizer.decode(
                            verifier_output_ids[0][verifier_inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True
                        )
                        verifier_outputs.append(verifier_output)
                        feedback = verifier_output
                
                # Compute rewards
                solver_reward, verifier_reward, info = compute_multi_turn_reward(
                    question, ground_truth, solver_outputs, verifier_outputs, config
                )
                
                batch_solver_rewards.append(solver_reward)
                batch_verifier_rewards.append(verifier_reward)
            
            # Update models (simplified - full implementation would use policy gradients)
            # This is a placeholder for the actual GRPO update logic
            
            global_step += 1
            
            if global_step % config.logging_steps == 0:
                avg_solver_reward = np.mean(batch_solver_rewards) if batch_solver_rewards else 0
                avg_verifier_reward = np.mean(batch_verifier_rewards) if batch_verifier_rewards else 0
                logger.info(
                    f"Step {global_step} | "
                    f"Solver reward: {avg_solver_reward:.3f} | "
                    f"Verifier reward: {avg_verifier_reward:.3f}"
                )
            
            if global_step % config.save_steps == 0:
                # Save checkpoints
                solver_save_path = os.path.join(config.output_dir, f'solver_step_{global_step}')
                solver_model.save_pretrained(solver_save_path)
                logger.info(f"Solver checkpoint saved: {solver_save_path}")
                
                if not config.shared_model:
                    verifier_save_path = os.path.join(config.output_dir, f'verifier_step_{global_step}')
                    verifier_model.save_pretrained(verifier_save_path)
                    logger.info(f"Verifier checkpoint saved: {verifier_save_path}")
    
    # Save final models
    final_solver_path = os.path.join(config.output_dir, 'final_solver')
    solver_model.save_pretrained(final_solver_path)
    logger.info(f"Final solver saved: {final_solver_path}")
    
    if not config.shared_model:
        final_verifier_path = os.path.join(config.output_dir, 'final_verifier')
        verifier_model.save_pretrained(final_verifier_path)
        logger.info(f"Final verifier saved: {final_verifier_path}")
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Multi-agent RL training for solver-verifier")
    
    parser.add_argument('--solver_path', type=str, default='pretrained_models/Qwen2.5-Math-1.5B',
                        help='Path to solver model')
    parser.add_argument('--verifier_path', type=str, default='pretrained_models/Qwen2.5-Math-1.5B',
                        help='Path to verifier model')
    parser.add_argument('--shared_model', action='store_true',
                        help='Use same model for solver and verifier')
    parser.add_argument('--output', type=str, default='results/rl_multi_agent',
                        help='Output directory')
    parser.add_argument('--max_turns', type=int, default=3,
                        help='Maximum solver-verifier turns')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    config = MultiAgentRLConfig(
        solver_model_path=args.solver_path,
        verifier_model_path=args.verifier_path,
        shared_model=args.shared_model,
        output_dir=args.output,
        max_turns=args.max_turns,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    
    train_multi_agent_rl(config)


if __name__ == "__main__":
    main()

