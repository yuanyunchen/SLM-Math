"""
Rule-based Reinforcement Learning (RL) Training for Reasoning Models.
Uses outcome-based rewards for mathematical reasoning tasks.

Key Features:
- Rule-based reward system (correctness, format, reasoning quality)
- PPO (Proximal Policy Optimization) algorithm
- Support for LoRA and QLoRA
- Compatible with CUDA, MPS, and CPU

Usage:
    python -m models.train_RL --config configs/rl_config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import yaml
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.train_utils import load_cot_data

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """RL Training Configuration"""
    # Model
    model_name: str
    model_path: str
    
    # RL Algorithm
    algorithm: str = "ppo"  # ppo, grpo, reinforce
    num_epochs: int = 3
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    # PPO Parameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    kl_coef: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    
    # Learning rates
    learning_rate: float = 1e-5
    value_lr: float = 5e-6
    
    # Reward weights
    correctness_weight: float = 1.0
    format_weight: float = 0.1
    reasoning_quality_weight: float = 0.2
    length_penalty_coef: float = 0.001
    
    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Training
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "results/rl_checkpoints"
    
    # Data
    train_files: List[str] = None
    max_samples: int = -1


class RewardComputer:
    """Compute rule-based rewards for reasoning outputs"""
    
    def __init__(
        self,
        correctness_weight: float = 1.0,
        format_weight: float = 0.1,
        reasoning_quality_weight: float = 0.2,
        length_penalty_coef: float = 0.001
    ):
        self.correctness_weight = correctness_weight
        self.format_weight = format_weight
        self.reasoning_quality_weight = reasoning_quality_weight
        self.length_penalty_coef = length_penalty_coef
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from \\boxed{} format"""
        import re
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        return None
    
    def compute_correctness_reward(
        self, 
        predicted: str, 
        ground_truth: str
    ) -> float:
        """Compute reward based on answer correctness"""
        pred_answer = self.extract_answer(predicted)
        gt_answer = self.extract_answer(ground_truth)
        
        if pred_answer is None:
            return -0.5  # Penalty for no answer
        
        if gt_answer is None:
            return 0.0
        
        # Normalize and compare
        pred_norm = pred_answer.lower().strip()
        gt_norm = gt_answer.lower().strip()
        
        if pred_norm == gt_norm:
            return 1.0
        
        # Try numeric comparison
        try:
            pred_num = float(pred_norm.replace(',', ''))
            gt_num = float(gt_norm.replace(',', ''))
            if abs(pred_num - gt_num) < 1e-3:
                return 1.0
        except:
            pass
        
        return -1.0  # Wrong answer
    
    def compute_format_reward(self, text: str) -> float:
        """Reward for proper formatting"""
        reward = 0.0
        
        # Check for thinking tags
        if '<think>' in text and '</think>' in text:
            reward += 0.5
        
        # Check for boxed answer
        if '\\boxed{' in text:
            reward += 0.5
        
        return reward
    
    def compute_reasoning_quality_reward(self, text: str) -> float:
        """Reward for reasoning quality"""
        reward = 0.0
        
        # Check for step-by-step reasoning
        step_indicators = ['step', 'first', 'then', 'next', 'finally', 'therefore']
        step_count = sum(1 for ind in step_indicators if ind in text.lower())
        reward += min(step_count * 0.1, 0.5)
        
        # Check for mathematical expressions
        math_indicators = ['+', '-', '*', '/', '=', '\\frac', '\\sqrt']
        math_count = sum(1 for ind in math_indicators if ind in text)
        reward += min(math_count * 0.05, 0.5)
        
        return reward
    
    def compute_length_penalty(self, text: str) -> float:
        """Penalty for excessive length"""
        token_count = len(text.split())
        if token_count > 1500:
            return -self.length_penalty_coef * (token_count - 1500)
        return 0.0
    
    def compute_reward(
        self, 
        generated_text: str, 
        ground_truth: str
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total reward with breakdown"""
        
        correctness = self.compute_correctness_reward(generated_text, ground_truth)
        format_reward = self.compute_format_reward(generated_text)
        reasoning_reward = self.compute_reasoning_quality_reward(generated_text)
        length_penalty = self.compute_length_penalty(generated_text)
        
        total_reward = (
            self.correctness_weight * correctness +
            self.format_weight * format_reward +
            self.reasoning_quality_weight * reasoning_reward +
            length_penalty
        )
        
        breakdown = {
            'correctness': correctness,
            'format': format_reward,
            'reasoning_quality': reasoning_reward,
            'length_penalty': length_penalty,
            'total': total_reward
        }
        
        return total_reward, breakdown


class RLTrainer:
    """RL Trainer for reasoning models"""
    
    def __init__(
        self,
        config: RLConfig,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_computer: RewardComputer,
        output_dir: Path
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_computer = reward_computer
        self.output_dir = output_dir
        
        self.device = next(model.parameters()).device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Training stats
        self.global_step = 0
        self.stats = {
            'rewards': [],
            'kl_divergence': [],
            'policy_loss': [],
            'value_loss': []
        }
    
    def generate_responses(
        self,
        prompts: List[str],
        max_new_tokens: int = 2048
    ) -> List[str]:
        """Generate responses for prompts"""
        self.model.eval()
        
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(generated_text)
        
        return responses
    
    def compute_kl_divergence(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference model"""
        self.model.eval()
        self.ref_model.eval()
        
        kl_divs = []
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                policy_logits = self.model(**inputs).logits
                ref_logits = self.ref_model(**inputs).logits
                
                policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                
                kl = torch.sum(
                    torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs),
                    dim=-1
                ).mean()
                
                kl_divs.append(kl.item())
        
        return torch.tensor(kl_divs).mean()
    
    def train_step(
        self,
        prompts: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Single training step"""
        
        # Generate responses
        responses = self.generate_responses(prompts, self.config.max_new_tokens)
        
        # Compute rewards
        rewards = []
        reward_breakdowns = []
        for response, gt in zip(responses, ground_truths):
            reward, breakdown = self.reward_computer.compute_reward(response, gt)
            rewards.append(reward)
            reward_breakdowns.append(breakdown)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(prompts, responses)
        
        # Policy gradient update (simplified PPO)
        self.model.train()
        total_loss = 0.0
        
        for prompt, response, reward in zip(prompts, responses, rewards):
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            
            # Compute policy loss with reward and KL penalty
            policy_loss = -outputs.loss * reward
            kl_penalty = self.config.kl_coef * kl_div
            loss = policy_loss + kl_penalty
            
            total_loss += loss
        
        # Backward and optimize
        avg_loss = total_loss / len(prompts)
        avg_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Collect stats
        avg_reward = np.mean(rewards)
        stats = {
            'reward': avg_reward,
            'kl_div': kl_div.item(),
            'policy_loss': avg_loss.item(),
            'correctness': np.mean([b['correctness'] for b in reward_breakdowns])
        }
        
        self.global_step += 1
        return stats
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="RL Training for Reasoning Models")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_name', type=str, help='Override model name')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    args = parser.parse_args()
    
    # Load config
    logger.info("="*80)
    logger.info("Rule-based RL Training for Reasoning Models")
    logger.info("="*80)
    
    config_dict = load_config(args.config)
    
    # TODO: Implement full training loop
    # This is a skeleton implementation
    
    logger.info("RL training implementation in progress...")
    logger.info("Please refer to the full implementation guide.")


if __name__ == "__main__":
    main()


