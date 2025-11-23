"""
On-Policy Distillation for Reasoning Models.
Student model learns from teacher's reasoning traces in an online manner.

Key Features:
- Online generation from teacher model
- Dynamic distillation (not static dataset)
- Support for multiple teacher models
- Compatible with LoRA and QLoRA

Usage:
    python -m models.train_distill --config configs/distill_config.yaml
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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
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
class DistillationConfig:
    """On-Policy Distillation Configuration"""
    # Models
    teacher_model_name: str
    teacher_model_path: str
    student_model_name: str
    student_model_path: str
    
    # Distillation strategy
    distillation_type: str = "on_policy"  # on_policy, iterative, self_distill
    online_generation: bool = True
    generation_batch_size: int = 4
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    
    # Distillation loss
    temperature: float = 2.0
    kl_loss_weight: float = 0.5
    ce_loss_weight: float = 0.5
    
    # Generation parameters
    max_new_tokens: int = 2048
    generation_temperature: float = 0.7
    top_p: float = 0.9
    num_generations_per_prompt: int = 1
    
    # Quality filtering
    filter_by_correctness: bool = True
    min_quality_score: float = 0.7
    
    # Training control
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "results/distill_checkpoints"


class TeacherModel:
    """Teacher model for generating reasoning traces"""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate_reasoning(
        self,
        prompts: List[str],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[List[str]]:
        """Generate reasoning traces for prompts"""
        
        all_generations = []
        
        for prompt in tqdm(prompts, desc="Teacher generating"):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generations = []
            for output in outputs:
                generated_text = self.tokenizer.decode(
                    output[inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                generations.append(generated_text)
            
            all_generations.append(generations)
        
        return all_generations


class DistillationLoss(nn.Module):
    """Combined distillation loss (KL divergence + Cross Entropy)"""
    
    def __init__(
        self,
        temperature: float = 2.0,
        kl_weight: float = 0.5,
        ce_weight: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits (detached)
            labels: Ground truth labels
        """
        
        # KL Divergence loss (knowledge distillation)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Cross Entropy loss (hard labels)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Combined loss
        total_loss = self.kl_weight * kl_loss + self.ce_weight * ce_loss
        
        loss_breakdown = {
            'kl_loss': kl_loss.item(),
            'ce_loss': ce_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_breakdown


class OnPolicyDistillationTrainer:
    """On-policy distillation trainer"""
    
    def __init__(
        self,
        config: DistillationConfig,
        teacher_model: TeacherModel,
        student_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        output_dir: Path
    ):
        self.config = config
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        
        self.device = next(student_model.parameters()).device
        
        # Setup loss
        self.distill_loss = DistillationLoss(
            temperature=config.temperature,
            kl_weight=config.kl_loss_weight,
            ce_weight=config.ce_loss_weight
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Training stats
        self.global_step = 0
        self.stats = {
            'kl_loss': [],
            'ce_loss': [],
            'total_loss': []
        }
    
    def filter_quality_data(
        self,
        prompts: List[str],
        generations: List[List[str]],
        ground_truths: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Filter generated data by quality"""
        
        filtered_prompts = []
        filtered_responses = []
        
        for prompt, gens, gt in zip(prompts, generations, ground_truths):
            # Simple quality check: has answer and correct format
            for gen in gens:
                if self.config.filter_by_correctness:
                    # Check if answer is correct
                    if '\\boxed{' in gen:
                        # TODO: Implement answer extraction and comparison
                        filtered_prompts.append(prompt)
                        filtered_responses.append(gen)
                        break
                else:
                    if '<think>' in gen or '\\boxed{' in gen:
                        filtered_prompts.append(prompt)
                        filtered_responses.append(gen)
                        break
        
        logger.info(f"Filtered: {len(filtered_prompts)}/{len(prompts)} samples passed quality check")
        return filtered_prompts, filtered_responses
    
    def train_step(
        self,
        prompts: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Single training step with online generation"""
        
        # Step 1: Teacher generates reasoning traces
        teacher_generations = self.teacher.generate_reasoning(
            prompts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.generation_temperature,
            top_p=self.config.top_p,
            num_return_sequences=self.config.num_generations_per_prompt
        )
        
        # Step 2: Filter quality data
        filtered_prompts, filtered_responses = self.filter_quality_data(
            prompts, teacher_generations, ground_truths
        )
        
        if len(filtered_prompts) == 0:
            logger.warning("No samples passed quality filter!")
            return {'total_loss': 0.0, 'kl_loss': 0.0, 'ce_loss': 0.0}
        
        # Step 3: Train student on teacher's traces
        self.student.train()
        total_loss = 0.0
        kl_losses = []
        ce_losses = []
        
        for prompt, response in zip(filtered_prompts, filtered_responses):
            full_text = prompt + response
            
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # Get student outputs
            student_outputs = self.student(**inputs)
            student_logits = student_outputs.logits
            
            # Get teacher outputs (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher.model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Compute distillation loss
            loss, breakdown = self.distill_loss(
                student_logits,
                teacher_logits,
                inputs['input_ids']
            )
            
            total_loss += loss
            kl_losses.append(breakdown['kl_loss'])
            ce_losses.append(breakdown['ce_loss'])
        
        # Backward and optimize
        avg_loss = total_loss / len(filtered_prompts)
        avg_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return {
            'total_loss': avg_loss.item(),
            'kl_loss': np.mean(kl_losses) if kl_losses else 0.0,
            'ce_loss': np.mean(ce_losses) if ce_losses else 0.0
        }
    
    def save_checkpoint(self, step: int):
        """Save student model checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.student.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="On-Policy Distillation Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--teacher_model', type=str, help='Override teacher model')
    parser.add_argument('--student_model', type=str, help='Override student model')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    args = parser.parse_args()
    
    # Load config
    logger.info("="*80)
    logger.info("On-Policy Distillation for Reasoning Models")
    logger.info("="*80)
    
    config_dict = load_config(args.config)
    
    # TODO: Implement full training loop
    # This is a skeleton implementation
    
    logger.info("On-policy distillation implementation in progress...")
    logger.info("Please refer to the full implementation guide.")


if __name__ == "__main__":
    main()


