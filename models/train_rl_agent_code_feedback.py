"""
GRPO RL Training for Agent with Code Feedback

This trains RL on the 2-step agent workflow from agent_with_code_feedback.py:
  Step 1: Generate reasoning + Python code
  Step 2: Execute code, inject result, generate final \boxed{} answer

The reward is based on the final answer correctness after the full 2-step process.

Usage:
    python -m models.train_rl_agent_code_feedback --model_path pretrained_models/Qwen2.5-Math-1.5B
"""

import os
import sys
import json
import yaml
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import LoraConfig, get_peft_model, TaskType


class StopOnBoxedAnswer(StoppingCriteria):
    """Halts generation once a \\boxed{} answer is produced (same as inference.py)."""

    def __init__(self, tokenizer, prompt_token_len: int):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len

    def _has_boxed_answer(self, text: str) -> bool:
        """Check if text contains a complete \\boxed{} with balanced braces."""
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return False
        
        after_boxed = text[idx + 7:]  # Skip "\\boxed{"
        brace_count = 1
        
        for char in after_boxed:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return True
        
        return False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_token_len:]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._has_boxed_answer(text)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.reward_utils import compute_reward
from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
from utils.python_code_execution import extract_python_code_blocks, execute_python_code

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class AgentCodeFeedbackRLConfig:
    """Config for Agent with Code Feedback RL Training"""
    # Model
    model_path: str = ""
    trust_remote_code: bool = True
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Data
    train_file: str = "data/rstart_sft_interactive/rstar_100k_clean.csv"
    max_samples: int = 300
    
    # GRPO
    num_return_sequences: int = 2
    reward_correct: float = 1.0
    reward_wrong: float = -1.0
    reward_no_answer: float = -0.5
    kl_coef: float = 0.0
    whiten_rewards: bool = True
    
    # Generation (MUST match agent_with_code_feedback.py & generation_config.py)
    max_new_tokens: int = 2048       # Same as MAX_NEW_TOKENS
    temperature: float = 0.7         # Same as TEMPERATURE
    top_p: float = 0.95              # Same as TOP_P (was 0.9, fixed!)
    do_sample: bool = True           # Same as DO_SAMPLE
    repetition_penalty: float = 1.15 # Same as REPETITION_PENALTY (was 1.0, fixed!)
    apply_chat_template: bool = False
    
    # Training
    num_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_steps: int = 50
    max_steps: int = -1
    output_dir: str = "results/rl_agent_code_feedback_checkpoints"
    
    # System
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True


def load_csv_data(data_path: str, max_samples: int = -1) -> List[Dict]:
    """Load training data from CSV"""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    data = []
    
    for idx, row in df.iterrows():
        query = row.get('query', '')
        response = row.get('response', '')
        
        if not query or not response:
            continue
        
        # Extract ground truth from response
        ground_truth = extract_answer(response)
        if ground_truth is None:
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
            if boxed_match:
                ground_truth = boxed_match.group(1).strip()
            else:
                continue
        
        data.append({
            'question': query.strip(),
            'ground_truth': ground_truth,
        })
        
        if max_samples > 0 and len(data) >= max_samples:
            break
    
    logger.info(f"Loaded {len(data)} samples")
    return data


def format_agent_prompt(question: str, dataset_name: str = "") -> str:
    """
    Format prompt exactly like agent_with_code_feedback.py
    
    This is the EXACT same prompt used in the agent for consistency.
    """
    # Base prompt from format_prompt_standard
    base_prompt = format_prompt_standard(question, dataset_name)
    
    # Tool instruction (same as agent_with_code_feedback.py line 72)
    tool_instruction = "\n\nYou may use Python code to help with calculations. Show your reasoning step by step."
    
    return base_prompt + tool_instruction


def run_agent_generation(
    model,
    tokenizer,
    prompt: str,
    config: AgentCodeFeedbackRLConfig,
    device: torch.device
) -> Tuple[str, str, List[Dict], str]:
    """
    Run the 2-step agent workflow:
    Step 1: Generate reasoning + code
    Step 2: Execute code, generate final answer with feedback
    
    Returns:
        Tuple of (full_response, final_answer, exec_results, case_type)
    """
    # Step 1: Generate initial response with reasoning + code
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048  # Match agent (was 512)
    ).to(device)
    
    prompt_len = inputs['input_ids'].shape[1]
    
    # Stop early when boxed answer is found (same as agent)
    stopping_criteria = StoppingCriteriaList([StopOnBoxedAnswer(tokenizer, prompt_len)])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,  # Match agent!
        )
    initial_response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    
    # Check if already has answer without code
    initial_answer = extract_answer(initial_response)
    has_code = "```python" in initial_response
    
    if initial_answer and not has_code:
        # Model gave answer directly without code
        return initial_response, initial_answer, [], "NO_CODE"
    
    # Step 2: Execute code if present
    exec_results = []
    code_output = ""
    
    if has_code:
        code_blocks = extract_python_code_blocks(initial_response)
        
        output_parts = []
        for i, code in enumerate(code_blocks, 1):
            result = execute_python_code(code, timeout=10)
            exec_results.append(result)
            
            if result['success']:
                if result['output'].strip():
                    output_parts.append(f"Code block {i} output:\n{result['output']}")
                else:
                    output_parts.append(f"Code block {i}: Executed successfully (no output)")
            else:
                output_parts.append(f"Code block {i} error:\n{result['error']}")
        
        code_output = "\n".join(output_parts)
    
    # If no code executed, return initial response
    if not exec_results or not code_output:
        return initial_response, initial_answer, exec_results, "NO_EXEC"
    
    # Step 3: Generate final answer with feedback
    # Extract numeric result from code output
    numbers = re.findall(r'[-+]?\d*\.?\d+', code_output)
    exec_result = numbers[-1] if numbers else code_output.strip()
    
    # Build feedback prompt (same as agent_with_code_feedback.py line 206)
    feedback_prompt = f"{prompt}\n\nI calculated the answer step by step. The computation gives {exec_result}. So the answer is \\boxed{{"
    
    # Generate final answer
    feedback_inputs = tokenizer(
        feedback_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048  # Match agent (was 1024)
    ).to(device)
    
    feedback_len = feedback_inputs['input_ids'].shape[1]
    
    # Stop early when boxed answer is complete (same as agent)
    feedback_stopping = StoppingCriteriaList([StopOnBoxedAnswer(tokenizer, feedback_len)])
    
    with torch.no_grad():
        feedback_outputs = model.generate(
            **feedback_inputs,
            max_new_tokens=50,  # Short - just need to complete the boxed answer
            temperature=config.temperature,
            do_sample=config.do_sample,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=feedback_stopping,  # Match agent!
        )
    final_response = tokenizer.decode(feedback_outputs[0][feedback_len:], skip_special_tokens=True)
    
    # Extract final answer
    full_response_for_extraction = f"\\boxed{{{final_response}}}" if "\\boxed{" not in final_response else final_response
    final_answer = extract_answer(full_response_for_extraction)
    
    if not final_answer:
        final_answer = exec_result
    
    # Combine responses
    full_response = initial_response + f"\n\n[Execution result: {exec_result}]\n\n" + final_response
    
    return full_response, final_answer, exec_results, "WITH_FEEDBACK"


class AgentCodeFeedbackRLTrainer:
    """GRPO Trainer for Agent with Code Feedback"""
    
    def __init__(
        self,
        config: AgentCodeFeedbackRLConfig,
        model: AutoModelForCausalLM,
        ref_model: Optional[AutoModelForCausalLM],
        tokenizer: AutoTokenizer,
        train_data: List[Dict],
        use_wandb: bool = False,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.use_wandb = use_wandb
        
        self.device = next(model.parameters()).device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.total_steps = (len(train_data) // config.per_device_train_batch_size) * config.num_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)
        self.global_step = 0
        self.epoch = 0
        
        # Metrics
        self.metrics = defaultdict(list)
        
        logger.info(f"Trainer initialized: {self.total_steps} steps, {self.warmup_steps} warmup")
    
    def get_lr_multiplier(self) -> float:
        """Cosine LR schedule with warmup"""
        if self.global_step < self.warmup_steps:
            return self.global_step / max(1, self.warmup_steps)
        progress = (self.global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    def update_lr(self):
        lr_mult = self.get_lr_multiplier()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.learning_rate * lr_mult
    
    def generate_agent_responses(self, questions: List[str], ground_truths: List[str]) -> List[Dict]:
        """
        Generate responses using the 2-step agent workflow.
        Returns list of results with responses, answers, and rewards.
        """
        self.model.eval()
        results = []
        
        for question, truth in zip(questions, ground_truths):
            prompt = format_agent_prompt(question)
            
            # Generate multiple sequences for GRPO variance
            for _ in range(self.config.num_return_sequences):
                full_response, final_answer, exec_results, case_type = run_agent_generation(
                    self.model, self.tokenizer, prompt, self.config, self.device
                )
                
                # Compute reward based on final answer
                is_correct = check_answer(final_answer, truth) if final_answer else False
                
                if final_answer is None:
                    reward = self.config.reward_no_answer
                elif is_correct:
                    reward = self.config.reward_correct
                else:
                    reward = self.config.reward_wrong
                
                results.append({
                    'prompt': prompt,
                    'response': full_response,
                    'answer': final_answer,
                    'ground_truth': truth,
                    'correct': is_correct,
                    'reward': reward,
                    'case_type': case_type,
                    'num_code_blocks': len(exec_results),
                })
        
        return results
    
    def train_step(self, batch: Dict, accumulate: bool = False) -> Dict[str, float]:
        """Single training step using agent workflow"""
        questions = batch['question']
        ground_truths = batch['ground_truth']
        
        # Generate responses using 2-step agent workflow
        generation_results = self.generate_agent_responses(questions, ground_truths)
        
        # Collect rewards
        rewards = torch.tensor(
            [r['reward'] for r in generation_results],
            dtype=torch.float32
        ).to(self.device)
        
        # Whiten rewards (normalize to mean=0, std=1)
        # This helps with training stability by making the scale consistent
        if self.config.whiten_rewards and len(rewards) > 1:
            reward_std = rewards.std()
            # Edge case: if all rewards are the same (std=0), skip whitening
            # to preserve the learning signal (all correct = encourage all, all wrong = discourage all)
            if reward_std > 1e-6:
                rewards = (rewards - rewards.mean()) / (reward_std + 1e-8)
            else:
                # All same reward - just center around 0 but keep the sign
                # If all correct (+1): rewards become +1 (encourage)
                # If all wrong (-1): rewards become -1 (discourage)
                # This preserves the learning signal
                rewards = rewards - rewards.mean() + rewards[0].sign()
        
        # Compute policy gradient loss
        self.model.train()
        
        # Prepare data for loss computation
        prompts = [r['prompt'] for r in generation_results]
        responses = [r['response'] for r in generation_results]
        batch_size = len(prompts)
        
        # Get prompt lengths
        prompt_lengths = []
        for prompt in prompts:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)  # Match agent
            prompt_lengths.append(prompt_ids['input_ids'].shape[1])
        
        # Tokenize full sequences
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=4096,  # 2048 prompt + 2048 response
            padding=True
        ).to(self.device)
        
        # Create labels (mask prompt)
        labels = inputs['input_ids'].clone()
        for i in range(batch_size):
            labels[i, :prompt_lengths[i]] = -100
            padding_mask = inputs['attention_mask'][i] == 0
            labels[i, padding_mask] = -100
        
        # Forward pass
        outputs = self.model(**inputs, labels=labels)
        logits = outputs.logits
        
        # Compute per-sample loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(batch_size, -1)
        
        valid_tokens = (shift_labels != -100).float()
        per_sample_loss = (per_token_loss * valid_tokens).sum(dim=1) / (valid_tokens.sum(dim=1) + 1e-8)
        
        # Apply rewards
        weighted_loss = per_sample_loss * rewards
        final_loss = weighted_loss.mean()
        
        # Scale for accumulation
        if accumulate:
            final_loss = final_loss / self.config.gradient_accumulation_steps
        
        # Backward
        final_loss.backward()
        
        if not accumulate:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Metrics
        accuracy = sum([r['correct'] for r in generation_results]) / len(generation_results)
        avg_reward = sum([r['reward'] for r in generation_results]) / len(generation_results)
        code_usage = sum([1 for r in generation_results if r['num_code_blocks'] > 0]) / len(generation_results)
        
        return {
            'loss': final_loss.item() * (self.config.gradient_accumulation_steps if accumulate else 1),
            'reward': avg_reward,
            'accuracy': accuracy,
            'code_usage': code_usage,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def train(self):
        """Main training loop"""
        logger.info("=" * 80)
        logger.info("Starting Agent Code Feedback RL Training")
        logger.info("=" * 80)
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            import random
            random.shuffle(self.train_data)
            
            batch_size = self.config.per_device_train_batch_size
            num_batches = len(self.train_data) // batch_size
            
            epoch_metrics = defaultdict(list)
            accumulated_metrics = defaultdict(list)
            
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
            
            for batch_idx in progress_bar:
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = self.train_data[start_idx:end_idx]
                
                batch = {
                    'question': [d['question'] for d in batch_data],
                    'ground_truth': [d['ground_truth'] for d in batch_data]
                }
                
                self.update_lr()
                is_accumulating = (batch_idx % self.config.gradient_accumulation_steps != 
                                   self.config.gradient_accumulation_steps - 1)
                metrics = self.train_step(batch, accumulate=is_accumulating)
                
                for k, v in metrics.items():
                    accumulated_metrics[k].append(v)
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                    if is_accumulating:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    for k, v in accumulated_metrics.items():
                        avg_v = np.mean(v)
                        epoch_metrics[k].append(avg_v)
                        self.metrics[k].append(avg_v)
                    accumulated_metrics = defaultdict(list)
                    self.global_step += 1
                    
                    # Log to wandb
                    if self.use_wandb and len(epoch_metrics['loss']) > 0:
                        try:
                            import wandb
                            latest = {k: v[-1] for k, v in epoch_metrics.items() if v}
                            wandb.log({
                                'train/loss': latest.get('loss', 0),
                                'train/reward': latest.get('reward', 0),
                                'train/accuracy': latest.get('accuracy', 0),
                                'train/code_usage': latest.get('code_usage', 0),
                                'train/lr': latest.get('lr', 0),
                                'progress/step': self.global_step,
                            }, step=self.global_step)
                        except Exception:
                            pass
                
                # Update progress bar
                if self.global_step % self.config.logging_steps == 0 and epoch_metrics['loss']:
                    avg = {k: np.mean(v[-self.config.logging_steps:]) for k, v in epoch_metrics.items() if v}
                    progress_bar.set_postfix({
                        'loss': f"{avg.get('loss', 0):.4f}",
                        'acc': f"{avg.get('accuracy', 0):.3f}",
                        'code': f"{avg.get('code_usage', 0):.2f}",
                    })
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Max steps check
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max_steps={self.config.max_steps}")
                    self.save_checkpoint()
                    return
            
            # Epoch summary
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            logger.info(f"Epoch {epoch + 1} Summary:")
            for k, v in avg_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)
    
    def save_checkpoint(self, name: str = None):
        if name is None:
            name = f"checkpoint-{self.global_step}"
        
        save_dir = Path(self.config.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'metrics': dict(self.metrics),
        }
        with open(save_dir / 'training_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="GRPO RL for Agent with Code Feedback")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--train_file', type=str, default='data/rstart_sft_interactive/rstar_100k_clean.csv')
    parser.add_argument('--output_dir', type=str, default='results/rl_agent_code_feedback_checkpoints')
    parser.add_argument('--max_samples', type=int, default=300)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_return_sequences', type=int, default=2)
    parser.add_argument('--kl_coef', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)  # Match generation_config.py
    parser.add_argument('--repetition_penalty', type=float, default=1.15)  # Match generation_config.py
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--apply_chat_template', type=str, default='false')
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default='nlp_final_math')
    parser.add_argument('--wandb_project', type=str, default='rl_agent_code_feedback')
    args = parser.parse_args()
    
    # Create config
    config = AgentCodeFeedbackRLConfig()
    config.model_path = args.model_path
    config.train_file = args.train_file
    config.output_dir = args.output_dir
    config.max_samples = args.max_samples
    config.num_epochs = args.num_epochs
    config.per_device_train_batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.learning_rate = args.learning_rate
    config.num_return_sequences = args.num_return_sequences
    config.kl_coef = args.kl_coef
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.repetition_penalty = args.repetition_penalty
    config.max_new_tokens = args.max_new_tokens
    config.apply_chat_template = args.apply_chat_template.lower() == 'true'
    config.logging_steps = args.logging_steps
    config.save_steps = args.save_steps
    config.max_steps = args.max_steps
    config.seed = args.seed
    
    set_seed(config.seed)
    
    # Initialize wandb
    if args.use_wandb:
        try:
            import wandb
            run_name = f"agent_code_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=run_name,
                config={
                    'model': config.model_path,
                    'samples': config.max_samples,
                    'batch_size': config.per_device_train_batch_size,
                    'lr': config.learning_rate,
                }
            )
            logger.info(f"W&B: {run_name}")
        except ImportError:
            args.use_wandb = False
    
    # Create output dir
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Log config
    logger.info("=" * 80)
    logger.info("Agent Code Feedback RL Training")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Samples: {config.max_samples}")
    logger.info(f"Batch: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    logger.info(f"LR: {config.learning_rate}")
    logger.info("=" * 80)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto"
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    if config.use_lora:
        logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load data
    train_data = load_csv_data(config.train_file, config.max_samples)
    
    if len(train_data) == 0:
        logger.error("No data loaded!")
        return
    
    # Create trainer
    trainer = AgentCodeFeedbackRLTrainer(
        config=config,
        model=model,
        ref_model=None,  # No KL penalty
        tokenizer=tokenizer,
        train_data=train_data,
        use_wandb=args.use_wandb
    )
    
    # Train
    trainer.train()
    
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

