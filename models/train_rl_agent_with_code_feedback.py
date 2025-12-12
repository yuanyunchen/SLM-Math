"""
GRPO (Group Relative Policy Optimization) Training Script for Agent with Code Feedback
Rule-based RL training for mathematical reasoning with code execution feedback.

Features:
- Two-step generation: initial reasoning+code → code execution → final answer
- Binary reward verifier (correct/wrong) based on final answer
- On-policy sampling with multiple generations per prompt
- Reference model for KL divergence
- LoRA for parameter-efficient training

Usage:
    python models/train_rl_agent_with_code_feedback.py --config configs/rl_grpo_config.yaml
    python models/train_rl_agent_with_code_feedback.py --config configs/rl_grpo_config.yaml --model_path <your_sft_checkpoint>
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
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset, load_from_disk

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.reward_utils import compute_reward, batch_compute_rewards
from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
from utils.python_code_execution import extract_python_code_blocks, execute_python_code
from models.inference import generate_response

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO Training Configuration"""
    # Model
    model_path: str
    trust_remote_code: bool = True
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Data
    train_file: str = ""
    max_samples: int = -1
    train_split_ratio: float = 0.95
    
    # GRPO
    num_return_sequences: int = 2
    reward_correct: float = 1.0
    reward_wrong: float = -1.0
    reward_no_answer: float = -0.5
    kl_coef: float = 0.05
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
    save_steps: int = 200
    eval_steps: int = 200
    max_steps: int = -1  # -1 means no limit, run full epoch
    output_dir: str = "results/rl_agent_code_feedback_checkpoints"
    
    # System
    seed: int = 42
    bf16: bool = True  # Use bfloat16 for consistency with SFT training
    gradient_checkpointing: bool = True


def load_config_from_yaml(config_path: str) -> GRPOConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested structure
    flat_config = {}
    
    # Model
    flat_config['model_path'] = config_dict['model']['path']
    flat_config['trust_remote_code'] = config_dict['model']['trust_remote_code']
    
    # LoRA
    lora = config_dict.get('lora', {})
    flat_config['use_lora'] = lora.get('enabled', True)
    flat_config['lora_r'] = lora.get('r', 16)
    flat_config['lora_alpha'] = lora.get('alpha', 32)
    flat_config['lora_dropout'] = lora.get('dropout', 0.05)
    flat_config['lora_target_modules'] = lora.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Data
    data = config_dict.get('data', {})
    flat_config['train_file'] = data.get('train_file', '')
    flat_config['max_samples'] = data.get('max_samples', -1)
    flat_config['train_split_ratio'] = data.get('train_split_ratio', 0.95)
    
    # GRPO
    grpo = config_dict.get('grpo', {})
    flat_config['num_return_sequences'] = grpo.get('num_return_sequences', 2)
    flat_config['reward_correct'] = grpo.get('reward_correct', 1.0)
    flat_config['reward_wrong'] = grpo.get('reward_wrong', -1.0)
    flat_config['reward_no_answer'] = grpo.get('reward_no_answer', -0.5)
    flat_config['kl_coef'] = grpo.get('kl_coef', 0.05)
    flat_config['whiten_rewards'] = grpo.get('whiten_rewards', True)
    flat_config['clip_range'] = grpo.get('clip_range', 0.2)
    
    # Generation
    gen = config_dict.get('generation', {})
    flat_config['max_new_tokens'] = gen.get('max_new_tokens', 2048)
    flat_config['temperature'] = gen.get('temperature', 0.7)
    flat_config['top_p'] = gen.get('top_p', 0.9)
    flat_config['top_k'] = gen.get('top_k', 50)
    flat_config['do_sample'] = gen.get('do_sample', True)
    
    # Training
    train = config_dict.get('training', {})
    flat_config['num_epochs'] = train.get('num_epochs', 3)
    flat_config['per_device_train_batch_size'] = train.get('per_device_train_batch_size', 4)
    flat_config['gradient_accumulation_steps'] = train.get('gradient_accumulation_steps', 4)
    flat_config['learning_rate'] = train.get('learning_rate', 5e-6)
    flat_config['max_grad_norm'] = train.get('max_grad_norm', 1.0)
    flat_config['warmup_ratio'] = train.get('warmup_ratio', 0.1)
    flat_config['logging_steps'] = train.get('logging_steps', 10)
    flat_config['save_steps'] = train.get('save_steps', 200)
    flat_config['eval_steps'] = train.get('eval_steps', 200)
    flat_config['output_dir'] = train.get('output_dir', 'results/rl_agent_code_feedback_checkpoints')
    flat_config['seed'] = train.get('seed', 42)
    flat_config['bf16'] = train.get('bf16', True)
    flat_config['gradient_checkpointing'] = train.get('gradient_checkpointing', True)
    
    return GRPOConfig(**flat_config)


def load_data(data_path: str, max_samples: int = -1) -> List[Dict]:
    """Load training data from JSON"""
    logger.info(f"Loading data from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if max_samples > 0 and max_samples < len(data):
        data = data[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    logger.info(f"Loaded {len(data)} samples")
    return data


def format_prompt_with_tools(question: str, dataset_name: str = "") -> str:
    """Format question into prompt with tool instruction (matches agent_with_code_feedback)"""
    tool_instruction = "\n\nYou may use Python code to help with calculations. Show your reasoning step by step."
    return format_prompt_standard(question, dataset_name) + tool_instruction


def load_eval_dataset(dataset_name: str, max_samples: int = 500):
    """Load evaluation dataset"""
    base_path = Path(__file__).parent.parent
    dataset_path = base_path / 'data' / dataset_name
    
    if not dataset_path.exists():
        logger.warning(f"Dataset {dataset_name} not found at {dataset_path}, skipping evaluation")
        return None
    
    try:
        dataset = load_from_disk(str(dataset_path))
        # Use test split if available
        if 'test' in dataset:
            eval_data = dataset['test']
        elif hasattr(dataset, 'keys') and len(dataset.keys()) > 0:
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
        if 'answer' in example:
            return example['answer'].split('####')[-1].strip()
    elif dataset_name in ["math", "math500"]:
        if 'answer' in example:
            return example['answer'].strip()
        if 'solution' in example:
            solution = example['solution']
            idx = solution.find('\\boxed{')
            if idx != -1:
                start = idx + len('\\boxed{')
                depth = 1
                end = start
                while end < len(solution) and depth > 0:
                    if solution[end] == '{':
                        depth += 1
                    elif solution[end] == '}':
                        depth -= 1
                    end += 1
                if depth == 0:
                    return solution[start:end-1].strip()
    return None


def evaluate_model_on_dataset(model, tokenizer, dataset, dataset_name: str, device, log_dir: Path = None):
    """Evaluate model using agent_with_code_feedback workflow"""
    if dataset is None:
        return None
    
    # Import agent function
    from agent.agent_with_code_feedback import run_agent_with_code_feedback
    
    model.eval()
    correct = 0
    total = 0
    
    import random
    sample_indices = set(random.sample(range(len(dataset)), min(10, len(dataset))))
    sample_logs = []
    
    progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Evaluating {dataset_name}", unit="sample")
    
    for idx, example in progress_bar:
        # Extract question
        if dataset_name == "gsm8k":
            question = example.get('question', '')
        elif dataset_name in ["math", "math500"]:
            question = example.get('problem', '')
        else:
            continue
        
        # Extract ground truth
        ground_truth = extract_ground_truth(example, dataset_name)
        if not ground_truth:
            continue
        
        # Run agent with code feedback
        try:
            result = run_agent_with_code_feedback(
                question=question,
                ground_truth=ground_truth,
                model=model,
                tokenizer=tokenizer,
                detailed=False,
                dataset_name=dataset_name,
                enable_tools=True,
                greedy=True,
                apply_chat_template=False
            )
            
            is_correct = result['final_correct']
            predicted = result['predicted_answer']
            response = result['response']
            
        except Exception as e:
            logger.warning(f"Error evaluating sample {idx}: {e}")
            is_correct = False
            predicted = None
            response = f"Error: {str(e)}"
        
        if is_correct:
            correct += 1
        total += 1
        
        # If this is a selected sample, record detailed info
        if idx in sample_indices:
            sample_logs.append({
                'index': idx,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'response': response[:500] if response else "",
                'correct': is_correct
            })
        
        # Update progress bar
        current_accuracy = correct / total if total > 0 else 0.0
        progress_bar.set_postfix({
            'Accuracy': f'{current_accuracy*100:.1f}%',
            'Correct': f'{correct}/{total}'
        })
    
    progress_bar.close()
    
    # Save sample logs
    if log_dir and sample_logs:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sample_log_file = log_dir / f"eval_samples_{dataset_name}_{timestamp}.txt"
        with open(sample_log_file, 'w', encoding='utf-8') as f:
            accuracy = correct / total if total > 0 else 0.0
            f.write("="*80 + "\n")
            f.write(f"Evaluation Sample Log - {dataset_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write("="*80 + "\n\n")
            
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


class GRPOTrainer:
    """Custom GRPO Trainer for agent with code feedback"""
    
    def __init__(
        self,
        config: GRPOConfig,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
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
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Calculate training steps
        self.total_steps = (len(train_data) // config.per_device_train_batch_size) * config.num_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_reward = -float('inf')
        
        # Metrics
        self.metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        
        logger.info(f"Trainer initialized: {self.total_steps} steps, {self.warmup_steps} warmup steps")
    
    def get_lr_multiplier(self):
        """Cosine learning rate schedule with warmup"""
        if self.global_step < self.warmup_steps:
            return self.global_step / max(1, self.warmup_steps)
        else:
            progress = (self.global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    def update_lr(self):
        """Update learning rate"""
        lr_mult = self.get_lr_multiplier()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.learning_rate * lr_mult
    
    def generate_initial_responses(self, prompts: List[str]) -> List[List[str]]:
        """Generate multiple initial responses (reasoning + code) for each prompt"""
        self.model.eval()
        
        # Batch tokenize all prompts with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        batch_size = len(prompts)
        num_return = self.config.num_return_sequences
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=num_return,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Get prompt lengths
        prompt_lengths = inputs['attention_mask'].sum(dim=1).tolist()
        
        # Decode responses
        all_responses = []
        for i in range(batch_size):
            responses = []
            prompt_len = prompt_lengths[i]
            for j in range(num_return):
                output_idx = i * num_return + j
                generated_ids = outputs[output_idx][prompt_len:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                responses.append(response)
            all_responses.append(responses)
        
        return all_responses
    
    def execute_code_and_generate_final(self, prompts: List[str], initial_responses: List[str]) -> Tuple[List[str], List[str]]:
        """Execute code and generate final answers based on execution results
        
        Returns:
            Tuple of (final_responses, full_sequences) where:
            - final_responses: Just the final answer generation
            - full_sequences: Complete sequences (prompt + initial_response + code_output + final_response)
        """
        self.model.eval()
        
        final_responses = []
        full_sequences = []
        
        for prompt, initial_response in zip(prompts, initial_responses):
            # Check if response has code
            has_code = "```python" in initial_response
            code_output = ""
            
            if has_code:
                # Extract and execute code
                code_blocks = extract_python_code_blocks(initial_response)
                output_parts = []
                for i, code in enumerate(code_blocks, 1):
                    result = execute_python_code(code, timeout=10)
                    if result['success']:
                        if result['output'].strip():
                            output_parts.append(f"Code block {i} output:\n{result['output']}")
                        else:
                            output_parts.append(f"Code block {i}: Executed successfully (no output)")
                    else:
                        output_parts.append(f"Code block {i} error:\n{result['error']}")
                code_output = "\n".join(output_parts)
            
            # Build feedback prompt
            if code_output:
                feedback_prompt = f"{prompt}\n\n{initial_response}\n\n```output\n{code_output}\n```\n\nBased on the execution result above, give your final answer in \\boxed{{}}."
            else:
                # No code executed, use initial response
                feedback_prompt = f"{prompt}\n\n{initial_response}\n\nBased on your reasoning above, give your final answer in \\boxed{{}}."
            
            # Generate final answer
            inputs = self.tokenizer(
                feedback_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            prompt_len = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][prompt_len:]
            final_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            final_responses.append(final_response)
            
            # Build full sequence for training
            full_sequence = f"{prompt}\n\n{initial_response}"
            if code_output:
                full_sequence += f"\n\n```output\n{code_output}\n```"
            full_sequence += f"\n\n{final_response}"
            full_sequences.append(full_sequence)
        
        return final_responses, full_sequences
    
    def compute_log_probs_batch(self, prompts: List[str], responses: List[str], model) -> torch.Tensor:
        """Compute log probabilities of responses (batched)"""
        batch_size = len(prompts)
        
        # Get prompt lengths
        prompt_lengths = []
        for prompt in prompts:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_lengths.append(prompt_ids['input_ids'].shape[1])
        
        # Tokenize full texts
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2560,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Compute log probs
        log_probs_list = []
        for i in range(batch_size):
            prompt_len = prompt_lengths[i]
            attention_mask = inputs['attention_mask'][i]
            seq_len = attention_mask.sum().item()
            
            response_logits = logits[i, prompt_len-1:seq_len-1, :]
            response_ids = inputs['input_ids'][i, prompt_len:seq_len]
            
            if len(response_ids) == 0:
                log_probs_list.append(torch.tensor(0.0, device=self.device))
                continue
            
            log_probs_per_token = F.log_softmax(response_logits, dim=-1)
            selected_log_probs = log_probs_per_token.gather(
                dim=-1,
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            mean_log_prob = selected_log_probs.mean()
            log_probs_list.append(mean_log_prob)
        
        return torch.stack(log_probs_list)
    
    def compute_kl_penalty(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute KL divergence penalty (batched)"""
        policy_log_probs = self.compute_log_probs_batch(prompts, responses, self.model)
        
        self.ref_model.eval()
        ref_log_probs = self.compute_log_probs_batch(prompts, responses, self.ref_model)
        
        kl = policy_log_probs - ref_log_probs
        return kl.mean()
    
    def train_step(self, batch: Dict, accumulate: bool = False) -> Dict[str, float]:
        """Single training step with code feedback workflow"""
        questions = batch['question']
        ground_truths = batch['ground_truth']
        dataset_names = batch.get('dataset_name', [''] * len(questions))
        
        # Format prompts
        prompts = [format_prompt_with_tools(q, ds) for q, ds in zip(questions, dataset_names)]
        
        # Generate initial responses (reasoning + code)
        all_initial_responses = self.generate_initial_responses(prompts)
        
        # Flatten for processing
        flat_prompts = []
        flat_initial_responses = []
        flat_truths = []
        
        for prompt, responses, truth in zip(prompts, all_initial_responses, ground_truths):
            for response in responses:
                flat_prompts.append(prompt)
                flat_initial_responses.append(response)
                flat_truths.append(truth)
        
        # Execute code and generate final answers
        final_responses, full_sequences = self.execute_code_and_generate_final(flat_prompts, flat_initial_responses)
        
        # Compute rewards based on final answers
        rewards, infos = batch_compute_rewards(
            final_responses,
            flat_truths,
            reward_correct=self.config.reward_correct,
            reward_wrong=self.config.reward_wrong,
            reward_no_answer=self.config.reward_no_answer
        )
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # Whiten rewards
        if self.config.whiten_rewards and len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute KL penalty on full sequences
        if self.config.kl_coef > 0:
            # For KL, compute on full sequences (prompt + response)
            # compute_log_probs_batch concatenates prompt + response, so we can pass full_sequences as responses
            # and empty strings as prompts, or better: extract response parts
            # Actually, let's use full sequences directly by passing them as responses with empty prompts
            # But that's not ideal. Better: compute KL on the generated parts (initial + final responses)
            # For simplicity, compute KL on full sequences by treating them as responses to empty prompts
            # This approximates the KL divergence
            empty_prompts = [""] * len(full_sequences)
            kl_penalty = self.compute_kl_penalty(empty_prompts, full_sequences)
        else:
            kl_penalty = torch.tensor(0.0, device=self.device)
        
        # Compute policy gradient loss on full sequences
        self.model.train()
        
        batch_size = len(flat_prompts)
        
        # Get prompt lengths (for masking prompt tokens in loss computation)
        prompt_lengths = []
        for prompt in flat_prompts:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_lengths.append(prompt_ids['input_ids'].shape[1])
        
        # Tokenize full sequences (prompt + initial_response + code_output + final_response)
        inputs = self.tokenizer(
            full_sequences,
            return_tensors="pt",
            truncation=True,
            max_length=2560,  # Increased to accommodate full sequences
            padding=True
        ).to(self.device)
        
        # Create labels: mask prompt tokens, keep response tokens
        labels = inputs['input_ids'].clone()
        for i in range(batch_size):
            # Mask prompt tokens
            labels[i, :prompt_lengths[i]] = -100
            # Mask padding tokens
            padding_mask = inputs['attention_mask'][i] == 0
            labels[i, padding_mask] = -100
        
        # Forward pass
        outputs = self.model(**inputs, labels=labels)
        logits = outputs.logits
        
        # Compute loss per token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(batch_size, -1)
        
        # Average loss per sample
        valid_tokens = (shift_labels != -100).float()
        per_sample_loss = (per_token_loss * valid_tokens).sum(dim=1) / (valid_tokens.sum(dim=1) + 1e-8)
        
        # Apply rewards
        weighted_loss = per_sample_loss * rewards
        avg_loss = weighted_loss.mean()
        final_loss = avg_loss + self.config.kl_coef * kl_penalty
        
        # Scale for gradient accumulation
        if accumulate:
            final_loss = final_loss / self.config.gradient_accumulation_steps
        
        # Backward
        final_loss.backward()
        
        # Optimizer step
        if not accumulate:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Metrics
        accuracy = sum([info['correct'] for info in infos]) / len(infos)
        avg_reward = rewards.mean().item()
        
        metrics = {
            'loss': final_loss.item() * (self.config.gradient_accumulation_steps if accumulate else 1),
            'reward': avg_reward,
            'accuracy': accuracy,
            'kl_penalty': kl_penalty.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def run_evaluation(self):
        """Run evaluation on test datasets"""
        if not self.eval_datasets:
            return {}
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Running evaluation at epoch {self.epoch + 1}")
        logger.info(f"{'='*80}")
        
        eval_results = {}
        log_dir = Path(self.config.output_dir)
        for dataset_name, dataset in self.eval_datasets.items():
            logger.info(f"Evaluating on {dataset_name}...")
            accuracy = evaluate_model_on_dataset(
                self.model, self.tokenizer, dataset, dataset_name, self.device,
                log_dir=log_dir
            )
            if accuracy is not None:
                eval_results[f"eval_{dataset_name}_accuracy"] = accuracy
                self.eval_metrics[dataset_name].append(accuracy)
                logger.info(f"  {dataset_name} accuracy: {accuracy:.4f}")
        
        logger.info(f"{'='*80}\n")
        return eval_results
    
    def train(self):
        """Main training loop"""
        logger.info("="*80)
        logger.info("Starting GRPO Training (Agent with Code Feedback)")
        logger.info("="*80)
        
        gradient_accumulation_steps = self.config.gradient_accumulation_steps
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Shuffle data
            import random
            random.shuffle(self.train_data)
            
            # Create batches
            batch_size = self.config.per_device_train_batch_size
            num_batches = len(self.train_data) // batch_size
            
            epoch_metrics = defaultdict(list)
            accumulated_metrics = defaultdict(list)
            
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
            
            for batch_idx in progress_bar:
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = self.train_data[start_idx:end_idx]
                
                batch = {
                    'question': [d['question'] for d in batch_data],
                    'ground_truth': [d['ground_truth'] for d in batch_data],
                    'dataset_name': [d.get('dataset_name', '') for d in batch_data]
                }
                
                # Training step
                self.update_lr()
                metrics = self.train_step(batch, accumulate=(batch_idx % gradient_accumulation_steps != gradient_accumulation_steps - 1))
                
                # Accumulate metrics
                for k, v in metrics.items():
                    accumulated_metrics[k].append(v)
                
                # Optimizer step after accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                    for k, v in accumulated_metrics.items():
                        avg_v = np.mean(v)
                        epoch_metrics[k].append(avg_v)
                        self.metrics[k].append(avg_v)
                    accumulated_metrics = defaultdict(list)
                    self.global_step += 1
                
                # Update progress bar
                if self.global_step % self.config.logging_steps == 0 and len(epoch_metrics['loss']) > 0:
                    avg_metrics = {k: np.mean(v[-self.config.logging_steps:]) for k, v in epoch_metrics.items() if v}
                    progress_bar.set_postfix(avg_metrics)
                    
                    # Log to wandb
                    if self.use_wandb:
                        try:
                            import wandb
                            wandb.log({
                                **avg_metrics,
                                'epoch': self.epoch + 1,
                                'global_step': self.global_step
                            }, step=self.global_step)
                        except ImportError:
                            pass
                
                # Evaluation
                if self.global_step > 0 and self.global_step % self.config.eval_steps == 0:
                    logger.info(f"\nRunning evaluation at step {self.global_step}...")
                    eval_results = self.run_evaluation()
                    
                    if eval_results and self.use_wandb:
                        try:
                            import wandb
                            wandb.log({
                                **eval_results,
                                'epoch': self.epoch + 1
                            }, step=self.global_step)
                        except ImportError:
                            pass
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check max_steps
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    logger.info(f"\nReached max_steps={self.config.max_steps}, stopping training...")
                    if self.global_step % self.config.eval_steps != 0:
                        eval_results = self.run_evaluation()
                    if self.global_step % self.config.save_steps != 0:
                        self.save_checkpoint()
                    return
            
            # Epoch summary
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            logger.info(f"Epoch {epoch + 1} Summary:")
            for k, v in avg_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            
            # Evaluation at epoch end
            steps_since_last_eval = self.global_step % self.config.eval_steps
            if steps_since_last_eval != 0:
                logger.info(f"\nRunning evaluation at end of epoch {epoch + 1}...")
                eval_results = self.run_evaluation()
                
                if eval_results and self.use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            **eval_results,
                            'epoch': self.epoch + 1
                        }, step=self.global_step)
                    except ImportError:
                        pass
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        logger.info("="*80)
        logger.info("Training completed!")
        logger.info("="*80)
    
    def save_checkpoint(self, name: str = None):
        """Save model checkpoint"""
        if name is None:
            name = f"checkpoint-{self.global_step}"
        
        save_dir = Path(self.config.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_reward': self.best_reward,
            'metrics': dict(self.metrics),
            'eval_metrics': dict(self.eval_metrics)
        }
        
        with open(save_dir / 'training_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save evaluation summary
        if self.eval_metrics:
            eval_summary_path = save_dir / 'eval_summary.txt'
            with open(eval_summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("Evaluation Results\n")
                f.write("=" * 80 + "\n\n")
                for dataset_name, accuracies in self.eval_metrics.items():
                    f.write(f"{dataset_name.upper()}:\n")
                    f.write(f"  Final Accuracy: {accuracies[-1]:.4f}\n")
                    f.write(f"  Best Accuracy: {max(accuracies):.4f}\n")
                    f.write(f"  All Results: {accuracies}\n\n")
                f.write("=" * 80 + "\n")
        
        logger.info(f"Checkpoint saved: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="GRPO RL Training for Agent with Code Feedback")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--model_path', type=str, help='Override model path from config')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--max_samples', type=int, help='Override max samples (for testing)')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='slm_math_rl', help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--logging_steps', type=int, default=None, help='Override logging steps from config')
    parser.add_argument('--eval_steps', type=int, default=None, help='Override evaluation steps from config')
    parser.add_argument('--save_steps', type=int, default=None, help='Override save steps from config')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum training steps (-1 for no limit)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Override number of epochs from config')
    parser.add_argument('--batch_size', type=int, default=None, help='Override per-device batch size from config')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None, help='Override gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate from config')
    parser.add_argument('--num_return_sequences', type=int, default=None, help='Number of generations per prompt')
    parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature')
    parser.add_argument('--kl_coef', type=float, default=None, help='KL divergence coefficient')
    parser.add_argument('--eval_samples', type=int, default=200, help='Number of samples per dataset for evaluation')
    args = parser.parse_args()
    
    # Load config
    config = load_config_from_yaml(args.config)
    
    # Override from command line
    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.logging_steps:
        config.logging_steps = args.logging_steps
    if args.eval_steps:
        config.eval_steps = args.eval_steps
    if args.save_steps:
        config.save_steps = args.save_steps
    if args.max_steps is not None:
        config.max_steps = args.max_steps
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
    
    # Set seed
    set_seed(config.seed)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name if args.wandb_run_name else f"grpo_code_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    'model_path': config.model_path,
                    'learning_rate': config.learning_rate,
                    'batch_size': config.per_device_train_batch_size,
                    'num_epochs': config.num_epochs,
                    'lora_r': config.lora_r,
                    'kl_coef': config.kl_coef,
                    'num_return_sequences': config.num_return_sequences
                }
            )
            logger.info(f"Weights & Biases initialized: project={args.wandb_project}, run={run_name}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            args.use_wandb = False
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("="*80)
    logger.info("GRPO RL Training Configuration (Agent with Code Feedback)")
    logger.info("="*80)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Data: {config.train_file}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"LoRA: {config.use_lora} (r={config.lora_r}, alpha={config.lora_alpha})")
    logger.info(f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info("="*80)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto"
    )
    
    # Enable gradient checkpointing
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
    
    # Load reference model (frozen copy)
    if config.kl_coef > 0:
        logger.info("Loading reference model for KL penalty...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            device_map="auto"
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    else:
        logger.info("Skipping reference model (kl_coef=0, saves ~3GB VRAM)")
        ref_model = None
    
    # Load data
    train_data = load_data(config.train_file, config.max_samples)
    
    # Load evaluation datasets
    logger.info("\nLoading evaluation datasets...")
    eval_datasets = {}
    
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
    
    # Create trainer
    trainer = GRPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_datasets=eval_datasets,
        use_wandb=args.use_wandb
    )
    
    # Train
    trainer.train()
    
    # Finish wandb run
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass
    
    logger.info("Training finished successfully!")


if __name__ == "__main__":
    main()

