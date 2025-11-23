"""
Training utilities for loading and preprocessing CoT data for SFT.
Handles data loading, formatting, and tokenization for reasoning model training.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class CoTDataConfig:
    """Configuration for CoT data processing"""
    max_seq_length: int = 1536
    train_split_ratio: float = 0.9
    add_special_tokens: bool = True
    padding: str = "max_length"
    truncation: bool = True


def load_cot_data(data_files: Union[str, List[str]]) -> List[Dict]:
    """Load CoT data from JSON file(s).
    
    Args:
        data_files: Single file path or list of file paths
    
    Returns:
        List of data items with question, thinking_process, solution, etc.
    """
    if isinstance(data_files, str):
        data_files = [data_files]
    
    all_data = []
    for file_path in data_files:
        logger.info(f"Loading data from {file_path}")
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict) and 'results' in data:
            # Format: {"metadata": {...}, "results": [...]}
            items = data['results']
            logger.info(f"Loaded {len(items)} samples from {path.name} (metadata format)")
        elif isinstance(data, list):
            # Format: [{"question": ..., ...}, ...]
            items = data
            logger.info(f"Loaded {len(items)} samples from {path.name} (list format)")
        else:
            raise ValueError(f"Unsupported data format in {file_path}")
        
        # Filter successful samples
        valid_items = [
            item for item in items 
            if item.get('status') == 'success' or 'status' not in item
        ]
        logger.info(f"  → {len(valid_items)} valid samples (filtered by status)")
        
        all_data.extend(valid_items)
    
    logger.info(f"Total loaded: {len(all_data)} samples from {len(data_files)} file(s)")
    return all_data


def format_cot_prompt(question: str, dataset_name: str = "gsm8k") -> str:
    """Format the input prompt for the model.
    
    Args:
        question: The math problem question
        dataset_name: Name of the dataset
    
    Returns:
        Formatted prompt string
    """
    return f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def format_cot_response(thinking_process: str, solution: str) -> str:
    """Format the target response with <think> tags.
    
    Args:
        thinking_process: The reasoning/thinking process
        solution: The step-by-step solution with final answer
    
    Returns:
        Formatted response string in DeepSeek-R1 style
    """
    # Clean up thinking process and solution
    thinking_process = thinking_process.strip()
    solution = solution.strip()
    
    # Format: <think>reasoning</think>\n\nsolution
    formatted_response = f"<think>\n{thinking_process}\n</think>\n\n{solution}"
    
    return formatted_response


def create_training_example(
    item: Dict, 
    dataset_name: str = "gsm8k"
) -> Dict[str, str]:
    """Create a training example from a CoT data item.
    
    Args:
        item: Data item with question, thinking_process, solution
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with 'prompt' and 'response' keys
    """
    question = item.get('question', '')
    thinking_process = item.get('thinking_process', '')
    solution = item.get('solution', '')
    
    if not question or not thinking_process or not solution:
        raise ValueError(f"Missing required fields in data item: {item}")
    
    prompt = format_cot_prompt(question, dataset_name)
    response = format_cot_response(thinking_process, solution)
    
    return {
        'prompt': prompt,
        'response': response,
        'question': question,
        'ground_truth': item.get('ground_truth', ''),
    }


def prepare_dataset(
    data_files: Union[str, List[str]],
    config: CoTDataConfig,
    dataset_name: str = "gsm8k"
) -> DatasetDict:
    """Prepare training and validation datasets.
    
    Args:
        data_files: Path(s) to CoT data JSON file(s)
        config: Data configuration
        dataset_name: Name of the dataset
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    # Load raw data
    raw_data = load_cot_data(data_files)
    
    # Create training examples
    logger.info("Creating training examples...")
    examples = []
    skipped = 0
    
    for item in raw_data:
        try:
            example = create_training_example(item, dataset_name)
            examples.append(example)
        except (ValueError, KeyError) as e:
            skipped += 1
            logger.debug(f"Skipped invalid item: {e}")
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} invalid items")
    
    logger.info(f"Created {len(examples)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    # Split into train/validation
    train_size = int(len(dataset) * config.train_split_ratio)
    val_size = len(dataset) - train_size
    
    logger.info(f"Splitting dataset: {train_size} train, {val_size} validation")
    
    dataset_dict = dataset.train_test_split(
        train_size=train_size,
        test_size=val_size,
        seed=42
    )
    
    return DatasetDict({
        'train': dataset_dict['train'],
        'validation': dataset_dict['test']
    })


def tokenize_function(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    padding: str = "max_length",
    truncation: bool = True
) -> Dict:
    """Tokenize examples for training.
    
    Args:
        examples: Batch of examples with 'prompt' and 'response'
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
    
    Returns:
        Dictionary with tokenized inputs
    """
    # Combine prompt and response for causal LM training
    full_texts = [
        prompt + response 
        for prompt, response in zip(examples['prompt'], examples['response'])
    ]
    
    # Tokenize
    tokenized = tokenizer(
        full_texts,
        max_length=max_seq_length,
        padding=padding,
        truncation=truncation,
        return_tensors=None,  # Return lists, not tensors
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def compute_data_statistics(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dict:
    """Compute statistics about the dataset.
    
    Args:
        dataset: Prepared dataset
        tokenizer: Tokenizer instance
    
    Returns:
        Dictionary with statistics
    """
    logger.info("Computing dataset statistics...")
    
    prompt_lengths = []
    response_lengths = []
    total_lengths = []
    
    # Sample subset for efficiency (max 1000 samples)
    sample_size = min(1000, len(dataset))
    samples = dataset.select(range(sample_size))
    
    for example in samples:
        prompt_tokens = tokenizer(example['prompt'], return_tensors='pt')['input_ids']
        response_tokens = tokenizer(example['response'], return_tensors='pt')['input_ids']
        
        prompt_len = prompt_tokens.shape[1]
        response_len = response_tokens.shape[1]
        
        prompt_lengths.append(prompt_len)
        response_lengths.append(response_len)
        total_lengths.append(prompt_len + response_len)
    
    stats = {
        'num_samples': len(dataset),
        'prompt_length': {
            'mean': sum(prompt_lengths) / len(prompt_lengths),
            'max': max(prompt_lengths),
            'min': min(prompt_lengths),
        },
        'response_length': {
            'mean': sum(response_lengths) / len(response_lengths),
            'max': max(response_lengths),
            'min': min(response_lengths),
        },
        'total_length': {
            'mean': sum(total_lengths) / len(total_lengths),
            'max': max(total_lengths),
            'min': min(total_lengths),
        }
    }
    
    logger.info(f"Dataset statistics (based on {sample_size} samples):")
    logger.info(f"  Total samples: {stats['num_samples']}")
    logger.info(f"  Prompt length: {stats['prompt_length']['mean']:.0f} (avg), "
                f"{stats['prompt_length']['max']} (max)")
    logger.info(f"  Response length: {stats['response_length']['mean']:.0f} (avg), "
                f"{stats['response_length']['max']} (max)")
    logger.info(f"  Total length: {stats['total_length']['mean']:.0f} (avg), "
                f"{stats['total_length']['max']} (max)")
    
    return stats

