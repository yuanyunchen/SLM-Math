"""
Dataset loading utilities (duplicate of utils/prompt_utils.py functions).
This file is kept for backward compatibility but functions are imported from utils.prompt_utils in the main codebase.
"""

import os
import re
from datasets import load_from_disk


def load_dataset_for_eval(dataset_name: str, base_path: str):
    """Load dataset from disk"""
    dataset_path = os.path.join(base_path, 'data', dataset_name)
    dataset = load_from_disk(dataset_path)
    
    if 'test' in dataset:
        return dataset['test']
    elif 'train' in dataset:
        return dataset['train']
    else:
        raise ValueError(f"No valid split found in dataset {dataset_name}")


def extract_question_and_answer(example: dict, dataset_name: str) -> tuple:
    """Extract question and ground truth answer from dataset example"""
    if dataset_name == "gsm8k":
        question = example['question']
        ground_truth = example['answer'].split('####')[-1].strip()
    elif dataset_name == "math":
        question = example['problem']
        solution = example['solution']
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            ground_truth = boxed_match.group(1)
        else:
            numbers = re.findall(r'[+-]?\d+\.?\d*', solution)
            ground_truth = numbers[-1] if numbers else solution
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return question, ground_truth
