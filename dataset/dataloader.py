"""
Dataset loading utilities.
"""

import os
import re
from datasets import load_from_disk


def load_dataset_for_eval(dataset_name: str, base_path: str, split_name: str = 'test'):
    """Load dataset split from disk"""
    dataset_path = os.path.join(base_path, 'data', dataset_name)
    dataset = load_from_disk(dataset_path)
    
    if split_name in dataset:
        return dataset[split_name]
    available = ', '.join(dataset.keys())
    raise ValueError(f"Split '{split_name}' not found in dataset {dataset_name}. Available splits: {available}")


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
