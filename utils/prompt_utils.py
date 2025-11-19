"""
Prompt utilities for formatting prompts, extracting answers, and processing responses.
"""

import os
import re
import json
from typing import Optional
from datasets import load_from_disk


# ============================================================================
# PROMPT FORMATTING FUNCTIONS
# ============================================================================

def format_prompt_standard(question: str, dataset_name: str) -> str:
    # """Format prompt for standard (non-thinking) mode."""
    # instruction = (
    #     "Solve using bullet-point math expressions only. "
    #     "Follow one reasoning path without restating or reconsidering steps. "
    #     "After the calculations, output **Final Answer**: \\boxed{numeric value} and stop."
    # )
    # return f"{instruction}\nQuestion: {question}"

    return f"""{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""


def format_prompt_thinking(question: str, dataset_name: str) -> str:
    # can be changed
    return format_prompt_standard(question, dataset_name)


# ============================================================================
# ANSWER EXTRACTION AND CHECKING FUNCTIONS
# ============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text, trying multiple patterns."""
    # Try specific answer patterns first
    patterns = [
        r'####\s*([+-]?\d+\.?\d*)',  # GSM8K format: #### 42
        r'(?:answer|Answer|ANSWER)[\s:=]+\$?([+-]?\d+\.?\d*)',  # "answer: 42" or "Answer = 42"
        r'\\boxed\{([^}]+)\}',  # LaTeX boxed
        r'\$([+-]?\d+\.?\d*)\s*$',  # Dollar amount at end: "$18"
        r'([+-]?\d+\.?\d*)\s*(?:dollars?|eggs?|meters?|bolts?|people?|students?|items?)\s*(?:\.|$)',  # "18 dollars" or "540 meters"
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Take the last match (usually the final answer)
            answer = matches[-1].group(1).strip()
            if answer:
                return normalize_answer(answer)
    
    # Try to find any number near the end of text
    last_numbers = re.findall(r'([+-]?\d+\.?\d*)', text[-200:])  # Check last 200 chars
    if last_numbers:
        return normalize_answer(last_numbers[-1])
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer to standardized format."""
    answer = answer.strip()
    # Remove common separators and symbols
    answer = answer.replace(',', '').replace('$', '').replace('%', '')
    answer = answer.replace(' ', '')  # Remove spaces
    
    try:
        if '/' in answer:
            # Handle fractions
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                if den != 0:
                    answer = str(num / den)
        
        # Try to convert to number
        float_val = float(answer)
        if float_val.is_integer():
            answer = str(int(float_val))
        else:
            answer = str(float_val)
    except:
        # If not a number, keep as is
        pass
    
    return answer.lower()


def check_answer(predicted: str, ground_truth: str) -> bool:
    if not predicted or not ground_truth:
        return False
    
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)
    
    if pred_normalized == gt_normalized:
        return True
    
    try:
        pred_num = float(pred_normalized)
        gt_num = float(gt_normalized)
        return abs(pred_num - gt_num) < 1e-3
    except:
        pass
    
    return pred_normalized in gt_normalized or gt_normalized in pred_normalized


# ============================================================================
# RESPONSE PARSING FUNCTIONS
# ============================================================================

def parse_thinking_output(response: str) -> dict:
    """Parse thinking mode output to extract analysis, CoT, and answer"""
    result = {
        'analysis': '',
        'chain_of_thought': '',
        'final_answer': ''
    }
    
    # Try to extract Analysis section (multiple formats)
    analysis_patterns = [
        r'\*\*Analysis\*\*:?(.*?)(?:\*\*|$)',
        r'Analysis:?(.*?)(?:Chain of Thought|Step|$)',
        r'<Analysis>(.*?)(?:</Analysis>|<|$)',
    ]
    for pattern in analysis_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result['analysis'] = match.group(1).strip()[:500]
            break
    
    # If no analysis found, use first part of response
    if not result['analysis']:
        lines = response.split('\n')
        result['analysis'] = ' '.join(lines[:3])[:500]
    
    # Try to extract Chain of Thought
    cot_patterns = [
        r'\*\*Chain of Thought\*\*:?(.*?)(?:\*\*Final Answer\*\*|\\boxed|$)',
        r'Chain of Thought:?(.*?)(?:Final Answer|\\boxed|$)',
        r'Step-by-step.*?:?(.*?)(?:Final Answer|\\boxed|$)',
    ]
    for pattern in cot_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result['chain_of_thought'] = match.group(1).strip()[:1000]
            break
    
    # If no CoT found, use full response
    if not result['chain_of_thought']:
        result['chain_of_thought'] = response[:1000]
    
    # Extract final answer
    answer = extract_answer(response)
    result['final_answer'] = answer if answer else ''
    
    return result


# ============================================================================
# DATASET LOADING FUNCTIONS
# ============================================================================

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


# ============================================================================
# FILE UTILITY FUNCTIONS
# ============================================================================

def save_results(results: dict, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


def load_results(input_file: str) -> dict:
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)
