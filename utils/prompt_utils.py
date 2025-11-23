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
    """Format prompt for standard (non-thinking) mode - using common benchmark format"""
    
    return f"""{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""


def format_prompt_thinking(question: str, dataset_name: str) -> str:
    # can be changed
    return format_prompt_standard(question, dataset_name)


# ============================================================================
# ANSWER EXTRACTION AND CHECKING FUNCTIONS
# ============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} with proper brace matching."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    
    # Count braces to find the matching closing brace
    after_boxed = text[idx + 7:]  # Skip "\\boxed{"
    brace_count = 1
    end_idx = 0
    
    for i, char in enumerate(after_boxed):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if brace_count == 0:
        return after_boxed[:end_idx].strip()
    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text, trying multiple patterns."""
    # First try \\boxed{} with proper brace matching
    boxed = extract_boxed_answer(text)
    if boxed:
        return normalize_answer(boxed)
    
    # Try other specific answer patterns
    patterns = [
        r'####\s*([+-]?\d+\.?\d*)',  # GSM8K format: #### 42
        r'(?:answer|Answer|ANSWER)[\s:=]+\$?([+-]?\d+\.?\d*)',  # "answer: 42" or "Answer = 42"
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


def normalize_latex(text: str) -> str:
    """Normalize LaTeX expressions for comparison."""
    if not text:
        return ""
    
    # Remove common LaTeX commands that don't affect value
    text = text.replace('\\left', '').replace('\\right', '')
    text = text.replace('\\,', '').replace('\\:', '').replace('\\;', '')
    text = text.replace('\\!', '').replace('~', '')
    
    # Normalize spacing
    text = re.sub(r'\s+', '', text)
    
    return text.strip().lower()


def normalize_answer(answer: str) -> str:
    """Normalize answer to standardized format."""
    if not answer:
        return ""
    
    answer = answer.strip()
    
    # If answer contains LaTeX commands, normalize it as LaTeX
    if '\\' in answer or any(cmd in answer for cmd in ['frac', 'sqrt', 'pi', 'theta']):
        return normalize_latex(answer)
    
    # Remove common separators and symbols
    answer = answer.replace(',', '').replace('$', '').replace('%', '')
    answer = answer.replace(' ', '')
    
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
    """Check if predicted answer matches ground truth with flexible comparison."""
    if not predicted or not ground_truth:
        return False
    
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)
    
    # Exact match after normalization
    if pred_normalized == gt_normalized:
        return True
    
    # Try numerical comparison
    try:
        pred_num = float(pred_normalized)
        gt_num = float(gt_normalized)
        return abs(pred_num - gt_num) < 1e-3
    except:
        pass
    
    # For LaTeX expressions, check if one contains the other
    # This handles cases like (3, pi/2) vs (3,pi/2) or spacing differences
    if pred_normalized in gt_normalized or gt_normalized in pred_normalized:
        return True
    
    # Additional check: remove all punctuation and compare
    pred_cleaned = re.sub(r'[(),\s]', '', pred_normalized)
    gt_cleaned = re.sub(r'[(),\s]', '', gt_normalized)
    if pred_cleaned == gt_cleaned:
        return True
    
    return False


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
