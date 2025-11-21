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


# Implementation for multi-agent: solver prompt
def format_prompt_solver(question: str, checker_feedback: str = "", dataset_name: str = "") -> str:
    """
    Multi-agent solver: accepts optional feedback from checker to improve solution.
    
    Args:
        question: The math problem to solve
        checker_feedback: Optional feedback/tip from checker (empty string for first attempt)
        dataset_name: Dataset name (unused but kept for compatibility)
    """
    if checker_feedback and checker_feedback.strip():
        return f"""{question}

Previous attempt feedback: {checker_feedback}

Please reconsider the problem with the feedback in mind. Reason step by step, and put your final answer within \\boxed{{}}."""
    else:
        return f"""{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""



# Implementation for multi-agent: extract solver's reasoning (CoT) without code blocks
def extract_solver_cot(solver_response: str) -> str:
    """
    Extract the Chain of Thought (reasoning) from solver response.
    Removes code blocks and outputs, keeps only the textual reasoning.
    """
    # Remove code blocks (```python ... ```)
    cot = re.sub(r'```python.*?```', '', solver_response, flags=re.DOTALL)
    # Remove output blocks (```output ... ```)
    cot = re.sub(r'```output.*?```', '', cot, flags=re.DOTALL)
    # Remove any remaining code blocks
    cot = re.sub(r'```.*?```', '', cot, flags=re.DOTALL)
    
    # Remove the final boxed answer
    cot = re.sub(r'\\boxed\{[^}]+\}', '[final answer]', cot)
    
    # Clean up excessive whitespace
    lines = [line.strip() for line in cot.split('\n') if line.strip()]
    cot = ' '.join(lines)
    
    # Limit length to ~400 characters (keep it concise)
    if len(cot) > 400:
        # Find a sentence break near 400 chars
        sentences = re.split(r'[.!?]\s+', cot)
        shortened = ""
        for sent in sentences:
            if len(shortened) + len(sent) < 400:
                shortened += sent + ". "
            else:
                break
        cot = shortened.strip()
    
    return cot.strip()


# Implementation for multi-agent: checker prompt
def format_prompt_checker(question: str, solver_response: str, dataset_name: str = "") -> str:
    """
    Build a prompt for the checker agent.
    
    The checker evaluates if the solver's reasoning (CoT) is sound for the given question.
    """
    # Extract the boxed answer from solver response
    solver_answer = extract_answer(solver_response)
    
    # If no answer found, try to find any boxed answer in the response
    if not solver_answer:
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solver_response)
        if boxed_match:
            solver_answer = boxed_match.group(1).strip()
        else:
            solver_answer = "No answer found"
    
    # Extract solver's reasoning (CoT)
    solver_cot = extract_solver_cot(solver_response)
    
    return f"""Problem: {question}

Solution approach: {solver_cot}
Final answer: {solver_answer}

Review checklist:
1. Does the solution use the right numbers from the problem?
2. Are the calculations logical?
3. Does the final answer make sense?

If you see an error or something wrong, respond: INCORRECT
If everything seems correct, respond: CORRECT
If unsure, respond: UNCLEAR

One word only:"""


# ============================================================================
# ANSWER EXTRACTION AND CHECKING FUNCTIONS
# ============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text, trying multiple patterns."""
    # Try specific answer patterns first
    patterns = [
        # Implementation for multi-agent: handle explicit "Final Answer" forms first
        r'Final Answer[:\s]*\\boxed\{([^}]+)\}',  # Final Answer: \boxed{42}
        r'Final Answer[:\s]+\$?([+-]?\d+\.?\d*)',  # Final Answer: 42
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


# Implementation for multi-agent: parse checker verdict
def parse_checker_verdict(response: str) -> Optional[str]:
    """
    Parse the checker agent's verdict from its response.
    Expected primary format:
        VERDICT: CORRECT
        or
        VERDICT: INCORRECT
    Falls back to simple keyword heuristics if the structured pattern is missing.
    """
    upper = response.upper()

    # Try to find explicit VERDICT: pattern first
    match = re.search(r'VERDICT\s*:\s*(CORRECT|INCORRECT|UNCLEAR)', upper)
    if match:
        return match.group(1).upper()

    # Look for standalone verdict words (our new format)
    # Check for the word at start of line or after "verdict:"
    standalone_match = re.search(r'(^|\n|VERDICT[:\s]+)(CORRECT|INCORRECT|UNCLEAR)(\s|$|\n)', upper)
    if standalone_match:
        return standalone_match.group(2).upper()

    # Handle common typos and variations - be fuzzy
    # Look for CORRECT with possible extra letters or typos
    if re.search(r'\bCORRE+CT\b', upper):  # CORREECT, CORRREECT, etc.
        return "CORRECT"
    if re.search(r'\bCOR+ECT\b', upper):  # CORREECT variant
        return "CORRECT"
    
    # Look for INCORRECT with typos
    if re.search(r'\bINCORRE+CT\b', upper):  # INCORREECT, etc.
        return "INCORRECT"
    if re.search(r'\bINCOR+ECT\b', upper):
        return "INCORRECT"
    
    # Look for UNCLEAR with typos
    if re.search(r'\bUNCLE+AR\b', upper):  # UNCLLEAR, etc.
        return "UNCLEAR"

    # Heuristic fallback: look for keywords in first 200 chars
    head = upper[:200]
    
    # Check for INCORRECT first (more specific than CORRECT)
    if "INCORRECT" in head and "CORRECT" not in head:
        return "INCORRECT"
    
    # Then check for CORRECT (excluding INCORRECT)
    if "CORRECT" in head and "INCORRECT" not in head:
        return "CORRECT"
    
    # Check for UNCLEAR
    if "UNCLEAR" in head:
        return "UNCLEAR"

    # Default: treat as UNCLEAR rather than None so downstream logic can rely on a string
    return "UNCLEAR"


# Implementation for multi-agent: extract checker tip when INCORRECT
def parse_checker_tip(response: str) -> str:
    """
    Extract the tip/feedback from checker response when verdict is INCORRECT.
    Since VERDICT is at the end, the tip is the text BEFORE the VERDICT line.
    Returns the tip text, or empty string if not found.
    """
    # Find the VERDICT line (should be at the end)
    verdict_match = re.search(r'VERDICT\s*:\s*(?:CORRECT|INCORRECT|UNCLEAR)', response, re.IGNORECASE)
    if verdict_match:
        # Get text BEFORE the VERDICT line (this is the tip/reasoning)
        tip_end = verdict_match.start()
        tip = response[:tip_end].strip()
        
        # Clean up the tip - remove common prefixes and empty lines
        tip = re.sub(r'^(tip|feedback|suggestion|hint|note|reasoning)[:\s]*', '', tip, flags=re.IGNORECASE)
        tip = re.sub(r'\n\s*\n', ' ', tip)  # Replace multiple newlines with space
        tip = tip.strip()
        
        # Limit tip length to reasonable size (2-3 sentences)
        if len(tip) > 300:
            # Try to find a good breaking point
            sentences = re.split(r'[.!?]\s+', tip)
            if len(sentences) > 1:
                tip = '. '.join(sentences[:2]) + '.'
            else:
                tip = tip[:300]
        
        return tip
    
    return ""


# Implementation for multi-agent: parse checker reasoning to show in console / logs
def parse_checker_reasoning(response: str) -> str:
    """
    Extract a clean version of the checker response.
    Tries to extract just the VERDICT line and a short context around it.
    If VERDICT is found, returns only that line. Otherwise returns a truncated version.
    """
    # First, try to find the VERDICT line
    verdict_match = re.search(r'(VERDICT\s*:\s*(?:CORRECT|INCORRECT|UNCLEAR))', response, re.IGNORECASE)
    if verdict_match:
        # Return just the VERDICT line
        return verdict_match.group(1).strip()
    
    # If no VERDICT found, try to find REASONING section
    reasoning_match = re.search(r'REASONING\s*:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1).strip()[:200]  # Limit to 200 chars
    
    # Fallback: return first 200 characters of response
    return response.strip()[:200]


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
