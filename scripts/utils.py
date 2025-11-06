import re
import json
from typing import Optional, Tuple

def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text, trying multiple patterns."""
    # Try specific answer patterns first
    patterns = [
        r'####\s*([+-]?\d+\.?\d*)',  # GSM8K format: #### 42
        r'(?:answer|Answer|ANSWER)[\s:=]+\$?([+-]?\d+\.?\d*)',  # "answer: 42" or "Answer = 42"
        r'\\boxed\{([^}]+)\}',  # LaTeX boxed
        r'\$([+-]?\d+\.?\d*)\s*$',  # Dollar amount at end: "$18"
        r'([+-]?\d+\.?\d*)\s*(?:dollars?|eggs?|meters?|bolts?)\s*(?:\.|$)',  # "18 dollars" or "540 meters"
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

def save_results(results: dict, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

def load_results(input_file: str) -> dict:
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

