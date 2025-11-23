"""
Prompt utilities for formatting prompts, extracting answers, and processing responses.
"""

import re
from typing import Optional


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
    """Enhanced answer checking with support for MATH500 formats"""
    if not predicted or not ground_truth:
        return False
    
    # Step 1: Basic normalization
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)
    
    if pred_normalized == gt_normalized:
        return True
    
    # Step 2: LaTeX format normalization
    def normalize_latex(s: str) -> str:
        """Normalize LaTeX formats"""
        # Unify fraction commands
        s = s.replace(r'\dfrac', r'\frac')
        s = s.replace(r'\tfrac', r'\frac')
        
        # Normalize sqrt: \sqrt{2} and \sqrt2
        s = re.sub(r'\\sqrt\s+(\d+)', r'\\sqrt{\1}', s)  # \sqrt 2 -> \sqrt{2}
        s = re.sub(r'\\sqrt([a-zA-Z0-9]+)(?![{])', r'\\sqrt{\1}', s)  # \sqrt2 -> \sqrt{2}
        
        # Remove LaTeX spacing commands
        s = s.replace(r'\!', '')
        s = s.replace(r'\,', '')
        s = s.replace(r'\:', '')
        s = s.replace(r'\;', '')
        s = s.replace(r'\ ', ' ')
        
        # Remove degree symbol variants
        s = s.replace('^\\circ', '')
        s = s.replace('\\circ', '')
        s = s.replace('degrees', '')
        s = s.replace('degree', '')
        
        # Remove dollar signs
        s = s.replace('\\$', '')
        s = s.replace('$', '')
        
        # Remove \text{} wrapper but keep content
        s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
        
        # Remove \left and \right
        s = s.replace(r'\left', '')
        s = s.replace(r'\right', '')
        
        # Normalize parentheses spacing
        s = re.sub(r'\(\s+', '(', s)
        s = re.sub(r'\s+\)', ')', s)
        s = re.sub(r',\s+', ',', s)
        
        # Remove spaces around operators
        s = re.sub(r'\s*([+\-*/=])\s*', r'\1', s)
        
        # Clean up general whitespace
        s = ' '.join(s.split())
        
        return s.strip()
    
    pred_latex = normalize_latex(pred_normalized)
    gt_latex = normalize_latex(gt_normalized)
    
    if pred_latex == gt_latex:
        return True
    
    # Step 3: Numerical comparison
    try:
        pred_num = float(pred_latex.replace(',', ''))
        gt_num = float(gt_latex.replace(',', ''))
        if abs(pred_num - gt_num) < 1e-3:
            return True
    except:
        pass
    
    # Step 4: SymPy symbolic comparison (for complex expressions)
    if any(x in pred_latex or x in gt_latex for x in [r'\frac', r'\sqrt', r'\pi', 'i']):
        try:
            from sympy import sympify, simplify, expand, I, pi, sqrt
            
            def latex_to_sympy_expr(s: str) -> str:
                """Convert LaTeX to SymPy-parseable expression"""
                s = s.replace(r'\pi', '*pi')
                s = re.sub(r'^\*pi', 'pi', s)
                
                # \frac{a}{b} -> (a)/(b)
                max_iter = 10
                for _ in range(max_iter):
                    match = re.search(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', s)
                    if match:
                        num, den = match.groups()
                        s = s[:match.start()] + f'(({num})/({den}))' + s[match.end():]
                    else:
                        break
                
                # \sqrt{a} -> sqrt(a)
                s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
                
                # Handle complex numbers: a+bi or a-bi
                s = s.replace('i', '*I')
                s = re.sub(r'([0-9])\*I', r'\1*I', s)
                s = re.sub(r'\+\*I', '+I', s)
                s = re.sub(r'\-\*I', '-I', s)
                
                # Clean up
                s = s.replace('\\', '')
                s = s.replace(' ', '')
                
                return s
            
            pred_sympy = latex_to_sympy_expr(pred_latex)
            gt_sympy = latex_to_sympy_expr(gt_latex)
            
            expr1 = sympify(pred_sympy, evaluate=True)
            expr2 = sympify(gt_sympy, evaluate=True)
            
            # Check difference
            diff = simplify(expr1 - expr2)
            if diff == 0:
                return True
            
            # Check expanded forms
            diff_expanded = simplify(expand(expr1) - expand(expr2))
            if diff_expanded == 0:
                return True
            
        except:
            pass  # Silently fail, continue to substring match
    
    # Step 5: Substring match (for partial matches)
    return pred_latex in gt_latex or gt_latex in pred_latex


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


