"""
Prompt utilities for formatting prompts, extracting answers, and processing responses.
"""

import re
from typing import Optional


# ============================================================================
# PROMPT FORMATTING FUNCTIONS
# ============================================================================

def format_prompt_standard(question: str, dataset_name: str) -> str:
    """Format prompt for standard (non-thinking) mode."""
    return f"""{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""


# ============================================================================
# ANSWER EXTRACTION AND CHECKING FUNCTIONS
# ============================================================================

def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} with balanced brace matching.
    
    Handles nested braces like \\boxed{\\frac{1}{2}} correctly.
    Returns the last \\boxed{} content found (usually the final answer).
    """
    results = []
    search_start = 0
    
    while True:
        idx = text.find('\\boxed{', search_start)
        if idx == -1:
            break
        
        start = idx + len('\\boxed{')
        depth = 1
        end = start
        
        while end < len(text) and depth > 0:
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            end += 1
        
        if depth == 0:
            content = text[start:end-1].strip()
            if content:
                results.append(content)
        
        search_start = end
    
    # Return the last boxed content (usually the final answer)
    return results[-1] if results else None


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text, trying multiple patterns."""
    # First try to extract from \boxed{} with proper nested brace handling
    boxed_content = extract_boxed_content(text)
    if boxed_content:
        return normalize_answer(boxed_content)
    
    # Try specific answer patterns
    patterns = [
        # Implementation for multi-agent: handle explicit "Final Answer" forms first
        r'Final Answer[:\s]+\$?([+-]?\d+\.?\d*)',  # Final Answer: 42
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


def normalize_answer(answer: str) -> str:
    """Normalize answer to standardized format.
    
    IMPORTANT: This function should NOT strip backslashes when the answer contains
    LaTeX expressions. Only strip them for simple numeric answers.
    """
    answer = answer.strip()
    
    # Check if this looks like a LaTeX expression (contains common LaTeX commands)
    # Check both backslash-prefixed and non-backslash versions
    latex_indicators = ['\\frac', '\\sqrt', '\\text', '\\pi', '\\left', '\\right', '\\circ', '^\\circ',
                        'frac{', 'sqrt{', 'text{', '^circ']
    is_latex = any(x in answer for x in latex_indicators)
    
    # Also check if it contains coordinate pairs or tuples with mathematical content
    # e.g., (3, pi/2) or (3,frac{...})
    has_math_tuple = bool(re.search(r'\([^)]*(?:frac|sqrt|pi)[^)]*\)', answer, re.IGNORECASE))
    
    # Remove dollar signs (currency or math mode delimiters)
    answer = answer.replace('\\$', '').replace('$', '')
    answer = answer.replace('\\%', '').replace('%', '')
    
    if is_latex or has_math_tuple:
        # For LaTeX expressions, only do minimal cleanup
        # Remove spaces around operators but preserve structure (including commas!)
        answer = ' '.join(answer.split())  # Normalize whitespace
        return answer.lower()
    
    # For non-LaTeX answers, do aggressive normalization
    answer = answer.replace('\\', '')  # Remove backslashes for simple answers
    answer = answer.replace(',', '')
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
    """Enhanced answer checking with support for MATH500 formats"""
    if not predicted or not ground_truth:
        return False
    
    # Step 0: Quick exact match (before any normalization)
    if predicted.strip().lower() == ground_truth.strip().lower():
        return True
    
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
        
        # Normalize sqrt: \sqrt{2} and \sqrt2 and sqrt{2}
        s = re.sub(r'\\sqrt\s+(\d+)', r'\\sqrt{\1}', s)  # \sqrt 2 -> \sqrt{2}
        s = re.sub(r'\\sqrt([a-zA-Z0-9]+)(?![{])', r'\\sqrt{\1}', s)  # \sqrt2 -> \sqrt{2}
        s = re.sub(r'(?<!\\)sqrt\{', r'\\sqrt{', s)  # sqrt{ -> \sqrt{
        s = re.sub(r'(?<!\\)sqrt(\d+)', r'\\sqrt{\1}', s)  # sqrt13 -> \sqrt{13}
        
        # Normalize pi: \pi -> pi (remove backslash for consistency)
        s = s.replace(r'\pi', 'pi')
        
        # Remove LaTeX spacing commands
        s = s.replace(r'\!', '')
        s = s.replace(r'\,', '')
        s = s.replace(r'\:', '')
        s = s.replace(r'\;', '')
        s = s.replace(r'\ ', ' ')
        
        # Remove degree symbol variants - BEFORE removing backslashes
        s = re.sub(r'\^\s*\\?circ', '', s)  # ^circ, ^\circ, ^ \circ
        s = s.replace('\\circ', '')
        s = s.replace('circ', '')  # Also handle stripped version
        s = s.replace('degrees', '')
        s = s.replace('degree', '')
        
        # Remove dollar signs
        s = s.replace('\\$', '')
        s = s.replace('$', '')
        
        # Remove \text{} wrapper but keep content
        s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
        s = re.sub(r'text\{([^}]+)\}', r'\1', s)  # Also handle stripped version
        
        # Remove \left and \right
        s = s.replace(r'\left', '')
        s = s.replace(r'\right', '')
        s = s.replace('left', '')  # Also handle stripped version
        s = s.replace('right', '')
        
        # Normalize frac without backslash
        s = re.sub(r'(?<!\\)frac\{', r'\\frac{', s)  # frac{ -> \frac{
        
        # Normalize parentheses spacing
        s = re.sub(r'\(\s+', '(', s)
        s = re.sub(r'\s+\)', ')', s)
        s = re.sub(r',\s+', ',', s)
        
        # Remove spaces around operators
        s = re.sub(r'\s*([+\-*/=])\s*', r'\1', s)
        
        # Clean up general whitespace
        s = ' '.join(s.split())
        
        return s.strip().lower()
    
    pred_latex = normalize_latex(pred_normalized)
    gt_latex = normalize_latex(gt_normalized)
    
    if pred_latex == gt_latex:
        return True
    
    # Step 3: Numerical comparison - try both original and latex-normalized versions
    def try_float(s: str) -> float:
        """Try to convert string to float, handling various formats"""
        s = s.replace(',', '').replace(' ', '')
        # Remove degree symbols
        s = re.sub(r'\^\s*\\?circ', '', s)
        s = s.replace('circ', '').replace('degree', '').replace('degrees', '')
        return float(s)
    
    for pred_str, gt_str in [(pred_normalized, gt_normalized), (pred_latex, gt_latex)]:
        try:
            pred_num = try_float(pred_str)
            gt_num = try_float(gt_str)
            if abs(pred_num - gt_num) < 1e-3:
                return True
        except:
            pass
    
    # Step 4: SymPy symbolic comparison (for complex expressions)
    # Check for LaTeX math expressions in BOTH original and normalized forms
    has_latex_expr = any(x in predicted.lower() or x in ground_truth.lower() or 
                        x in pred_latex or x in gt_latex 
                        for x in ['frac', 'sqrt', 'pi', '\\pi'])
    
    if has_latex_expr:
        try:
            from sympy import sympify, simplify, expand, I, pi, sqrt, N
            
            def latex_to_sympy_expr(s: str) -> str:
                """Convert LaTeX to SymPy-parseable expression"""
                # Handle pi
                s = s.replace(r'\pi', '*pi')
                s = s.replace('pi', '*pi')
                s = re.sub(r'^\*pi', 'pi', s)
                s = re.sub(r'([0-9])\*\*pi', r'\1*pi', s)  # Fix double *
                
                # \frac{a}{b} -> (a)/(b)
                max_iter = 10
                for _ in range(max_iter):
                    match = re.search(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', s)
                    if match:
                        num, den = match.groups()
                        s = s[:match.start()] + f'(({num})/({den}))' + s[match.end():]
                    else:
                        break
                
                # Also handle frac without backslash
                for _ in range(max_iter):
                    match = re.search(r'(?<!\\)frac\{([^{}]*)\}\{([^{}]*)\}', s)
                    if match:
                        num, den = match.groups()
                        s = s[:match.start()] + f'(({num})/({den}))' + s[match.end():]
                    else:
                        break
                
                # \sqrt{a} -> sqrt(a) and sqrt{a} -> sqrt(a)
                s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
                s = re.sub(r'sqrt\{([^}]+)\}', r'sqrt(\1)', s)
                s = re.sub(r'sqrt(\d+)', r'sqrt(\1)', s)  # sqrt13 -> sqrt(13)
                
                # Add multiplication sign before sqrt when preceded by a number
                s = re.sub(r'(\d)sqrt\(', r'\1*sqrt(', s)  # 3sqrt(13) -> 3*sqrt(13)
                
                # Handle complex numbers: a+bi or a-bi
                s = s.replace('i', '*I')
                s = re.sub(r'([0-9])\*I', r'\1*I', s)
                s = re.sub(r'\+\*I', '+I', s)
                s = re.sub(r'\-\*I', '-I', s)
                
                # Clean up backslashes and spaces
                s = s.replace('\\', '')
                s = s.replace(' ', '')
                
                return s
            
            # Try symbolic comparison
            pred_sympy = latex_to_sympy_expr(pred_latex)
            gt_sympy = latex_to_sympy_expr(gt_latex)
            
            try:
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
                pass
            
            # Try numerical comparison for symbolic expressions
            # (e.g., predicted=10.816... vs ground_truth=3\sqrt{13})
            try:
                # If pred is a number and gt has sqrt/frac, evaluate gt
                pred_num = float(pred_normalized.replace(',', ''))
                gt_expr = sympify(gt_sympy, evaluate=True)
                gt_num = float(N(gt_expr))
                if abs(pred_num - gt_num) < 1e-3:
                    return True
            except:
                pass
            
            try:
                # Vice versa
                gt_num = float(gt_normalized.replace(',', ''))
                pred_expr = sympify(pred_sympy, evaluate=True)
                pred_num = float(N(pred_expr))
                if abs(pred_num - gt_num) < 1e-3:
                    return True
            except:
                pass
            
        except:
            pass  # Silently fail, continue to other checks
    
    # Step 5: Exact match after additional normalization (no substring matching)
    # BUGFIX: Removed dangerous substring matching that caused "pi" to match "frac{pi}{2}"
    # Only use exact match after removing common formatting variations
    def additional_normalize(s: str) -> str:
        """Additional normalization for edge cases"""
        s = s.lower().strip()
        # Remove common wrappers
        s = s.replace('text{', '').replace('}', '').replace('{', '')
        s = s.replace('left', '').replace('right', '')
        s = s.replace('(', '').replace(')', '')
        s = s.replace(' ', '')
        return s
    
    pred_extra = additional_normalize(pred_latex)
    gt_extra = additional_normalize(gt_latex)
    
    if pred_extra == gt_extra:
        return True
    
    return False


# ============================================================================
# DATASET LOADING FUNCTIONS
# ============================================================================

def extract_question_and_answer(example: dict, dataset_name: str) -> tuple:
    """Extract question and ground truth answer from dataset example"""
    if dataset_name == "gsm8k":
        question = example['question']
        ground_truth = example['answer'].split('####')[-1].strip()
    elif dataset_name == "math" or dataset_name == "math500":
        question = example['problem']
        # For MATH500, prefer the 'answer' field if available (clean answer)
        if 'answer' in example:
            ground_truth = example['answer'].strip()
        else:
            # Fallback to extracting from solution with proper nested brace handling
            solution = example['solution']
            boxed_content = extract_boxed_content(solution)
            if boxed_content:
                ground_truth = boxed_content
            else:
                numbers = re.findall(r'[+-]?\d+\.?\d*', solution)
                ground_truth = numbers[-1] if numbers else solution
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return question, ground_truth


