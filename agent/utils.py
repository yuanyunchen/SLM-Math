"""
Agent utility functions for prompt formatting, parsing, and generation.
"""

import re
import torch
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
from models.inference import StopOnBoxedAnswer


def format_prompt_solver(question: str, checker_feedback: str = "", dataset_name: str = "") -> str:
    """
    Format prompt for solver agent with feedback support (OPTIMIZED VERSION).
    
    OPTIMIZATIONS:
    - Clearer instruction to reconsider solution when feedback is given
    - Emphasis on showing calculation steps
    - Explicit request for boxed answer format
    
    Args:
        question: The math problem to solve
        checker_feedback: Optional feedback from checker (empty for first attempt)
        dataset_name: Dataset name (for compatibility)
    
    Returns:
        Formatted prompt string
    """
    if checker_feedback and checker_feedback.strip():
        return f"""{question}

Your previous solution had issues. Feedback: {checker_feedback}

Please solve the problem again from scratch. Show clear calculation steps and put your final answer in \\boxed{{answer}} format."""
    else:
        # First iteration uses standard prompt
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.prompt_utils import format_prompt_standard
        return format_prompt_standard(question, dataset_name)


def extract_solver_cot(solver_response: str) -> str:
    """
    Extract Chain-of-Thought reasoning from solver response.
    Removes code blocks and outputs, keeps only textual reasoning.
    """
    # Remove code blocks
    cot = re.sub(r'```python.*?```', '', solver_response, flags=re.DOTALL)
    cot = re.sub(r'```output.*?```', '', cot, flags=re.DOTALL)
    cot = re.sub(r'```.*?```', '', cot, flags=re.DOTALL)
    
    # Remove final boxed answer
    cot = re.sub(r'\\boxed\{[^}]+\}', '[final answer]', cot)
    
    # Clean up whitespace
    lines = [line.strip() for line in cot.split('\n') if line.strip()]
    cot = ' '.join(lines)
    
    # Limit length to ~400 characters
    if len(cot) > 400:
        sentences = re.split(r'[.!?]\s+', cot)
        shortened = ""
        for sent in sentences:
            if len(shortened) + len(sent) < 400:
                shortened += sent + ". "
            else:
                break
        cot = shortened.strip()
    
    return cot.strip()


def format_prompt_checker(question: str, solver_response: str, dataset_name: str = "") -> str:
    """
    Format prompt for checker agent (V6 - Stricter verification).
    
    OPTIMIZATION: More specific instructions to force clear verdict output.
    - Asks model to verify the numerical answer
    - Requires explicit INCORRECT or CORRECT at the start
    
    Args:
        question: The original math problem
        solver_response: Solver's full response
        dataset_name: Dataset name (for compatibility)
    
    Returns:
        Formatted checker prompt
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.prompt_utils import extract_answer
    
    # Extract solver answer
    solver_answer = extract_answer(solver_response)
    if not solver_answer:
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solver_response)
        if boxed_match:
            solver_answer = boxed_match.group(1).strip()
        else:
            solver_answer = "No answer found"
    
    # Checker prompt V7 - emphasize verification, not re-solving
    # Truncate question to reduce model's tendency to re-solve
    short_question = question[:150] + "..." if len(question) > 150 else question
    
    return f"""Quick check: For "{short_question}"

Proposed answer: {solver_answer}

Does this look right? Say CORRECT or INCORRECT."""


def parse_checker_verdict(response: str) -> str:
    """
    Parse checker verdict from response (BALANCED VERSION V5).
    
    OPTIMIZATION V5: 
    - Detect when model is re-solving instead of verifying
    - Handle short affirmative responses better
    - Better balance for degraded case prevention
    
    Returns:
        "CORRECT" or "INCORRECT"
    """
    if not response or len(response.strip()) < 2:
        return "INCORRECT"
    
    upper = response.strip().upper()
    original = response.strip()
    
    # Primary: Check if response STARTS with verdict
    if upper.startswith('CORRECT'):
        if not re.search(r'^CORRECT\s*(BUT|HOWEVER|IS\s+NOT|WRONG)', upper):
            return "CORRECT"
    if upper.startswith('INCORRECT'):
        return "INCORRECT"
    
    # OPTIMIZATION: Detect "re-solving" pattern
    # If checker starts with "To determine...", "Let's...", "First...", it's re-solving
    # These responses are unreliable - default to trusting the original answer
    resolving_patterns = [
        r'^TO\s+DETERMINE',
        r'^LET\'?S',
        r'^FIRST',
        r'^WE\s+NEED',
        r'^STEP\s+1',
        r'^TO\s+CHECK',
        r'^TO\s+VERIFY',
    ]
    for pattern in resolving_patterns:
        if re.search(pattern, upper):
            # Model is re-solving, not verifying - unreliable
            # Check if it eventually says something clear
            if re.search(r'\bINCORRECT\b|\bWRONG\b', upper[-100:]):
                return "INCORRECT"
            if re.search(r'\bCORRECT\b', upper[-50:]):
                return "CORRECT"
            # Otherwise, trust the original answer (lean CORRECT)
            return "CORRECT"
    
    # Secondary: Look for INCORRECT indicators
    incorrect_patterns = [
        r'\bINCORRECT\b',
        r'\bWRONG\b',
        r'\bERROR\b',
        r'\bINVALID\b',
        r'\bMISTAKE\b',
        r'NOT\s+CORRECT',
        r'IS\s+NOT\s+RIGHT',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, upper):
            return "INCORRECT"
    
    # Third: Look for CORRECT indicators
    correct_patterns = [
        r'\bCORRECT\b',
        r'\bRIGHT\b',
        r'^YES\b',
        r'LOOKS?\s+(RIGHT|CORRECT|GOOD)',
        r'IS\s+CORRECT',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, upper):
            if not re.search(r'NOT\s+' + pattern, upper):
                return "CORRECT"
    
    # Fourth: Short response handling
    if len(original) < 30:
        # Very short - likely partial output or just agreement
        # Lean toward CORRECT for short unclear responses
        return "CORRECT"
    
    # Long unclear response - default INCORRECT
    return "INCORRECT"


def parse_checker_tip(response: str) -> str:
    """
    Extract feedback/tip from checker response when verdict is INCORRECT.
    
    Returns:
        Tip text or empty string
    """
    # Find VERDICT line
    verdict_match = re.search(r'VERDICT\s*:\s*(?:CORRECT|INCORRECT)', response, re.IGNORECASE)
    if verdict_match:
        tip_end = verdict_match.start()
        tip = response[:tip_end].strip()
        
        # Clean up
        tip = re.sub(r'^(tip|feedback|suggestion|hint|note|reasoning)[:\s]*', '', tip, flags=re.IGNORECASE)
        tip = re.sub(r'\n\s*\n', ' ', tip)
        tip = tip.strip()
        
        # Limit length
        if len(tip) > 300:
            sentences = re.split(r'[.!?]\s+', tip)
            if len(sentences) > 1:
                tip = '. '.join(sentences[:2]) + '.'
            else:
                tip = tip[:300]
        
        return tip
    
    return ""


class StopAfterCheckerConclusion(StoppingCriteria):
    """
    Stop checker generation once VERDICT is found (OPTIMIZED V7).
    
    OPTIMIZATIONS V7:
    - Simple prompt = allow early verdict (after 5 chars)
    - Stop on clear CORRECT/INCORRECT
    - Longer max to capture verdict if model is verbose
    """
    
    def __init__(self, tokenizer, prompt_token_len: int):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_token_len:]
        if generated_ids.numel() == 0:
            return False
        
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        stripped = text.strip().upper()
        
        # Minimum 5 chars to avoid stopping on partial tokens
        if len(stripped) < 5:
            return False
        
        # Stop if output starts with CORRECT or INCORRECT
        if stripped.startswith('CORRECT') or stripped.startswith('INCORRECT'):
            return True
        
        # Stop on any clear verdict pattern in first 50 chars
        first_part = stripped[:50]
        if re.search(r'\b(CORRECT|INCORRECT)\b', first_part):
            return True
        
        # Stop if generating too much (200 chars max to give model more room)
        if len(text) > 200:
            return True
        
        return False


def generate_response_checker(model, tokenizer, prompt: str, detailed: bool = False):
    """
    Generate response for checker (OPTIMIZED V4).
    
    OPTIMIZATIONS V4:
    - Use dedicated checker generation, NOT StopOnBoxedAnswer
    - Use StopAfterCheckerConclusion for proper stopping
    - Shorter max tokens since checker only needs to verify
    """
    import torch
    from transformers import TextStreamer
    
    # Apply chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    prompt_length = inputs['input_ids'].shape[1]
    
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None
    
    # Use checker-specific stopping criteria
    stopping_criteria = StoppingCriteriaList([
        StopAfterCheckerConclusion(tokenizer, prompt_length)
    ])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Checker needs less tokens
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            stopping_criteria=stopping_criteria,
            streamer=streamer
        )
    
    response_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    return response

