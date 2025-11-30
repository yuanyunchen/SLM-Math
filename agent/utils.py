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
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY,
    CHECKER_MAX_TOKENS, CHECKER_TEMPERATURE, CHECKER_TOP_P, CHECKER_REPETITION_PENALTY
)


def format_prompt_solver(question: str, checker_feedback: str = "", dataset_name: str = "") -> str:
    """
    Format prompt for solver agent (with feedback support).
    
    Args:
        question: The math problem to solve
        checker_feedback: Optional feedback from checker (empty for first attempt)
        dataset_name: Dataset name (for compatibility)
    
    Returns:
        Formatted prompt string
    """
    if checker_feedback and checker_feedback.strip():
        return f"""{question}

Previous attempt feedback: {checker_feedback}

Please reconsider the problem with the feedback in mind. Reason step by step, and put your final answer within \\boxed{{}}."""
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
    Format prompt for checker agent.
    
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
    
    # Extract solver's reasoning
    solver_cot = extract_solver_cot(solver_response)
    
    return f"""You are a math checker. Review the following solution carefully.

Problem: {question}

Solver's reasoning: {solver_cot}

Solver's final answer: {solver_answer}

Your task:
1. Extract all numbers from the problem
2. Verify each calculation step
3. Check if the logic is correct
4. Recalculate the answer independently if needed

Think step by step:
- Are the numbers from the problem used correctly?
- Is the mathematical approach valid?
- Are there any calculation errors?
- Does the final answer match your calculation?

After your analysis, respond with ONE word only:
- CORRECT (if solution is right)
- INCORRECT (if you find any error)

You must choose one. If unsure, carefully recalculate and decide.

Your verdict:"""


def parse_checker_verdict(response: str) -> str:
    """
    Parse checker verdict from response.
    Only accepts explicit verdict patterns, not casual mentions.
    
    Returns:
        "CORRECT" or "INCORRECT" (defaults to INCORRECT if unclear)
    """
    upper = response.upper()
    
    # Try explicit "YOUR VERDICT: XXX" or "VERDICT: XXX" pattern (most specific)
    match = re.search(r'(YOUR\s+)?VERDICT\s*:\s*(CORRECT|INCORRECT)', upper)
    if match:
        return match.group(2).upper()
    
    # Look for verdict at END of response (after substantial reasoning)
    # Must be on its own line or after conclusion markers
    lines = upper.strip().split('\n')
    if len(lines) > 0:
        last_line = lines[-1].strip()
        # Check if last line is just the verdict
        if last_line in ['CORRECT', 'INCORRECT']:
            return last_line
        # Check if last line is "VERDICT: XXX"
        if re.match(r'^(YOUR\s+)?(VERDICT|CONCLUSION)\s*:\s*(CORRECT|INCORRECT)\s*$', last_line):
            verdict_match = re.search(r'(CORRECT|INCORRECT)', last_line)
            if verdict_match:
                return verdict_match.group(1)
    
    # Look for standalone CORRECT or INCORRECT in the response
    if 'INCORRECT' in upper:
        return "INCORRECT"
    if 'CORRECT' in upper:
        return "CORRECT"
    
    # If no clear verdict found, default to INCORRECT (conservative)
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
    """Stop checker generation once VERDICT is found"""
    
    def __init__(self, tokenizer, prompt_token_len: int):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_token_len:]
        if generated_ids.numel() == 0:
            return False
        
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        upper = text.upper()
        
        # Only stop if we have substantial reasoning AND then a verdict
        # Require at least 50 chars before looking for verdict
        if len(text.strip()) < 50:
            return False
        
        # Check for explicit "Your verdict:" pattern followed by verdict
        verdict_pattern = r'(YOUR VERDICT|VERDICT)\s*:\s*(CORRECT|INCORRECT)'
        if re.search(verdict_pattern, upper):
            return True
        
        # Check for standalone verdict at the END after reasoning
        # Must have newline or "verdict:" before it
        if len(text.strip()) > 100:
            if re.search(r'(VERDICT|CONCLUSION)\s*:\s*(CORRECT|INCORRECT)(\s|$)', upper):
                return True
        
        # Stop if too long
        if len(text) > 400:
            return True
        
        return False


def generate_response_checker(model, tokenizer, prompt: str, detailed: bool = False):
    """Generate response for checker with reasoning space"""
    import torch
    
    # Check if model is an inference engine
    if hasattr(model, 'generate_single'):
        # Using inference engine (from load_inference_engine_wrapper)
        response = model.generate_single(
            prompt,
            max_new_tokens=CHECKER_MAX_TOKENS,
            temperature=CHECKER_TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_p=CHECKER_TOP_P,
            repetition_penalty=CHECKER_REPETITION_PENALTY,
            detailed=detailed
        )
        return response
    
    # Using standard model (from load_model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    prompt_length = inputs['input_ids'].shape[1]
    
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None
    
    stopping_criteria = StoppingCriteriaList([StopAfterCheckerConclusion(tokenizer, prompt_length)])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CHECKER_MAX_TOKENS,
            temperature=CHECKER_TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_p=CHECKER_TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=CHECKER_REPETITION_PENALTY,
            stopping_criteria=stopping_criteria,
            streamer=streamer
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response

