"""
Solver-Checker-Summarizer Multi-Agent Workflow (Chat Mode)
带总结层的Solver-Checker工作流 - Chat模式

ARCHITECTURE:
Similar to stateless but maintains conversation history with [ROLE] tags

CHAT MODE FEATURES:
- Maintains full conversation history
- Uses [SOLVER], [CHECKER], [SUMMARIZER] tags
- Better for larger models (>2B parameters)
- Warning: May cause hallucination in small models

适用场景：
- 大模型 (>=2B)
- 需要上下文连贯性
- 实验性功能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import torch
from agent.unified_config import FIRST_ROUND_SOLVER_CONFIG, SUBSEQUENT_ROUND_CONFIG


def build_conversation_text(conversation_history: List[Dict]) -> str:
    """
    Build conversation text from history with [ROLE] tags
    
    Args:
        conversation_history: List of message dicts with 'role' and 'content'
    
    Returns:
        Formatted conversation text
    """
    conversation_text = ""
    for msg in conversation_history:
        role_label = msg['role'].upper()
        conversation_text += f"[{role_label}]: {msg['content']}\n\n"
    return conversation_text


def verify_answer_by_resolve(
    model,
    tokenizer,
    question: str,
    solver_answer: str,
    detailed: bool = False,
    dataset_name: str = ""
) -> tuple:
    """
    Verify answer by having the model independently solve the problem.
    
    This approach is more reliable than asking "is X correct?" because:
    1. Models tend to be agreeable when asked to verify
    2. Independent solving forces actual computation
    3. We can compare numerical answers directly
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        question: Original problem
        solver_answer: Answer to verify
        detailed: Show detailed output
        dataset_name: Dataset name
    
    Returns:
        Tuple of (verdict: str, response: str)
    """
    from utils.prompt_utils import extract_answer, check_answer
    
    # If no answer to verify, return INCORRECT
    if not solver_answer or solver_answer.strip() == '':
        return "INCORRECT", "No answer provided to verify."
    
    # Build a short verification prompt - ask for quick calculation
    verify_prompt = f"""Quickly verify: {question}
Answer in one line with \\boxed{{}}."""
    
    # Apply chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": verify_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = verify_prompt
    
    # Generate short response
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Increased from 256 to avoid truncation
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # Extract checker's answer
    checker_answer = extract_answer(response)
    
    if detailed:
        print(f"  Checker re-solve: {checker_answer}")
        print(f"  Solver answer: {solver_answer}")
    
    # Compare answers
    if checker_answer and solver_answer:
        # Use the existing check_answer function for robust comparison
        if check_answer(solver_answer, checker_answer):
            verdict = "CORRECT"
        else:
            verdict = "INCORRECT"
    elif not checker_answer:
        # If checker couldn't extract answer, use simple string matching
        if solver_answer in response or str(solver_answer) in response:
            verdict = "CORRECT"
        else:
            # Default to trusting solver if checker fails
            verdict = "CORRECT"
    else:
        verdict = "INCORRECT"
    
    if detailed:
        print(f"  Verdict: {verdict}")
    
    return verdict, response


def is_repetitive_text_chat(text: str, threshold: float = 0.7) -> bool:
    """Check if text contains high repetition (garbage output detection)"""
    import re
    
    if len(text) < 50:
        return False
    
    words = re.findall(r'\w+', text.lower())
    if len(words) < 10:
        return False
    
    # Check unique word ratio
    unique_ratio = len(set(words)) / len(words)
    
    # Check for repeated sequences
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    max_repetition = max(word_counts.values()) if word_counts else 0
    max_rep_ratio = max_repetition / len(words) if words else 0
    
    return unique_ratio < (1 - threshold) or max_rep_ratio > 0.5


def parse_checker_verdict_strict(response: str) -> str:
    """
    Parse checker verdict with STRICT mode (defaults to INCORRECT).
    
    OPTIMIZED V2: More robust detection for small models.
    - Looks for explicit INCORRECT/WRONG/ERROR first
    - Only accepts CORRECT if very clearly stated
    - Handles common response patterns from Qwen models
    
    Returns:
        "CORRECT" or "INCORRECT"
    """
    import re
    
    if not response or len(response.strip()) < 3:
        return "INCORRECT"
    
    upper = response.strip().upper()
    
    # Check for explicit "INCORRECT" first (highest priority)
    if re.search(r'\bINCORRECT\b', upper):
        return "INCORRECT"
    
    # Check for error/wrong indicators
    error_patterns = [
        r'\bWRONG\b',
        r'\bERROR\b',
        r'\bMISTAKE\b',
        r'\bNOT\s+CORRECT\b',
        r'\bFALSE\b',
        r'\bINVALID\b',
        r'\bNO[,.\s]',  # "No," or "No." at start
        r'^NO\s',
    ]
    for pattern in error_patterns:
        if re.search(pattern, upper):
            # Make sure it's not negated
            if not re.search(r'(NO|NOT)\s+(ERROR|MISTAKE|WRONG)', upper):
                return "INCORRECT"
    
    # Check if response STARTS with verdict (expected for simple prompt)
    if upper.startswith('CORRECT'):
        # Make sure it's not "CORRECT" followed by negative context
        if not re.search(r'^CORRECT\s*(BUT|HOWEVER|IS\s+NOT|WRONG)', upper):
            return "CORRECT"
    
    # Look for explicit verdict patterns
    verdict_patterns = [
        r'VERDICT\s*:\s*CORRECT\b',
        r'ANSWER\s*:\s*CORRECT\b',
        r'IS\s+CORRECT\b',
        r'\bYES\s*[,.]?\s*(?:THE\s+)?(?:ANSWER\s+)?(?:IS\s+)?CORRECT\b',
    ]
    for pattern in verdict_patterns:
        if re.search(pattern, upper):
            return "CORRECT"
    
    # Check last line for standalone CORRECT
    lines = [line.strip() for line in upper.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        if last_line == 'CORRECT' or re.match(r'^CORRECT[.!]?$', last_line):
            return "CORRECT"
    
    # Default to INCORRECT to force iteration (conservative)
    return "INCORRECT"


def parse_checker_verdict_balanced(response: str) -> str:
    """
    Parse checker verdict with BALANCED mode.
    
    Not too strict (avoid rejecting correct answers)
    Not too lenient (avoid accepting wrong answers)
    
    Returns:
        "CORRECT" or "INCORRECT"
    """
    import re
    upper = response.upper()
    
    # First, check for clear verdict patterns
    # INCORRECT patterns (check first - if model explicitly rejects, trust it)
    if re.search(r'\bINCORRECT\b', upper):
        return "INCORRECT"
    
    if re.search(r'\b(WRONG|ERROR|MISTAKE)\b', upper):
        # Check it's not "no error" or similar
        if not re.search(r'(NO|NOT|WITHOUT)\s+(WRONG|ERROR|MISTAKE)', upper):
            return "INCORRECT"
    
    # CORRECT patterns
    if re.search(r'\bCORRECT\b', upper):
        # Check it's not "not correct" or "incorrect"
        if not re.search(r'(NOT|IN)\s*CORRECT', upper):
            return "CORRECT"
    
    # Check for other positive indicators
    positive_patterns = [
        r'\bYES\b',
        r'\bRIGHT\b',
        r'\bVALID\b',
        r'\bACCURATE\b',
    ]
    for pattern in positive_patterns:
        if re.search(pattern, upper):
            return "CORRECT"
    
    # Check for negative indicators
    negative_patterns = [
        r'\bNO\b(?!\s+PROBLEM|\s+ISSUE)',  # "NO" but not "NO PROBLEM"
        r'\bFALSE\b',
        r'\bINVALID\b',
    ]
    for pattern in negative_patterns:
        if re.search(pattern, upper):
            return "INCORRECT"
    
    # Default: trust the solver if no clear rejection
    # (Small models often generate ambiguous responses when answer is correct)
    return "CORRECT"


def generate_chat_response(
    model,
    tokenizer,
    conversation_history: List[Dict],
    next_role: str,
    role_instruction: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    detailed: bool = False,
    do_sample: bool = None,
    top_p: float = None
) -> str:
    """
    Generate response for a role in chat mode (with garbage detection)
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        conversation_history: Conversation history
        next_role: Role that will speak next (e.g., "SOLVER")
        role_instruction: Optional instruction for the role
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        detailed: Show detailed output
        do_sample: Override do_sample (None = auto from temperature)
        top_p: Override top_p (None = use config defaults)
    
    Returns:
        Generated response text
    """
    # Build conversation text
    conversation_text = build_conversation_text(conversation_history)
    
    # Add next role tag with optional instruction
    if role_instruction:
        conversation_text += f"[{next_role.upper()}]: {role_instruction}\n\n"
    else:
        conversation_text += f"[{next_role.upper()}]: "
    
    # Tokenize
    inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Adjust parameters based on role (only if not explicitly provided)
    if next_role.upper() == "SUMMARIZER":
        max_new_tokens = min(max_new_tokens, 150)
        temperature = 0.3
        repetition_penalty = 1.3
    elif next_role.upper() == "CHECKER":
        # Checker should be deterministic and strict
        max_new_tokens = min(max_new_tokens, 150)
        temperature = 0.0  # Deterministic
        repetition_penalty = 1.2
    else:
        repetition_penalty = 1.2
    
    # Auto-determine do_sample from temperature if not specified
    if do_sample is None:
        do_sample = temperature > 0
    # Use config default for top_p if not specified
    if top_p is None:
        top_p = FIRST_ROUND_SOLVER_CONFIG['top_p'] if not do_sample else SUBSEQUENT_ROUND_CONFIG['top_p']
    
    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(conversation_text):].strip()
        
        # Detect garbage for SUMMARIZER - use simple fallback instead
        if next_role.upper() == "SUMMARIZER" and (len(response) < 20 or is_repetitive_text_chat(response)):
            if detailed:
                print(f"[Warning] {next_role} response appears invalid, using simple fallback")
            response = "Please review the solution carefully."
        
        if detailed:
            print(response[:200] + ("..." if len(response) > 200 else ""))
        
        return response
        
    except Exception as e:
        if detailed:
            print(f"[Error] Generation failed: {e}")
        return ""


def run_solver_checker_summarizer_chat_workflow(
    question: str,
    ground_truth: str,
    model,  # Shared model for all roles
    tokenizer,
    max_iterations: int = 5,
    detailed: bool = False,
    dataset_name: str = ""
) -> Dict:
    """
    Run Solver-Checker-Summarizer workflow (Chat Mode)
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        model: Shared model for all roles
        tokenizer: Tokenizer
        max_iterations: Maximum iterations
        detailed: Verbose output
        dataset_name: Dataset name
    
    Returns:
        Dict with workflow results
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-CHECKER-SUMMARIZER WORKFLOW (Chat)")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Mode: Chat with conversation history")
        print(f"{'='*80}\n")
    
    # Conversation history
    conversation_history = []
    
    iterations = []
    solver_responses = []
    solver_answers = []
    checker_responses = []
    checker_verdicts = []
    
    predicted_answer = None
    final_verdict = None
    
    # Iterative loop
    for iteration_num in range(1, max_iterations + 1):
        if detailed:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration_num}/{max_iterations}")
            print(f"{'='*80}")
        
        # ========== SOLVER TURN ==========
        if detailed:
            print(f"\n[Solver Turn]")
        
        if iteration_num == 1:
            # First turn: add question (use standard format for consistency)
            standard_prompt = format_prompt_standard(question, dataset_name)
            conversation_history.append({
                'role': 'user',
                'content': standard_prompt
            })
            # Use unified config for first round
            solver_temperature = FIRST_ROUND_SOLVER_CONFIG['temperature']
            solver_do_sample = FIRST_ROUND_SOLVER_CONFIG['do_sample']
            solver_top_p = FIRST_ROUND_SOLVER_CONFIG['top_p']
        else:
            # Subsequent rounds use agent's own config
            solver_temperature = SUBSEQUENT_ROUND_CONFIG['temperature']
            solver_do_sample = SUBSEQUENT_ROUND_CONFIG['do_sample']
            solver_top_p = SUBSEQUENT_ROUND_CONFIG['top_p']
        
        # Generate solver response
        solver_response = generate_chat_response(
            model,
            tokenizer,
            conversation_history,
            next_role="SOLVER",
            max_new_tokens=512,
            temperature=solver_temperature,
            detailed=detailed,
            do_sample=solver_do_sample,
            top_p=solver_top_p
        )
        
        solver_answer = extract_answer(solver_response)
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        # Add to conversation
        conversation_history.append({
            'role': 'solver',
            'content': solver_response
        })
        
        if detailed:
            print(f"\n[Solver Answer]: {solver_answer}")
        
        # ========== SUMMARIZER (Solver → Checker) ==========
        # OPTIMIZED V2: Skip verbose summarizer step, go directly to verification
        # The verification function handles this independently
        if detailed:
            print(f"\n[Summarizer: Answer = {solver_answer}]")
        
        # ========== CHECKER TURN ==========
        if detailed:
            print(f"\n[Checker Turn]")
        
        # OPTIMIZED V2: Use independent re-solve verification
        # Instead of asking "is X correct?", we verify by asking model to solve independently
        # and comparing answers
        checker_verdict, checker_response = verify_answer_by_resolve(
            model,
            tokenizer,
            question,
            solver_answer,
            detailed=detailed,
            dataset_name=dataset_name
        )
        
        checker_responses.append(checker_response)
        
        checker_verdicts.append(checker_verdict)
        
        # Add to conversation
        conversation_history.append({
            'role': 'checker',
            'content': checker_response
        })
        
        if detailed:
            print(f"\n[Checker Verdict]: {checker_verdict}")
        
        # ========== SUMMARIZER (Checker → Solver) ==========
        if checker_verdict != "CORRECT":
            if detailed:
                print(f"\n[Summarizer: Preparing feedback for Solver]")
            
            # OPTIMIZED V2: Extract checker's answer to provide more specific feedback
            from utils.prompt_utils import extract_answer as extract_ans
            checker_answer = extract_ans(checker_response)
            
            if checker_answer and checker_answer != solver_answer:
                # Specific feedback when checker found different answer
                checker_summary = f"Verification found a different answer ({checker_answer}). Your answer was {solver_answer}. Please re-check your calculations carefully."
            else:
                # Generic feedback when no clear checker answer
                checker_summary = f"Your answer {solver_answer} could not be verified. Please solve the problem again step by step, showing all calculations."
            
            if detailed:
                print(checker_summary[:200] + ("..." if len(checker_summary) > 200 else ""))
            
            # Add to conversation
            conversation_history.append({
                'role': 'summarizer',
                'content': checker_summary
            })
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "conversation_length": len(conversation_history)
        }
        
        # Check actual correctness
        is_actually_correct = False
        if solver_answer:
            is_actually_correct = check_answer(solver_answer, ground_truth)
        iteration_data["is_actually_correct"] = is_actually_correct
        
        iterations.append(iteration_data)
        
        # Decision logic
        if checker_verdict == "CORRECT":
            if solver_answer:
                predicted_answer = solver_answer
                final_verdict = "CORRECT"
                if detailed:
                    print(f"\n✓ Checker confirmed CORRECT, breaking")
                break
        
        if iteration_num >= max_iterations:
            if detailed:
                print(f"\nReached max iterations ({max_iterations})")
            break
    
    # If no CORRECT verdict, use last valid answer
    if predicted_answer is None:
        for i in range(len(iterations) - 1, -1, -1):
            iter_answer = iterations[i]['solver_answer']
            if iter_answer and iter_answer.strip():
                predicted_answer = iter_answer
                final_verdict = f"LAST_VALID_ITER_{iterations[i]['iteration']}"
                break
        
        if predicted_answer is None:
            final_verdict = "NO_VALID_ANSWER"
    
    # Check final correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    # Determine case type
    first_answer = solver_answers[0] if solver_answers else None
    first_correct = check_answer(first_answer, ground_truth) if first_answer else False
    
    case_type = None
    if first_correct and final_correct:
        case_type = "FIRST_TRY_SUCCESS"
    elif not first_correct and final_correct:
        case_type = "IMPROVED"
    elif first_correct and not final_correct:
        case_type = "DEGRADED"
    elif not first_correct and not final_correct:
        case_type = "FAILED"
    else:
        case_type = "OTHER"
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "final_correct": final_correct,
        "final_verdict": final_verdict,
        "total_iterations": len(iterations),
        "iterations": iterations,
        "solver_answers": solver_answers,
        "checker_verdicts": checker_verdicts,
        "first_answer": first_answer,
        "first_correct": first_correct,
        "case_type": case_type,
        "conversation_history": conversation_history,
        "config": {
            "use_summarizer": True,
            "mode": "chat"
        }
    }


# For testing
if __name__ == "__main__":
    print("Solver-Checker-Summarizer (Chat) - Quick Test")
    print("=" * 80)
    
    test_question = "A bakery sells 3 types of bread. Type A costs $2, Type B costs $3, and Type C costs $5. If someone buys 2 of Type A, 1 of Type B, and 1 of Type C, how much do they spend?"
    test_ground_truth = "12"
    
    try:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        from models.inference import load_model
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning Solver-Checker-Summarizer Chat workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_solver_checker_summarizer_chat_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            model=model,
            tokenizer=tokenizer,
            max_iterations=2,
            detailed=True,
            dataset_name="gsm8k"
        )
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Expected: {result['ground_truth']}")
        print(f"Correct: {result['final_correct']}")
        print(f"Case Type: {result['case_type']}")
        print(f"Iterations: {result['total_iterations']}")
        print(f"Conversation Length: {len(result['conversation_history'])} messages")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

