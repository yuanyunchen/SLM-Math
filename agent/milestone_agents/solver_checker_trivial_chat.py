"""
Solver-Checker Trivial Chat Multi-Agent Workflow
Simple chat implementation using text concatenation and [ROLE]: tags.

TRIVIAL CHAT MODE - straightforward concatenation approach
- Uses [SOLVER]: and [CHECKER]: tags.
- Does not use a chat template.
- No KV cache optimizations.
- Smaller models may hallucinate easily.

Recommendation: use the newer solver_checker_chat.py (optimized) or solver_checker_stateless.py.
"""

from typing import Dict, List, Tuple
import re
import torch
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY,
    CHECKER_MAX_TOKENS, CHECKER_TEMPERATURE
)


class StopOnBoxedAnswerOrVerdict(StoppingCriteria):
    """Stop generation when boxed answer is found or checker verdict is given"""
    
    def __init__(self, tokenizer, prompt_token_len: int, role: str = "solver"):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len
        self.role = role.lower()
    
    def _has_boxed_answer(self, text: str) -> bool:
        """Check if text contains a complete \boxed{} with balanced braces."""
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return False
        
        after_boxed = text[idx + 7:]
        brace_count = 1
        
        for char in after_boxed:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return True
        
        return False
    
    def _has_verdict(self, text: str) -> bool:
        """Check if checker has given a verdict"""
        upper = text.upper()
        # Check for explicit verdict patterns
        if re.search(r'VERDICT\s*:\s*(CORRECT|INCORRECT|UNCLEAR)', upper):
            return True
        # Check for standalone verdict words
        if re.search(r'(^|\n|\s)(CORRECT|INCORRECT|UNCLEAR)(\s|$|\n)', upper):
            return True
        return False
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_token_len:]
        if generated_ids.numel() == 0:
            return False
        
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if self.role == "solver":
            return self._has_boxed_answer(text)
        else:  # checker
            return self._has_verdict(text) or len(text) > 200


def format_chat_prompt(conversation_history: List[Dict[str, str]], current_role: str, question: str, dataset_name: str = "") -> str:
    """
    Format conversation history into a CHAT prompt with [ROLE] tags.
    
    Args:
        conversation_history: List of messages with 'role' and 'content'
        current_role: 'solver' or 'checker'
        question: Original math problem
        dataset_name: Dataset name
    
    Returns:
        Formatted prompt string with conversation history
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.prompt_utils import format_prompt_standard
    
    # Build conversation context
    conversation_text = ""
    
    # Add system instruction for first turn
    if len(conversation_history) == 0:
        if current_role == "solver":
            # First turn: solver gets standard prompt
            return format_prompt_standard(question, dataset_name)
        else:
            # Shouldn't happen, checker always comes after solver
            return ""
    
    # Build conversation history with [ROLE]: tags
    for msg in conversation_history:
        role_label = msg['role'].upper()
        content = msg['content']
        conversation_text += f"[{role_label}]: {content}\n\n"
    
    # Add current turn instruction
    if current_role == "solver":
        conversation_text += f"[SOLVER]: "
    else:  # checker
        conversation_text += f"[CHECKER]: "
    
    return conversation_text


def format_checker_feedback_prompt(conversation_history: List[Dict[str, str]], question: str) -> str:
    """
    Format prompt for checker to provide detailed feedback.
    
    Args:
        conversation_history: Full conversation history
        question: Original math problem
    
    Returns:
        Formatted checker prompt
    """
    # Get last solver response
    last_solver_msg = None
    for msg in reversed(conversation_history):
        if msg['role'] == 'solver':
            last_solver_msg = msg['content']
            break
    
    if not last_solver_msg:
        return ""
    
    # Extract answer and reasoning from solver response
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.prompt_utils import extract_answer
    from agent.utils import extract_solver_cot
    
    solver_answer = extract_answer(last_solver_msg)
    solver_cot = extract_solver_cot(last_solver_msg)
    
    # Build conversation context
    conversation_text = ""
    for msg in conversation_history:
        role_label = msg['role'].upper()
        content = msg['content']
        conversation_text += f"[{role_label}]: {content}\n\n"
    
    # Add checker instruction
    checker_instruction = f"""Problem: {question}

Review the solver's solution above. If it's incorrect, provide specific feedback in bullet points:
- Point out what's wrong
- Suggest what to check or reconsider
- Be concise and actionable

If correct, respond: VERDICT: CORRECT
If incorrect, respond with feedback, then: VERDICT: INCORRECT
If unclear, respond: VERDICT: UNCLEAR

[CHECKER]: """
    
    return conversation_text + checker_instruction


def parse_checker_verdict_from_chat(response: str) -> Tuple[str, str]:
    """
    Parse checker verdict and feedback from chat response.
    
    Returns:
        Tuple of (verdict, feedback)
        verdict: "CORRECT" or "INCORRECT" (defaults to INCORRECT if unclear)
        feedback: Feedback text (empty if CORRECT)
    """
    upper = response.upper()
    
    # Extract verdict
    verdict = "INCORRECT"  # Default to INCORRECT
    verdict_match = re.search(r'VERDICT\s*:\s*(CORRECT|INCORRECT)', upper)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    else:
        # Look for standalone verdict
        standalone_match = re.search(r'(^|\n|\s)(CORRECT|INCORRECT)(\s|$|\n)', upper)
        if standalone_match:
            verdict = standalone_match.group(2).upper()
        # If neither found, check for keywords
        elif 'CORRECT' in upper and 'INCORRECT' not in upper:
            verdict = "CORRECT"
        # Otherwise stays INCORRECT (default)
    
    # Extract feedback (everything before VERDICT)
    feedback = ""
    if verdict == "INCORRECT":
        verdict_pos = response.upper().find("VERDICT")
        if verdict_pos > 0:
            feedback = response[:verdict_pos].strip()
            # Clean up feedback
            feedback = re.sub(r'^(feedback|tip|suggestion|note|review)[:\s]*', '', feedback, flags=re.IGNORECASE)
            feedback = re.sub(r'\n\s*\n', '\n', feedback)
            # Extract bullet points or key points
            lines = [line.strip() for line in feedback.split('\n') if line.strip()]
            feedback = '\n'.join(lines[:5])  # Limit to 5 points
    
    return verdict, feedback


def generate_chat_response(
    model, 
    tokenizer, 
    prompt: str, 
    role: str, 
    detailed: bool = False,
    max_new_tokens: int = 512
) -> str:
    """
    Generate response in chat context.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Full conversation prompt
        role: 'solver' or 'checker'
        detailed: Whether to stream output
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated response text
    """
    # Handle both inference engines and standard models
    if hasattr(model, 'generate_single'):
        # Using inference engine (vLLM or TransformersEngine)
        temperature = 0.7 if role == "solver" else 0.3
        top_p = 0.95 if role == "solver" else 0.9
        
        response = model.generate_single(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            repetition_penalty=1.2,
            detailed=detailed
        )
    else:
        # Using standard PyTorch model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        prompt_length = inputs['input_ids'].shape[1]
        
        if detailed:
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer = None
        
        stopping_criteria = StoppingCriteriaList([
            StopOnBoxedAnswerOrVerdict(tokenizer, prompt_length, role)
        ])
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7 if role == "solver" else 0.3,
                do_sample=True,
                top_p=0.95 if role == "solver" else 0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                stopping_criteria=stopping_criteria,
                streamer=streamer
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
    
    return response


def truncate_conversation_history(conversation_history: List[Dict[str, str]], max_turns: int = 6) -> List[Dict[str, str]]:
    """
    Truncate conversation history to keep only recent turns.
    Always keeps the first turn (initial question) and recent turns.
    
    Args:
        conversation_history: Full conversation history
        max_turns: Maximum number of turns to keep (excluding first)
    
    Returns:
        Truncated conversation history
    """
    if len(conversation_history) <= max_turns + 1:
        return conversation_history
    
    # Keep first turn and last max_turns turns
    truncated = [conversation_history[0]]  # First turn
    truncated.extend(conversation_history[-(max_turns):])  # Recent turns
    
    return truncated


def run_solver_checker_chat_workflow(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    max_iterations: int = 5,
    detailed: bool = False,
    dataset_name: str = ""
) -> Dict:
    """
    Run solver-checker chat-based iterative workflow.
    
    Args:
        question: Math problem question
        ground_truth: Ground truth answer
        model: Shared model for both solver and checker
        tokenizer: Shared tokenizer
        max_iterations: Maximum number of solver-checker pairs
        detailed: Whether to show detailed output
        dataset_name: Dataset name
    
    Returns:
        Dictionary with workflow results
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.inference import generate_response
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    
    # Conversation history: list of {'role': 'solver'/'checker', 'content': '...'}
    conversation_history = []
    
    # Storage for all iterations
    iterations = []
    solver_responses = []
    solver_answers = []
    checker_responses = []
    checker_verdicts = []
    
    predicted_answer = None
    final_verdict = None
    
    # Iterative loop
    for iteration_num in range(1, max_iterations + 1):
        # Step 1: Solver generates response
        if iteration_num == 1:
            # First turn: use standard prompt
            solver_prompt = format_prompt_standard(question, dataset_name)
        else:
            # Subsequent turns: use conversation history
            solver_prompt = format_chat_prompt(conversation_history, "solver", question, dataset_name)
        
        try:
            solver_response = generate_chat_response(
                model, tokenizer, solver_prompt, "solver", detailed, max_new_tokens=512
            )
        except Exception as e:
            solver_response = f"Error: {e}"
        
        # Fallback if empty
        if not solver_response.strip():
            fallback_prompt = format_prompt_standard(question, dataset_name)
            try:
                from models.inference import generate_response
                solver_response = generate_response(model, tokenizer, fallback_prompt, "standard", detailed)
            except Exception as e:
                solver_response = f"Error: {e}"
        
        solver_answer = extract_answer(solver_response)
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        # Add solver response to conversation
        conversation_history.append({
            'role': 'solver',
            'content': solver_response
        })
        
        # Step 2: Checker evaluates and provides feedback
        checker_prompt = format_checker_feedback_prompt(conversation_history, question)
        
        if not checker_prompt:
            checker_response = "VERDICT: UNCLEAR"
        else:
            try:
                checker_response = generate_chat_response(
                    model, tokenizer, checker_prompt, "checker", detailed, max_new_tokens=200
                )
            except Exception as e:
                checker_response = f"Error: {e}"
        
        # Fallback if empty
        if not checker_response.strip():
            checker_response = "VERDICT: UNCLEAR"
        
        checker_verdict, checker_feedback = parse_checker_verdict_from_chat(checker_response)
        if checker_verdict not in ["CORRECT", "INCORRECT"]:
            checker_verdict = "INCORRECT"  # Default to INCORRECT
        
        checker_responses.append(checker_response)
        checker_verdicts.append(checker_verdict)
        
        # Add checker response to conversation
        conversation_history.append({
            'role': 'checker',
            'content': checker_response
        })
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "solver_prompt": solver_prompt,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "checker_prompt": checker_prompt,
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "checker_feedback": checker_feedback,
            "conversation_length": len(conversation_history)
        }
        iterations.append(iteration_data)
        
        # Check if solver answer is actually correct (ground truth check)
        is_actually_correct = False
        if solver_answer:
            is_actually_correct = check_answer(solver_answer, ground_truth)
        
        iteration_data["is_actually_correct"] = is_actually_correct
        
        # Decision logic
        if checker_verdict == "CORRECT":
            if solver_answer:
                predicted_answer = solver_answer
                final_verdict = "CORRECT"
                break
        
        # Truncate conversation history if too long (keep recent context)
        if len(conversation_history) > 10:  # More than 5 turns
            conversation_history = truncate_conversation_history(conversation_history, max_turns=6)
    
    # If no CORRECT verdict, use last valid answer (backward search)
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
    if first_correct and final_correct and len(iterations) == 1:
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
        "conversation_history": conversation_history
    }

