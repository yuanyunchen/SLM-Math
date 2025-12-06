"""
Solver-Checker Stateless Multi-Agent Workflow
独立实现solver-checker迭代工作流的核心逻辑

STATELESS MODE - 每轮使用独立的prompt，不维护对话历史
- 更稳定，不会出现幻觉问题
- 适合小模型(<2B参数)
- 推荐用于生产环境

原名: solver_checker_separate.py
"""

from typing import Dict
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY,
    CHECKER_MAX_TOKENS, CHECKER_TEMPERATURE, CHECKER_TOP_P, CHECKER_REPETITION_PENALTY
)


def apply_chat_template_if_enabled(prompt: str, tokenizer, apply_chat_template: bool) -> str:
    """Wrap prompt with chat template if enabled."""
    if not apply_chat_template:
        return prompt
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def run_solver_checker_workflow(
    question: str,
    ground_truth: str,
    solver_model,
    solver_tokenizer,
    checker_model,
    checker_tokenizer,
    max_iterations: int = 5,
    detailed: bool = False,
    dataset_name: str = "",
    apply_chat_template: bool = False
) -> Dict:
    """
    Run solver-checker iterative workflow.
    
    Args:
        question: Math problem question
        ground_truth: Ground truth answer
        solver_model: Solver model
        solver_tokenizer: Solver tokenizer
        checker_model: Checker model (can be same as solver)
        checker_tokenizer: Checker tokenizer
        max_iterations: Maximum number of iterations
        detailed: Whether to show detailed output
        dataset_name: Dataset name
        apply_chat_template: Whether to apply chat template to prompts
    
    Returns:
        Dictionary with workflow results
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.inference import generate_response
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from agent.utils import (
        format_prompt_solver,
        format_prompt_checker,
        parse_checker_verdict,
        parse_checker_tip,
        generate_response_checker
    )
    
    # Storage for all iterations
    iterations = []
    solver_responses = []
    solver_answers = []
    checker_responses = []
    checker_verdicts = []
    
    checker_feedback = ""
    predicted_answer = None
    final_verdict = None
    
    # Iterative loop
    for iteration_num in range(1, max_iterations + 1):
        # Step 1: Solver generates response
        # First iteration uses standard prompt, later iterations use feedback
        if iteration_num == 1:
            solver_prompt = format_prompt_standard(question, dataset_name)
        else:
            solver_prompt = format_prompt_solver(question, checker_feedback, dataset_name)
        solver_prompt = apply_chat_template_if_enabled(solver_prompt, solver_tokenizer, apply_chat_template)
        
        try:
            # Check if model is an inference engine or raw model
            if hasattr(solver_model, 'generate_single'):
                # Using inference engine
                solver_response = solver_model.generate_single(
                    solver_prompt,
                    max_new_tokens=4096,
                    temperature=0.1,
                    do_sample=False,
                    repetition_penalty=1.2,
                    detailed=detailed
                )
            else:
                # Using standard model
                solver_response = generate_response(solver_model, solver_tokenizer, solver_prompt, "standard", detailed)
        except Exception as e:
            solver_response = f"Error: {e}"
        
        # Fallback if empty
        if not solver_response.strip():
            fallback_prompt = format_prompt_standard(question, dataset_name)
            fallback_prompt = apply_chat_template_if_enabled(fallback_prompt, solver_tokenizer, apply_chat_template)
            try:
                if hasattr(solver_model, 'generate_single'):
                    solver_response = solver_model.generate_single(
                        fallback_prompt,
                        max_new_tokens=4096,
                        temperature=0.1,
                        do_sample=False,
                        repetition_penalty=1.2,
                        detailed=detailed
                    )
                else:
                    solver_response = generate_response(solver_model, solver_tokenizer, fallback_prompt, "standard", detailed)
            except Exception as e:
                solver_response = f"Error: {e}"
        
        solver_answer = extract_answer(solver_response)
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        # Step 2: Checker evaluates
        checker_prompt = format_prompt_checker(question, solver_response, dataset_name)
        checker_prompt = apply_chat_template_if_enabled(checker_prompt, checker_tokenizer, apply_chat_template)
        
        try:
            # Check if model is an inference engine or raw model
            if hasattr(checker_model, 'generate_single'):
                # Using inference engine
                checker_response = checker_model.generate_single(
                    checker_prompt,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    detailed=detailed
                )
            else:
                # Using standard model
                checker_response = generate_response_checker(checker_model, checker_tokenizer, checker_prompt, detailed)
        except Exception as e:
            checker_response = f"Error: {e}"
        
        # Fallback if empty
        if not checker_response.strip():
            simple_prompt = f"Q: {question}\nA: {solver_answer}\n\nVERDICT:"
            simple_prompt = apply_chat_template_if_enabled(simple_prompt, checker_tokenizer, apply_chat_template)
            try:
                if hasattr(checker_model, 'generate_single'):
                    checker_response = checker_model.generate_single(
                        simple_prompt,
                        max_new_tokens=256,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.3,
                        detailed=detailed
                    )
                else:
                    checker_response = generate_response_checker(checker_model, checker_tokenizer, simple_prompt, detailed)
            except Exception as e:
                checker_response = "VERDICT: UNCLEAR"
        
        checker_verdict = parse_checker_verdict(checker_response)
        if checker_verdict not in ["CORRECT", "INCORRECT"]:
            checker_verdict = "INCORRECT"  # Default to INCORRECT
        
        checker_responses.append(checker_response)
        checker_verdicts.append(checker_verdict)
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "solver_prompt": solver_prompt,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "checker_prompt": checker_prompt,
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "checker_feedback": checker_feedback
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
        else:  # INCORRECT
            checker_feedback = parse_checker_tip(checker_response)
            if not checker_feedback:
                checker_feedback = "The previous solution was incorrect. Please reconsider the problem."
    
    # If no CORRECT verdict, use last valid answer (backward search)
    if predicted_answer is None:
        # Search backward from last iteration to find first valid answer
        for i in range(len(iterations) - 1, -1, -1):
            iter_answer = iterations[i]['solver_answer']
            if iter_answer and iter_answer.strip():
                predicted_answer = iter_answer
                final_verdict = f"LAST_VALID_ITER_{iterations[i]['iteration']}"
                break
        
        # If no valid answer found in any iteration
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
        "case_type": case_type
    }
