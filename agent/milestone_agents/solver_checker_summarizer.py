"""
Solver-Checker-Summarizer Multi-Agent Workflow (Stateless)
带总结层的Solver-Checker工作流 - Stateless模式

ARCHITECTURE:
┌─────────────────────────────────────────────────────────┐
│                  Iteration Loop                         │
└─────────────────────────────────────────────────────────┘
         │
         ├─► Solver generates solution
         │      │
         │      ▼
         │   Summarizer (Solver → Checker)
         │      │ 总结关键推理和答案
         │      ▼
         │   Checker evaluates summary
         │      │
         │      ▼
         │   Summarizer (Checker → Solver)
         │      │ 总结反馈要点
         │      ▼
         └─► Solver sees summarized feedback

KEY FEATURES:
- Stateless架构 (稳定、无幻觉)
- 双向总结: solver→checker, checker→solver
- 减少噪音，突出关键信息
- 帮助小模型聚焦重点

适用场景：
- Solver输出冗长的场景
- 需要提取关键信息的复杂推理
- 小模型容易被长文本干扰的情况
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import torch
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY,
    SUMMARY_MAX_TOKENS, SUMMARY_TEMPERATURE, CHECKER_MAX_TOKENS, CHECKER_TEMPERATURE
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


def summarize_solver_output(
    question: str,
    solver_response: str,
    solver_answer: str,
    model,
    tokenizer,
    detailed: bool = False
) -> str:
    """
    Summarize solver's output for checker
    总结Solver输出供Checker查看
    
    Args:
        question: Original question
        solver_response: Solver's full response
        solver_answer: Extracted answer
        model: Summarizer model
        tokenizer: Tokenizer
        detailed: Verbose output
    
    Returns:
        Summarized text
    """
    from models.inference import generate_response
    
    prompt = f"""You are a summarizer. Read the following solution and create a concise summary for verification.

Original Question: {question}

Full Solution:
{solver_response[:1000]}  # Truncate if too long

Your task:
1. Extract the key reasoning steps (3-5 steps max)
2. Note the final answer
3. Highlight any calculations or assumptions
4. Keep it concise (under 150 words)

Summary:"""
    
    if detailed:
        print("\n[Summarizing Solver Output for Checker]")
    
    # Check if model is an inference engine or raw model
    if hasattr(model, 'generate_single'):
        summary = model.generate_single(
            prompt,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2,
            detailed=False
        )
    else:
        summary = generate_response(
            model,
            tokenizer,
            prompt,
            "standard",
            detailed=False  # Don't show streaming for summarization
        )
    
    # Ensure summary includes the answer
    if solver_answer and solver_answer not in summary:
        summary += f"\n\nFinal Answer: {solver_answer}"
    
    if detailed:
        print(f"[Summary Length]: {len(summary)} chars (Original: {len(solver_response)} chars)")
    
    return summary


def summarize_checker_feedback(
    question: str,
    checker_response: str,
    checker_verdict: str,
    model,
    tokenizer,
    detailed: bool = False
) -> str:
    """
    Summarize checker's feedback for solver
    总结Checker反馈供Solver查看
    
    Args:
        question: Original question
        checker_response: Checker's full response
        checker_verdict: CORRECT/INCORRECT/UNCLEAR
        model: Summarizer model
        tokenizer: Tokenizer
        detailed: Verbose output
    
    Returns:
        Summarized feedback
    """
    from models.inference import generate_response
    
    if checker_verdict == "CORRECT":
        # No need for detailed summary if correct
        return "Your solution is correct. No changes needed."
    
    prompt = f"""You are a summarizer. Read the checker's feedback and create a concise summary for the solver.

Original Question: {question}

Checker's Full Feedback:
{checker_response[:800]}

Checker's Verdict: {checker_verdict}

Your task:
1. Extract the main issues identified (if any)
2. Provide actionable suggestions for improvement
3. Be specific but concise (under 100 words)
4. Focus on what needs to be fixed

Summary for Solver:"""
    
    if detailed:
        print("\n[Summarizing Checker Feedback for Solver]")
    
    # Check if model is an inference engine or raw model
    if hasattr(model, 'generate_single'):
        summary = model.generate_single(
            prompt,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2,
            detailed=False
        )
    else:
        summary = generate_response(
            model,
            tokenizer,
            prompt,
            "standard",
            detailed=False
        )
    
    # Ensure verdict is clear
    if checker_verdict not in summary:
        summary = f"[{checker_verdict}] {summary}"
    
    if detailed:
        print(f"[Summary Length]: {len(summary)} chars (Original: {len(checker_response)} chars)")
    
    return summary


def run_solver_checker_summarizer_workflow(
    question: str,
    ground_truth: str,
    solver_model,
    solver_tokenizer,
    checker_model,
    checker_tokenizer,
    summarizer_model=None,  # Can use same or different model
    summarizer_tokenizer=None,
    max_iterations: int = 5,
    detailed: bool = False,
    dataset_name: str = "",
    apply_chat_template: bool = False
) -> Dict:
    """
    Run Solver-Checker-Summarizer workflow (Stateless)
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        solver_model: Solver model
        solver_tokenizer: Solver tokenizer
        checker_model: Checker model
        checker_tokenizer: Checker tokenizer
        summarizer_model: Summarizer model (default: use solver_model)
        summarizer_tokenizer: Summarizer tokenizer
        max_iterations: Maximum iterations
        detailed: Verbose output
        dataset_name: Dataset name
        apply_chat_template: Whether to apply chat template to prompts
    
    Returns:
        Dict with workflow results
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response
    from agent.utils import (
        format_prompt_checker,
        parse_checker_verdict,
        parse_checker_tip,
        generate_response_checker
    )
    
    # Use solver model as summarizer if not specified
    if summarizer_model is None:
        summarizer_model = solver_model
        summarizer_tokenizer = solver_tokenizer
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-CHECKER-SUMMARIZER WORKFLOW (Stateless)")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Using Summarizer: {'Yes' if summarizer_model else 'No'}")
        print(f"{'='*80}\n")
    
    iterations = []
    solver_responses = []
    solver_answers = []
    checker_responses = []
    checker_verdicts = []
    
    predicted_answer = None
    final_verdict = None
    checker_feedback_summary = ""
    
    # Iterative loop
    for iteration_num in range(1, max_iterations + 1):
        if detailed:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration_num}/{max_iterations}")
            print(f"{'='*80}")
        
        # ========== SOLVER TURN ==========
        if iteration_num == 1:
            solver_prompt = format_prompt_standard(question, dataset_name)
        else:
            # Use summarized feedback
            solver_prompt = format_prompt_standard(question, dataset_name)
            solver_prompt += f"\n\nPrevious attempt feedback (summarized):\n{checker_feedback_summary}\n\nPlease address these issues and provide an improved solution."
        
        # Apply chat template if enabled
        solver_prompt = apply_chat_template_if_enabled(solver_prompt, solver_tokenizer, apply_chat_template)
        
        if detailed:
            print(f"\n[Solver Turn]")
        
        # Check if model is an inference engine or raw model
        if hasattr(solver_model, 'generate_single'):
            solver_response = solver_model.generate_single(
                solver_prompt,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.2,
                detailed=detailed
            )
        else:
            solver_response = generate_response(
                solver_model,
                solver_tokenizer,
                solver_prompt,
                "standard",
                detailed
            )
        
        solver_answer = extract_answer(solver_response)
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        if detailed:
            print(f"\n[Solver Answer]: {solver_answer}")
        
        # ========== SUMMARIZE SOLVER OUTPUT ==========
        solver_summary = summarize_solver_output(
            question,
            solver_response,
            solver_answer,
            summarizer_model,
            summarizer_tokenizer,
            detailed
        )
        
        # ========== CHECKER TURN ==========
        if detailed:
            print(f"\n[Checker Turn - Evaluating Summary]")
        
        # Build checker prompt using SUMMARY instead of full response
        checker_prompt = f"""You are a math solution checker. Verify the following solution summary.

Problem: {question}

Solution Summary:
{solver_summary}

Your task:
1. Check if the reasoning is correct
2. Verify calculations
3. Assess if the answer makes sense

Provide your verdict as:
- VERDICT: CORRECT (if solution is right)
- VERDICT: INCORRECT (if there are errors)  
- VERDICT: UNCLEAR (if you cannot determine)

Then explain your reasoning.

Evaluation:"""
        
        # Apply chat template if enabled
        checker_prompt = apply_chat_template_if_enabled(checker_prompt, checker_tokenizer, apply_chat_template)
        
        checker_response = generate_response_checker(
            checker_model,
            checker_tokenizer,
            checker_prompt,
            detailed
        )
        
        checker_responses.append(checker_response)
        checker_verdict = parse_checker_verdict(checker_response)
        
        if checker_verdict not in ["CORRECT", "INCORRECT"]:
            checker_verdict = "INCORRECT"  # Default to INCORRECT
        
        checker_verdicts.append(checker_verdict)
        
        if detailed:
            print(f"\n[Checker Verdict]: {checker_verdict}")
        
        # ========== SUMMARIZE CHECKER FEEDBACK ==========
        if checker_verdict != "CORRECT":
            checker_feedback_summary = summarize_checker_feedback(
                question,
                checker_response,
                checker_verdict,
                summarizer_model,
                summarizer_tokenizer,
                detailed
            )
        else:
            checker_feedback_summary = ""
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "solver_summary": solver_summary,  # Store summary
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "checker_feedback_summary": checker_feedback_summary,  # Store summary
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
        "config": {
            "use_summarizer": True,
            "mode": "stateless"
        }
    }


# For testing
if __name__ == "__main__":
    print("Solver-Checker-Summarizer (Stateless) - Quick Test")
    print("=" * 80)
    
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_ground_truth = "18"
    
    try:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        from models.inference import load_model
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning Solver-Checker-Summarizer workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_solver_checker_summarizer_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            solver_model=model,
            solver_tokenizer=tokenizer,
            checker_model=model,
            checker_tokenizer=tokenizer,
            summarizer_model=model,  # Same model for summarization
            summarizer_tokenizer=tokenizer,
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
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

