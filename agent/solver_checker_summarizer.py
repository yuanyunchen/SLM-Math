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


def is_repetitive_text(text: str, threshold: float = 0.7) -> bool:
    """
    Check if text contains high repetition (garbage output detection)
    
    Args:
        text: Text to check
        threshold: Ratio threshold (0.7 means 70% repetition is considered bad)
    
    Returns:
        True if text is highly repetitive
    """
    import re
    
    if len(text) < 50:
        return False
    
    # Remove whitespace and split into words
    words = re.findall(r'\w+', text.lower())
    
    if len(words) < 10:
        return False
    
    # Check unique word ratio
    unique_ratio = len(set(words)) / len(words)
    
    # Also check for repeated sequences
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    max_repetition = max(word_counts.values()) if word_counts else 0
    max_rep_ratio = max_repetition / len(words) if words else 0
    
    # Text is repetitive if unique ratio is low OR a single word dominates
    return unique_ratio < (1 - threshold) or max_rep_ratio > 0.5


def validate_solver_relevance(question: str, solver_response: str) -> bool:
    """
    Check if solver response is actually addressing the given question.
    Detects cases where model generates unrelated problems/answers.
    
    Args:
        question: Original question
        solver_response: Solver's response
    
    Returns:
        True if response seems relevant to question, False otherwise
    """
    # Extract key numbers from question
    import re
    question_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', question))
    response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', solver_response[:500]))
    
    # Check if at least some question numbers appear in response
    if question_numbers:
        overlap = question_numbers & response_numbers
        # If less than 30% of question numbers appear, likely irrelevant
        if len(overlap) / len(question_numbers) < 0.3:
            return False
    
    # Check for key words from question appearing in response
    question_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', question.lower()))
    response_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', solver_response[:500].lower()))
    
    if question_words:
        # Remove common words
        common_words = {'that', 'this', 'with', 'from', 'have', 'will', 'each', 
                       'many', 'much', 'what', 'does', 'answer', 'solution'}
        question_words = question_words - common_words
        if question_words:
            overlap = question_words & response_words
            # If less than 20% of unique words match, likely irrelevant
            if len(overlap) / len(question_words) < 0.2:
                return False
    
    return True


def summarize_solver_output(
    question: str,
    solver_response: str,
    solver_answer: str,
    model,
    tokenizer,
    detailed: bool = False
) -> str:
    """
    Summarize solver's output for checker (with garbage detection)
    
    OPTIMIZATION: Added relevance validation to detect when solver
    generates unrelated problems.
    
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
    import torch
    
    # Truncate solver response if too long (keep first 800 chars)
    truncated_response = solver_response[:800] if len(solver_response) > 800 else solver_response
    
    # OPTIMIZATION: Check if solver response is relevant to the question
    is_relevant = validate_solver_relevance(question, solver_response)
    if not is_relevant and detailed:
        print("\n[WARNING] Solver response may not address the original question")
    
    # For small models, skip summarization - just pass key info directly
    # Summarization often produces garbage with 1.5B models
    if solver_answer:
        # Create a simple, structured summary without LLM generation
        # Include relevance warning if needed
        if is_relevant:
            simple_summary = f"Solution steps: {truncated_response[:300]}...\n\nFinal Answer: {solver_answer}"
        else:
            simple_summary = f"[WARNING: Response may not address the question]\n\nSolution steps: {truncated_response[:300]}...\n\nFinal Answer: {solver_answer}"
        if detailed:
            print("\n[Using Simple Summary (skip LLM summarization)]")
            print(f"[Summary Length]: {len(simple_summary)} chars")
        return simple_summary
    
    prompt = f"""You are a math solution summarizer. Create a SHORT summary (maximum 100 words).

Question: {question}

Solution: {truncated_response}

Final Answer: {solver_answer}

Write a concise summary with:
1. Key calculation steps (2-3 points only)
2. Final answer
3. Nothing else

Summary:"""
    
    if detailed:
        print("\n[Summarizing Solver Output for Checker]")
    
    try:
        # Apply chat template if available
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,        # Strict limit
                temperature=0.3,            # Lower temperature for focus
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.5,    # Strong anti-repetition
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_output[len(prompt):].strip() if full_output.startswith(prompt) else full_output.strip()
        
        # Detect garbage output
        if len(summary) < 20 or is_repetitive_text(summary):
            if detailed:
                print(f"[Warning] Summary appears to be garbage/repetitive, using fallback")
            # Fallback to simple format
            summary = f"Steps: {truncated_response[:200]}\n\nFinal Answer: {solver_answer}"
        
        # Ensure summary includes the answer
        if solver_answer and solver_answer not in summary:
            summary += f"\n\nFinal Answer: {solver_answer}"
        
        if detailed:
            print(f"[Summary Length]: {len(summary)} chars (Original: {len(solver_response)} chars)")
        
        return summary
        
    except Exception as e:
        if detailed:
            print(f"[Warning] Summarization failed: {e}")
        # Fallback: return simple format
        fallback = f"Solution: {truncated_response[:200]}\n\nFinal Answer: {solver_answer}"
        return fallback


def summarize_checker_feedback(
    question: str,
    checker_response: str,
    checker_verdict: str,
    model,
    tokenizer,
    detailed: bool = False
) -> str:
    """
    Summarize checker's feedback for solver (with garbage detection)
    总结Checker反馈供Solver查看 (带垃圾输出检测)
    
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
    import torch
    
    if checker_verdict == "CORRECT":
        # No need for detailed summary if correct
        return "Your solution is correct. No changes needed."
    
    # Truncate checker response if too long (keep first 500 chars)
    truncated_response = checker_response[:500] if len(checker_response) > 500 else checker_response
    
    prompt = f"""Summarize this feedback briefly (max 80 words).

Question: {question}

Checker Feedback: {truncated_response}

Verdict: {checker_verdict}

Provide:
1. Main issue (1-2 sentences)
2. What to fix (1 sentence)

Feedback:"""
    
    if detailed:
        print("\n[Summarizing Checker Feedback for Solver]")
    
    try:
        # Use stricter generation parameters
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,         # Shorter for feedback
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_output[len(prompt):].strip() if full_output.startswith(prompt) else full_output.strip()
        
        # Detect garbage output
        if len(summary) < 10 or is_repetitive_text(summary):
            if detailed:
                print(f"[Warning] Feedback summary appears invalid, using fallback")
            summary = f"Please review your calculation. Checker found issues."
        
        # Ensure verdict is clear
        if checker_verdict not in summary.upper():
            summary = f"[{checker_verdict}] {summary}"
        
        if detailed:
            print(f"[Summary Length]: {len(summary)} chars (Original: {len(checker_response)} chars)")
        
        return summary
        
    except Exception as e:
        if detailed:
            print(f"[Warning] Summarization failed: {e}")
        # Fallback: return basic feedback
        return f"[{checker_verdict}] Please review and revise your solution."


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
    dataset_name: str = ""
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
    from agent.unified_config import FIRST_ROUND_SOLVER_CONFIG
    
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
            if detailed:
                print(f"\n[Solver Turn - First Round with Unified Config]")
            # Use unified config for first round (deterministic)
            solver_response = generate_response(
                solver_model,
                solver_tokenizer,
                solver_prompt,
                "standard",
                detailed,
                temperature=FIRST_ROUND_SOLVER_CONFIG['temperature'],
                do_sample=FIRST_ROUND_SOLVER_CONFIG['do_sample'],
                top_p=FIRST_ROUND_SOLVER_CONFIG['top_p']
            )
        else:
            # Use summarized feedback
            solver_prompt = format_prompt_standard(question, dataset_name)
            solver_prompt += f"\n\nPrevious attempt feedback (summarized):\n{checker_feedback_summary}\n\nPlease address these issues and provide an improved solution."
            
            if detailed:
                print(f"\n[Solver Turn]")
            
            # Use default config for subsequent rounds
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
        
        # ========== VALIDATE SOLVER RELEVANCE ==========
        is_relevant = validate_solver_relevance(question, solver_response)
        if not is_relevant and detailed:
            print(f"\n[WARNING] Solver response seems irrelevant to the question!")
        
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
        
        # Use summarized version for checking (more focused)
        # Note: We use solver_summary instead of solver_response
        checker_prompt = format_prompt_checker(question, solver_summary, dataset_name)
        
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
        
        # OPTIMIZATION: If solver response is irrelevant, force INCORRECT verdict
        # This catches cases where model generates unrelated problems
        if not is_relevant and checker_verdict == "CORRECT":
            if detailed:
                print(f"[OVERRIDE] Changing CORRECT to INCORRECT due to irrelevant response")
            checker_verdict = "INCORRECT"
        
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
            "solver_relevant": is_relevant,    # Store relevance check result
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
    
    # If no CORRECT verdict, use smart answer selection
    # OPTIMIZATION V3: Prefer first answer if it was relevant and valid
    # This reduces DEGRADED cases where checker wrongly rejects correct first answer
    if predicted_answer is None:
        first_answer = solver_answers[0] if solver_answers else None
        first_relevant = iterations[0].get('solver_relevant', True) if iterations else False
        
        # Check if first answer looks like a valid numeric answer
        def is_valid_numeric(ans):
            if not ans:
                return False
            # Remove common LaTeX/formatting
            cleaned = str(ans).replace('\\', '').replace('$', '').replace(',', '')
            try:
                float(cleaned)
                return True
            except:
                # Check if it's a simple integer
                return cleaned.isdigit() or (cleaned.startswith('-') and cleaned[1:].isdigit())
        
        # Strategy: Prefer first answer if it was relevant and looks numeric
        if first_answer and first_relevant and is_valid_numeric(first_answer):
            predicted_answer = first_answer
            final_verdict = "FIRST_ANSWER_PROTECTED"
            if detailed:
                print(f"[STRATEGY] Using protected first answer: {predicted_answer}")
        else:
            # Fallback: search backward for any valid answer
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

