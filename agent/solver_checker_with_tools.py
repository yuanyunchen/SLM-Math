"""
Solver-Checker with Tools Multi-Agent Workflow
带工具调用的Solver-Checker工作流

STATELESS MODE + TOOL EXECUTION
- 基于stateless架构（稳定、无幻觉）
- Solver可以生成并执行Python代码
- Checker使用代码独立验证答案
- 自动代码执行提升准确性

工作流程：
1. Solver生成reasoning + code
2. 自动执行Solver的代码，提取答案
3. Checker独立计算并验证
4. 比较结果决定verdict
5. 基于反馈迭代改进

适用场景：
- 需要精确计算的数学问题
- 小模型容易算错的问题
- 需要验证计算过程的场景
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import torch


# Simple tool instruction - compatible with Qwen2.5-Math's default system prompt
SOLVER_TOOL_INSTRUCTION = """

Use Python code in ```python``` blocks for calculations. The code will be executed and results shown."""


def extract_answer_from_code_output(exec_results: list) -> str:
    """Extract numerical answer from code execution results."""
    if not exec_results:
        return None
    
    for result in reversed(exec_results):
        if result['success'] and result['output']:
            output_text = result['output'].strip()
            lines = output_text.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    num = float(line)
                    return str(num)
                except ValueError:
                    pass
                numbers = re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', line)
                if numbers:
                    return numbers[-1]
    return None


def compare_numerical_answers(ans1: str, ans2: str) -> bool:
    """
    Compare two numerical answers with tolerance for floating-point precision.
    Returns True if answers match (or are numerically equivalent).
    """
    if not ans1 or not ans2:
        return False
    
    # Clean answers
    ans1 = str(ans1).strip().replace(',', '')
    ans2 = str(ans2).strip().replace(',', '')
    
    # Direct string match
    if ans1 == ans2:
        return True
    
    # Try numerical comparison
    try:
        num1 = float(ans1)
        num2 = float(ans2)
        
        # Exact match (for integers)
        if num1 == num2:
            return True
        
        # Relative tolerance for floats
        if abs(num1) > 1e-10:
            rel_diff = abs(num1 - num2) / abs(num1)
            if rel_diff < 0.001:  # 0.1% tolerance
                return True
        
        # Absolute tolerance for small numbers
        if abs(num1 - num2) < 0.01:
            return True
            
    except (ValueError, TypeError):
        pass
    
    # Fallback to check_answer
    from utils.prompt_utils import check_answer
    return check_answer(ans1, ans2)


def run_checker_with_code(
    question: str,
    solver_answer: str,
    model,
    tokenizer,
    detailed: bool = False
) -> Dict:
    """
    Run checker that independently solves the problem with code and compares.
    
    Returns verdict, checker_answer, and whether code was executed.
    Verdict is CORRECT if answers match, INCORRECT otherwise.
    """
    from models.inference import generate_response
    from utils.python_code_execution import process_text_with_code_execution
    
    # Simple prompt - let model solve independently
    checker_prompt = f"""{question}

Use Python code in ```python``` blocks to solve this. Print the final numerical answer."""
    
    # Generate checker's independent solution
    response = generate_response(
        model, tokenizer, checker_prompt, "standard", detailed,
        temperature=0.0, do_sample=False, top_p=1.0
    )
    
    # Execute code
    response_with_output, exec_results = process_text_with_code_execution(
        response, share_variables=True
    )
    
    # Extract checker's answer from code output
    checker_answer = extract_answer_from_code_output(exec_results)
    code_executed = len(exec_results) > 0 if exec_results else False
    
    # If no code output, try boxed answer
    if not checker_answer:
        from utils.prompt_utils import extract_answer
        checker_answer = extract_answer(response)
    
    # Compare answers
    verdict = "INCORRECT"
    if checker_answer and solver_answer:
        if compare_numerical_answers(checker_answer, solver_answer):
            verdict = "CORRECT"
    elif not checker_answer:
        # Checker failed to produce answer - trust solver
        verdict = "CORRECT"
    
    if detailed:
        print(f"  Checker's answer: {checker_answer}")
        print(f"  Solver's answer: {solver_answer}")
        print(f"  Verdict: {verdict}")
    
    return {
        "verdict": verdict,
        "checker_answer": checker_answer,
        "response": response_with_output if exec_results else response,
        "code_executed": code_executed
    }


def run_solver_checker_with_tools_workflow(
    question: str,
    ground_truth: str,
    solver_model,
    solver_tokenizer,
    checker_model,
    checker_tokenizer,
    max_iterations: int = 5,
    detailed: bool = False,
    dataset_name: str = "",
    enable_solver_tools: bool = True,
    enable_checker_tools: bool = True
) -> Dict:
    """
    Run Solver-Checker workflow with tool execution
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        solver_model: Solver model
        solver_tokenizer: Solver tokenizer
        checker_model: Checker model
        checker_tokenizer: Checker tokenizer
        max_iterations: Maximum iterations
        detailed: Verbose output
        dataset_name: Dataset name
        enable_solver_tools: Enable code execution for solver
        enable_checker_tools: Enable code execution for checker
    
    Returns:
        Dict with workflow results
    """
    from utils.prompt_utils import extract_answer, check_answer
    from models.inference import generate_response
    from utils.python_code_execution import process_text_with_code_execution
    from agent.unified_config import FIRST_ROUND_SOLVER_CONFIG
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-CHECKER WITH TOOLS WORKFLOW")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Solver Tools: {'enabled' if enable_solver_tools else 'disabled'}")
        print(f"Checker Tools: {'enabled' if enable_checker_tools else 'disabled'}")
        print(f"{'='*80}\n")
    
    iterations = []
    solver_responses = []
    solver_answers = []
    checker_responses = []
    checker_verdicts = []
    
    predicted_answer = None
    final_verdict = None
    checker_feedback = ""
    
    # Iterative loop
    for iteration_num in range(1, max_iterations + 1):
        if detailed:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration_num}/{max_iterations}")
            print(f"{'='*80}")
        
        # ========== SOLVER TURN ==========
        if iteration_num == 1:
            # First iteration: simple prompt (Qwen2.5-Math has default system prompt)
            solver_prompt = question
            if enable_solver_tools:
                solver_prompt += SOLVER_TOOL_INSTRUCTION
            
            if detailed:
                print(f"\n[Solver Turn - First Round]")
            
            # Use unified config for first round
            solver_response = generate_response(
                solver_model, solver_tokenizer, solver_prompt, "standard", detailed,
                temperature=FIRST_ROUND_SOLVER_CONFIG['temperature'],
                do_sample=FIRST_ROUND_SOLVER_CONFIG['do_sample'],
                top_p=FIRST_ROUND_SOLVER_CONFIG['top_p']
            )
        else:
            # Subsequent iterations: incorporate feedback
            solver_prompt = f"""{question}

Previous attempt was incorrect. {checker_feedback}

Please solve again carefully."""
            if enable_solver_tools:
                solver_prompt += SOLVER_TOOL_INSTRUCTION
            
            if detailed:
                print(f"\n[Solver Turn - Retry]")
            
            solver_response = generate_response(
                solver_model, solver_tokenizer, solver_prompt, "standard", detailed,
                temperature=0.3, do_sample=True, top_p=0.9
            )
        
        # Execute code in solver response
        solver_exec_results = []
        if enable_solver_tools and "```python" in solver_response:
            solver_response_with_output, solver_exec_results = process_text_with_code_execution(
                solver_response, share_variables=True
            )
            
            if solver_exec_results:
                if detailed:
                    print(f"\n[Solver Code Execution]")
                    for i, result in enumerate(solver_exec_results, 1):
                        if result['success']:
                            output_preview = result['output'][:80] if result['output'] else "(no output)"
                            print(f"  Block {i}: SUCCESS -> {output_preview}")
                        else:
                            print(f"  Block {i}: ERROR -> {result['error'][:60]}")
                solver_response = solver_response_with_output
        
        # Extract solver answer - prioritize code output
        solver_answer = None
        if solver_exec_results:
            solver_answer = extract_answer_from_code_output(solver_exec_results)
        if not solver_answer:
            solver_answer = extract_answer(solver_response)
        
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        if detailed:
            print(f"\n[Solver Answer]: {solver_answer}")
        
        # ========== CHECKER TURN ==========
        if detailed:
            print(f"\n[Checker Turn]")
        
        if enable_checker_tools:
            # Checker independently solves and compares
            checker_result = run_checker_with_code(
                question, solver_answer,
                checker_model, checker_tokenizer, detailed
            )
            checker_response = checker_result['response']
            checker_verdict = checker_result['verdict']
            checker_answer = checker_result['checker_answer']
            checker_tools_used = checker_result['code_executed']
            
            # Generate feedback if INCORRECT
            if checker_verdict == "INCORRECT" and checker_answer:
                checker_feedback = f"Your answer {solver_answer} differs from verified answer {checker_answer}. Please recalculate."
            else:
                checker_feedback = "The solution appears incorrect. Please check your reasoning."
        else:
            # Simple text-based checking (fallback)
            from agent.utils import format_prompt_checker, parse_checker_verdict, generate_response_checker
            checker_prompt = format_prompt_checker(question, solver_response, dataset_name)
            checker_response = generate_response_checker(
                checker_model, checker_tokenizer, checker_prompt, detailed
            )
            checker_verdict = parse_checker_verdict(checker_response)
            checker_answer = None
            checker_tools_used = False
            checker_feedback = "The previous solution may be incorrect. Please reconsider."
        
        if checker_verdict not in ["CORRECT", "INCORRECT"]:
            checker_verdict = "INCORRECT"
        
        checker_responses.append(checker_response)
        checker_verdicts.append(checker_verdict)
        
        if detailed:
            print(f"\n[Checker Verdict]: {checker_verdict}")
        
        # Store iteration data
        is_actually_correct = check_answer(solver_answer, ground_truth) if solver_answer else False
        
        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "checker_feedback": checker_feedback,
            "solver_tools_used": enable_solver_tools and "```python" in solver_response,
            "checker_tools_used": checker_tools_used,
            "is_actually_correct": is_actually_correct
        }
        if enable_checker_tools:
            iteration_data["checker_answer"] = checker_answer
        
        iterations.append(iteration_data)
        
        # Decision logic
        if checker_verdict == "CORRECT":
            if solver_answer:
                predicted_answer = solver_answer
                final_verdict = "CORRECT"
                if detailed:
                    print(f"\n[Checker confirmed CORRECT]")
                break
        
        if iteration_num >= max_iterations:
            if detailed:
                print(f"\nReached max iterations ({max_iterations})")
            break
    
    # If no CORRECT verdict, prefer FIRST answer (to avoid degradation)
    # The first try has 82% accuracy, so trust it over later iterations
    if predicted_answer is None:
        # First choice: use first valid answer
        first_answer_found = None
        for i in range(len(iterations)):
            iter_answer = iterations[i]['solver_answer']
            if iter_answer and iter_answer.strip():
                first_answer_found = iter_answer
                break
        
        if first_answer_found:
            predicted_answer = first_answer_found
            final_verdict = "FIRST_ANSWER_FALLBACK"
        else:
            # Fallback: any valid answer
            for i in range(len(iterations) - 1, -1, -1):
                iter_answer = iterations[i]['solver_answer']
                if iter_answer and iter_answer.strip():
                    predicted_answer = iter_answer
                    final_verdict = f"LAST_VALID_ITER_{iterations[i]['iteration']}"
                    break
            if predicted_answer is None:
                final_verdict = "NO_VALID_ANSWER"
    
    # Check final correctness
    final_correct = check_answer(predicted_answer, ground_truth) if predicted_answer else False
    
    # Determine case type
    first_answer = solver_answers[0] if solver_answers else None
    first_correct = check_answer(first_answer, ground_truth) if first_answer else False
    
    if first_correct and final_correct:
        case_type = "FIRST_TRY_SUCCESS"
    elif not first_correct and final_correct:
        case_type = "IMPROVED"
    elif first_correct and not final_correct:
        case_type = "DEGRADED"
    else:
        case_type = "FAILED"
    
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
        "tools_config": {
            "solver_tools": enable_solver_tools,
            "checker_tools": enable_checker_tools
        }
    }


# For testing
if __name__ == "__main__":
    print("Solver-Checker with Tools - Quick Test")
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
        
        print(f"\nRunning Solver-Checker with Tools workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_solver_checker_with_tools_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            solver_model=model,
            solver_tokenizer=tokenizer,
            checker_model=model,
            checker_tokenizer=tokenizer,
            max_iterations=3,
            detailed=True,
            dataset_name="gsm8k",
            enable_solver_tools=True,
            enable_checker_tools=True
        )
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Expected: {result['ground_truth']}")
        print(f"Correct: {result['final_correct']}")
        print(f"Case Type: {result['case_type']}")
        print(f"Iterations: {result['total_iterations']}")
        
        # Show tool usage statistics
        solver_tool_usage = sum(1 for it in result['iterations'] if it.get('solver_tools_used', False))
        checker_tool_usage = sum(1 for it in result['iterations'] if it.get('checker_tools_used', False))
        print(f"\nTool Usage:")
        print(f"  Solver used tools: {solver_tool_usage}/{len(result['iterations'])} iterations")
        print(f"  Checker used tools: {checker_tool_usage}/{len(result['iterations'])} iterations")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

