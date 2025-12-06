"""
Solver-Checker with Tools Multi-Agent Workflow
带工具调用的Solver-Checker工作流

STATELESS MODE + TOOL EXECUTION
- 基于stateless架构（稳定、无幻觉）
- Solver可以生成并执行Python代码
- Checker可以验证代码和结果
- 自动代码执行提升准确性

工作流程：
1. Solver生成reasoning + code
2. 自动执行Solver的代码
3. Solver看到执行结果继续推理
4. Checker验证代码逻辑和执行结果
5. 基于反馈迭代改进

适用场景：
- 需要精确计算的数学问题
- 小模型容易算错的问题
- 需要验证计算过程的场景
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import torch
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
    enable_checker_tools: bool = True,
    apply_chat_template: bool = False
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
    from utils.python_code_execution import process_text_with_code_execution
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-CHECKER WITH TOOLS WORKFLOW")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Solver Tools: {'✓' if enable_solver_tools else '✗'}")
        print(f"Checker Tools: {'✓' if enable_checker_tools else '✗'}")
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
            # First iteration: standard prompt
            solver_prompt = format_prompt_standard(question, dataset_name)
            
            # Add tool usage instruction
            if enable_solver_tools:
                solver_prompt += "\n\nYou can write Python code in ```python``` blocks to help with calculations. The code will be executed automatically and you'll see the results."
        else:
            # Subsequent iterations: incorporate feedback
            solver_prompt = format_prompt_standard(question, dataset_name)
            solver_prompt += f"\n\nPrevious attempt feedback:\n{checker_feedback}\n\nPlease address the issues and provide an improved solution."
            
            if enable_solver_tools:
                solver_prompt += "\n\nYou can write Python code in ```python``` blocks for calculations."
        
        # Apply chat template if enabled
        solver_prompt = apply_chat_template_if_enabled(solver_prompt, solver_tokenizer, apply_chat_template)
        
        if detailed:
            print(f"\n[Solver Turn]")
        
        # Generate solver response - check if model is an inference engine or raw model
        if hasattr(solver_model, 'generate_single'):
            # Using inference engine (from load_inference_engine_wrapper)
            solver_response = solver_model.generate_single(
                solver_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                detailed=detailed
            )
        else:
            # Using standard model (from load_model)
            solver_response = generate_response(
                solver_model,
                solver_tokenizer,
                solver_prompt,
                "standard",
                detailed
            )
        
        # Execute code in solver response if enabled
        if enable_solver_tools:
            solver_response_with_output, exec_results = process_text_with_code_execution(
                solver_response,
                share_variables=True
            )
            
            if exec_results:
                if detailed:
                    print(f"\n[Code Execution]")
                    for i, result in enumerate(exec_results, 1):
                        if result['success']:
                            print(f"  Block {i}: ✓ Output: {result['output'][:50]}")
                        else:
                            print(f"  Block {i}: ✗ Error: {result['error'][:50]}")
                
                # Use response with execution results
                solver_response = solver_response_with_output
                
                # Check if any execution failed and add note for checker
                failed_blocks = [r for r in exec_results if not r['success']]
                if failed_blocks:
                    error_note = f"\n\n[Note: {len(failed_blocks)} code block(s) failed to execute properly]"
                    solver_response += error_note
        
        solver_answer = extract_answer(solver_response)
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        if detailed:
            print(f"\n[Solver Answer]: {solver_answer}")
        
        # ========== CHECKER TURN ==========
        if detailed:
            print(f"\n[Checker Turn]")
        
        # Build checker prompt with tool support
        checker_prompt = format_prompt_checker(
            question,
            solver_response,
            dataset_name
        )
        
        if enable_checker_tools:
            checker_prompt += "\n\nYou can write Python code in ```python``` blocks to verify calculations or test the solution."
        
        # Apply chat template if enabled
        checker_prompt = apply_chat_template_if_enabled(checker_prompt, checker_tokenizer, apply_chat_template)
        
        # Generate checker response - check if model is an inference engine or raw model
        if hasattr(checker_model, 'generate_single'):
            # Using inference engine (from load_inference_engine_wrapper)
            checker_response = checker_model.generate_single(
                checker_prompt,
                max_new_tokens=CHECKER_MAX_TOKENS,
                temperature=CHECKER_TEMPERATURE,
                do_sample=DO_SAMPLE,
                top_p=CHECKER_TOP_P,
                repetition_penalty=CHECKER_REPETITION_PENALTY,
                detailed=detailed
            )
        else:
            # Using standard model (from load_model)
            checker_response = generate_response_checker(
                checker_model,
                checker_tokenizer,
                checker_prompt,
                detailed
            )
        
        # Execute code in checker response if enabled
        if enable_checker_tools:
            checker_response_with_output, checker_exec_results = process_text_with_code_execution(
                checker_response,
                share_variables=True
            )
            
            if checker_exec_results:
                if detailed:
                    print(f"\n[Checker Code Execution]")
                    for i, result in enumerate(checker_exec_results, 1):
                        if result['success']:
                            print(f"  Block {i}: ✓ {result['output'][:50]}")
                        else:
                            print(f"  Block {i}: ✗ {result['error'][:50]}")
                
                # Use response with execution results
                checker_response = checker_response_with_output
                
                # Check if any checker execution failed
                checker_failed_blocks = [r for r in checker_exec_results if not r['success']]
                if checker_failed_blocks:
                    # Checker's verification code failed - this affects reliability of verdict
                    if detailed:
                        print(f"  [Warning: Checker's verification code had {len(checker_failed_blocks)} error(s)]")
        
        checker_responses.append(checker_response)
        checker_verdict = parse_checker_verdict(checker_response)
        
        if checker_verdict not in ["CORRECT", "INCORRECT"]:
            checker_verdict = "INCORRECT"  # Default to INCORRECT
        
        checker_verdicts.append(checker_verdict)
        
        # Extract feedback for next iteration
        if checker_verdict == "INCORRECT":
            checker_feedback = parse_checker_tip(checker_response)
            if not checker_feedback:
                checker_feedback = "The previous solution was incorrect. Please reconsider the problem."
        elif checker_verdict == "UNCLEAR":
            checker_feedback = "The previous solution was unclear. Please provide clearer reasoning."
        else:
            checker_feedback = ""
        
        if detailed:
            print(f"\n[Checker Verdict]: {checker_verdict}")
            if checker_feedback:
                print(f"[Feedback]: {checker_feedback[:100]}...")
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "checker_feedback": checker_feedback,
            "solver_tools_used": enable_solver_tools and "```python" in solver_response,
            "checker_tools_used": enable_checker_tools and "```python" in checker_response
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
        
        # Max iterations check
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
    
    # Compile results
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

