"""
Solver-Checker with Tools Multi-Agent Workflow V2
带工具调用的Solver-Checker工作流 V2

改进点：
1. 修复 UNCLEAR 分支逻辑
2. 更清晰的 case_type 命名
3. 基于执行结果而非字符串检测工具使用
4. 答案提取后再添加错误注释，避免干扰提取
5. 参考 agent_with_code_feedback.py 的代码执行逻辑

工作流程：
1. Solver生成reasoning + code
2. 自动执行Solver的代码
3. Solver看到执行结果继续推理
4. Checker验证代码逻辑和执行结果
5. 基于反馈迭代改进
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Tuple
import re
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


def execute_code_blocks(response: str, detailed: bool = False) -> Tuple[str, List[Dict], bool]:
    """
    Extract and execute Python code blocks from response.
    
    Returns:
        Tuple of (response_with_output, exec_results, any_executed)
    """
    from utils.python_code_execution import extract_python_code_blocks, execute_python_code
    
    code_blocks = extract_python_code_blocks(response)
    
    if not code_blocks:
        return response, [], False
    
    exec_results = []
    output_parts = []
    
    for i, code in enumerate(code_blocks, 1):
        result = execute_python_code(code, timeout=10)
        exec_results.append(result)
        
        if result['success']:
            if result['output'].strip():
                output_parts.append(f"Code block {i} output:\n{result['output']}")
            else:
                output_parts.append(f"Code block {i}: Executed successfully (no output)")
            if detailed:
                output_preview = result['output'][:100] if result['output'] else "(no output)"
                print(f"  Block {i}: Success - {output_preview}")
        else:
            output_parts.append(f"Code block {i} error:\n{result['error']}")
            if detailed:
                print(f"  Block {i}: Error - {result['error'][:100]}")
    
    # Append execution results to response
    if output_parts:
        code_output = "\n".join(output_parts)
        response_with_output = f"{response}\n\n[Code Execution Results]\n{code_output}"
    else:
        response_with_output = response
    
    return response_with_output, exec_results, len(exec_results) > 0


def determine_case_type(first_correct: bool, final_correct: bool, iterations_count: int, final_verdict: str = "") -> str:
    """
    Determine case type with clear semantics.
    
    Case types:
    - FIRST_TRY_SUCCESS: First answer correct, confirmed in first iteration
    - IMPROVED: First answer wrong, final answer correct
    - DEGRADED: First answer correct, final answer wrong
    - FAILED: Both first and final answers wrong
    - EVENTUALLY_CONFIRMED: First answer correct, but took multiple iterations to confirm
    - CONSISTENT_CORRECT: Answer was consistent across iterations and correct
    - CONSISTENT_WRONG: Answer was consistent but wrong
    """
    # Check if stopped due to consistency
    is_consistent = final_verdict.startswith("CONSISTENT_")
    
    if first_correct and final_correct:
        if iterations_count == 1:
            return "FIRST_TRY_SUCCESS"
        elif is_consistent:
            return "CONSISTENT_CORRECT"
        else:
            return "EVENTUALLY_CONFIRMED"
    elif not first_correct and final_correct:
        return "IMPROVED"
    elif first_correct and not final_correct:
        if is_consistent:
            return "CONSISTENT_WRONG"
        return "DEGRADED"
    else:
        if is_consistent:
            return "CONSISTENT_WRONG"
        return "FAILED"


def run_solver_checker_with_tools_workflow_v2(
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
    apply_chat_template: bool = False,
    consistency_threshold: int = 2
) -> Dict:
    """
    Run Solver-Checker workflow with tool execution (V2 - improved)
    
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
        consistency_threshold: Stop if solver gives same answer N times consecutively (default: 2)
    
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
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-CHECKER WITH TOOLS WORKFLOW V2")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Solver Tools: {'Enabled' if enable_solver_tools else 'Disabled'}")
        print(f"Checker Tools: {'Enabled' if enable_checker_tools else 'Disabled'}")
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
        # Build solver prompt (avoid triple backticks in instruction which confuse some models)
        tool_instruction = ""
        if enable_solver_tools:
            tool_instruction = "\n\nYou may use Python code to help with calculations. Show your reasoning step by step."
        
        if iteration_num == 1:
            solver_prompt = format_prompt_standard(question, dataset_name) + tool_instruction
        else:
            # Incorporate last answer + feedback from checker (minimal context, saves tokens)
            last_answer = solver_answers[-1] if solver_answers else "unknown"
            solver_prompt = format_prompt_standard(question, dataset_name)
            solver_prompt += f"\n\nYour previous answer was: {last_answer}"
            solver_prompt += f"\n\nChecker feedback: {checker_feedback}"
            solver_prompt += "\n\nPlease reconsider the problem and provide a corrected solution."
            solver_prompt += tool_instruction
        
        # Apply chat template if enabled
        solver_prompt = apply_chat_template_if_enabled(solver_prompt, solver_tokenizer, apply_chat_template)
        
        if detailed:
            print(f"\n[Solver Turn]")
        
        # Generate solver response
        if hasattr(solver_model, 'generate_single'):
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
            solver_response = generate_response(
                solver_model,
                solver_tokenizer,
                solver_prompt,
                "standard",
                detailed
            )
        
        # Extract answer BEFORE adding execution results (avoid interference)
        solver_answer = extract_answer(solver_response)
        
        # Execute code in solver response if enabled
        solver_exec_results = []
        solver_tools_used = False
        
        if enable_solver_tools:
            if detailed:
                print(f"\n[Code Execution]")
            
            solver_response_with_output, solver_exec_results, solver_tools_used = execute_code_blocks(
                solver_response, detailed
            )
            
            if solver_tools_used:
                solver_response = solver_response_with_output
                
                # Add error note if any execution failed
                failed_blocks = [r for r in solver_exec_results if not r['success']]
                if failed_blocks:
                    error_note = f"\n\n[Note: {len(failed_blocks)} code block(s) failed to execute properly]"
                    solver_response += error_note
        
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        if detailed:
            print(f"\n[Solver Answer]: {solver_answer}")
        
        # ========== CHECKER TURN ==========
        if detailed:
            print(f"\n[Checker Turn]")
        
        # Build checker prompt (avoid triple backticks in instruction)
        checker_tool_instruction = ""
        if enable_checker_tools:
            checker_tool_instruction = "\n\nYou may use Python code to verify calculations or test the solution."
        
        checker_prompt = format_prompt_checker(
            question,
            solver_response,
            dataset_name
        ) + checker_tool_instruction
        
        # Apply chat template if enabled
        checker_prompt = apply_chat_template_if_enabled(checker_prompt, checker_tokenizer, apply_chat_template)
        
        # Generate checker response
        if hasattr(checker_model, 'generate_single'):
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
            checker_response = generate_response_checker(
                checker_model,
                checker_tokenizer,
                checker_prompt,
                detailed
            )
        
        # Execute code in checker response if enabled
        checker_exec_results = []
        checker_tools_used = False
        
        if enable_checker_tools:
            if detailed:
                print(f"\n[Checker Code Execution]")
            
            checker_response_with_output, checker_exec_results, checker_tools_used = execute_code_blocks(
                checker_response, detailed
            )
            
            if checker_tools_used:
                checker_response = checker_response_with_output
                
                # Check if any checker execution failed
                checker_failed_blocks = [r for r in checker_exec_results if not r['success']]
                if checker_failed_blocks and detailed:
                    print(f"  [Warning: Checker's verification code had {len(checker_failed_blocks)} error(s)]")
        
        checker_responses.append(checker_response)
        
        # Parse checker verdict - handle UNCLEAR properly
        raw_verdict = parse_checker_verdict(checker_response)
        
        # Normalize verdict with proper UNCLEAR handling
        if raw_verdict == "CORRECT":
            checker_verdict = "CORRECT"
        elif raw_verdict == "UNCLEAR":
            checker_verdict = "UNCLEAR"
        else:
            # Default to INCORRECT for any other value (including None, empty, etc.)
            checker_verdict = "INCORRECT"
        
        checker_verdicts.append(checker_verdict)
        
        # Extract feedback for next iteration based on verdict
        if checker_verdict == "INCORRECT":
            checker_feedback = parse_checker_tip(checker_response)
            if not checker_feedback:
                checker_feedback = "The previous solution was incorrect. Please reconsider the problem and check your calculations."
        elif checker_verdict == "UNCLEAR":
            checker_feedback = parse_checker_tip(checker_response)
            if not checker_feedback:
                checker_feedback = "The previous solution was unclear or incomplete. Please provide clearer reasoning and explicit steps."
        else:
            checker_feedback = ""
        
        if detailed:
            print(f"\n[Checker Verdict]: {checker_verdict}")
            if checker_feedback:
                print(f"[Feedback]: {checker_feedback[:100]}...")
        
        # Check actual correctness
        is_actually_correct = False
        if solver_answer:
            is_actually_correct = check_answer(solver_answer, ground_truth)
        
        # Store iteration data - use actual execution results for tool usage detection
        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "checker_feedback": checker_feedback,
            "solver_tools_used": solver_tools_used,
            "checker_tools_used": checker_tools_used,
            "solver_exec_results": solver_exec_results,
            "checker_exec_results": checker_exec_results,
            "is_actually_correct": is_actually_correct
        }
        
        iterations.append(iteration_data)
        
        # Decision logic
        if checker_verdict == "CORRECT":
            if solver_answer:
                predicted_answer = solver_answer
                final_verdict = "CORRECT"
                if detailed:
                    print(f"\n[Checker confirmed CORRECT, stopping iteration]")
                break
        
        # Answer consistency check: if solver gives same answer N times, trust it
        if consistency_threshold > 0 and len(solver_answers) >= consistency_threshold:
            recent_answers = solver_answers[-consistency_threshold:]
            # Normalize answers for comparison (strip whitespace)
            normalized = [a.strip() if a else "" for a in recent_answers]
            if len(set(normalized)) == 1 and normalized[0]:
                predicted_answer = solver_answer
                final_verdict = f"CONSISTENT_{consistency_threshold}"
                if detailed:
                    print(f"\n[Answer consistency: same answer {consistency_threshold} times, stopping iteration]")
                break
        
        # For UNCLEAR, we also want to give solver another chance
        # INCORRECT already handled - will continue to next iteration
        
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
    
    # Determine case type with clear semantics
    first_answer = solver_answers[0] if solver_answers else None
    first_correct = check_answer(first_answer, ground_truth) if first_answer else False
    case_type = determine_case_type(first_correct, final_correct, len(iterations), final_verdict or "")
    
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
            "checker_tools": enable_checker_tools,
            "consistency_threshold": consistency_threshold
        }
    }


# For testing
if __name__ == "__main__":
    print("Solver-Checker with Tools V2 - Quick Test")
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
        
        print(f"\nRunning Solver-Checker with Tools V2 workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_solver_checker_with_tools_workflow_v2(
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

