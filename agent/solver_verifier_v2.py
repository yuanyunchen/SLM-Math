"""
Solver-Verifier V2 Agent
带代码自验证 + Verifier 仲裁的工作流 V2

V2 改进：
- 即使 code == boxed 一致，也运行 Verifier 验证一次
- 避免两者都错但一致的情况被漏掉

工作流程：
1. Solver 生成 reasoning + code
2. 执行 code 得到 code_result，提取 boxed_answer
3. 如果一致 → 运行 Verifier 验证
   - VERIFIED → 采纳
   - FAILED → 下一轮
4. 如果不一致 → Verifier 仲裁
   - 验证 code_result → VERIFIED 采纳
   - 验证 boxed_answer → VERIFIED 采纳
   - 都 FAILED → 下一轮
5. 下一轮反馈含：reasoning + 执行结果/报错 + Verifier 反馈
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple
import re
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY,
    CHECKER_MAX_TOKENS, CHECKER_TEMPERATURE, CHECKER_TOP_P, CHECKER_REPETITION_PENALTY
)


def extract_reasoning_without_code(response: str) -> str:
    """Extract reasoning from response, removing code blocks, boxed answers, and LaTeX."""
    reasoning = re.sub(r'```python.*?```', '[code]', response, flags=re.DOTALL)
    reasoning = re.sub(r'```output.*?```', '', reasoning, flags=re.DOTALL)
    reasoning = re.sub(r'```.*?```', '', reasoning, flags=re.DOTALL)
    reasoning = re.sub(r'\[Code Execution Results\].*', '', reasoning, flags=re.DOTALL)
    # Remove \boxed{} answers to prevent confusion in next iteration
    reasoning = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '[answer]', reasoning)
    # Remove LaTeX block math \[...\] to prevent model confusion
    reasoning = re.sub(r'\\\[.*?\\\]', '[math]', reasoning, flags=re.DOTALL)
    # Remove LaTeX inline math \(...\)
    reasoning = re.sub(r'\\\(.*?\\\)', '[expr]', reasoning)
    reasoning = re.sub(r'\n\s*\n\s*\n', '\n\n', reasoning)
    # Truncate at sentence boundary to avoid incomplete sentences causing EOS
    max_len = 800
    if len(reasoning) > max_len:
        # Find last sentence end before max_len
        truncated = reasoning[:max_len]
        last_period = max(truncated.rfind('. '), truncated.rfind('.\n'), truncated.rfind('。'))
        if last_period > max_len // 2:  # Only truncate at period if it's not too early
            reasoning = truncated[:last_period + 1] + " [truncated]"
        else:
            reasoning = truncated + "..."
    return reasoning.strip()


def format_code_execution_feedback(exec_results: List[Dict]) -> str:
    """Format code execution results or errors for feedback."""
    if not exec_results:
        return "No code was executed."
    
    feedback_parts = []
    for i, result in enumerate(exec_results, 1):
        if result['success']:
            output = result['output'].strip()
            if output:
                if len(output) > 200:
                    output = output[:200] + "..."
                feedback_parts.append(f"Code output: {output}")
            else:
                feedback_parts.append("Code executed successfully (no output)")
        else:
            error = result['error'].strip()
            if len(error) > 300:
                error = error[:300] + "..."
            feedback_parts.append(f"Code ERROR: {error}")
    
    return "\n".join(feedback_parts)


def extract_code_and_execute(response: str, detailed: bool = False) -> Tuple[str, List[Dict], bool]:
    """Extract and execute Python code blocks from response."""
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
            if detailed:
                output_preview = result['output'][:100] if result['output'] else "(no output)"
                print(f"  Block {i}: Success - {output_preview}")
        else:
            output_parts.append(f"Code block {i} error:\n{result['error']}")
            if detailed:
                print(f"  Block {i}: Error - {result['error'][:100]}")
    
    if output_parts:
        code_output = "\n".join(output_parts)
        response_with_output = f"{response}\n\n[Code Execution Results]\n{code_output}"
    else:
        response_with_output = response
    
    return response_with_output, exec_results, len(exec_results) > 0


def format_verifier_prompt(question: str, solver_answer: str) -> str:
    """Minimal verification: just calculate and print result."""
    return f"""{question}
```python
# Calculate step by step and print the final answer
"""


def parse_verification_result(exec_results: List[Dict], solver_answer: str = None) -> Tuple[str, str]:
    """Compare verifier's calculated answer with solver's answer."""
    if not exec_results:
        return "ERROR", "No code executed"
    
    last_result = exec_results[-1]
    
    if not last_result['success']:
        return "ERROR", f"Error: {last_result['error'][:80]}"
    
    output = last_result['output'].strip()
    if not output:
        return "ERROR", "No output"
    
    # Extract last number from output
    numbers = re.findall(r'[-+]?\d*\.?\d+', output.split('\n')[-1])
    if not numbers:
        return "UNCLEAR", f"No number in output"
    
    calc = numbers[-1]
    
    # Normalize for comparison
    def norm(x):
        x = str(x).strip()
        if '.' in x:
            x = x.rstrip('0').rstrip('.')
        return x
    
    if norm(calc) == norm(solver_answer):
        return "VERIFIED", f"both={calc}"
    else:
        return "FAILED", f"calc={calc}, given={solver_answer}"


def run_solver_verifier_v2_workflow(
    question: str,
    ground_truth: str,
    solver_model,
    solver_tokenizer,
    verifier_model=None,
    verifier_tokenizer=None,
    max_iterations: int = 3,
    detailed: bool = False,
    dataset_name: str = "",
    enable_solver_tools: bool = True,
    consistency_threshold: int = 2
) -> Dict:
    """
    Run Solver-Verifier V2 workflow.
    
    V2: Always run Verifier, even when code == boxed.
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response
    
    if verifier_model is None:
        verifier_model = solver_model
        verifier_tokenizer = solver_tokenizer
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-VERIFIER V2 WORKFLOW")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"{'='*80}\n")
    
    iterations = []
    solver_responses = []
    solver_answers = []
    verifier_responses = []
    verifier_verdicts = []
    
    predicted_answer = None
    final_verdict = None
    verification_feedback = ""
    
    for iteration_num in range(1, max_iterations + 1):
        if detailed:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration_num}/{max_iterations}")
            print(f"{'='*80}")
        
        # ========== SOLVER TURN ==========
        tool_instruction = ""
        if enable_solver_tools:
            tool_instruction = "\n\nYou may use Python code to help with calculations. Show your reasoning step by step."
        
        if iteration_num == 1:
            solver_prompt = format_prompt_standard(question, dataset_name) + tool_instruction
        else:
            solver_prompt = format_prompt_standard(question, dataset_name)
            solver_prompt += f"\n\n--- Previous Attempt Feedback ---\n{verification_feedback}"
            solver_prompt += "\n\nPlease carefully review the feedback and provide a corrected solution."
            solver_prompt += tool_instruction
        
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
        
        # Extract answer BEFORE adding execution results
        boxed_answer = extract_answer(solver_response)
        
        # Execute solver's code if present
        solver_exec_results = []
        solver_tools_used = False
        
        if enable_solver_tools:
            if detailed:
                print(f"\n[Solver Code Execution]")
            
            solver_response_with_output, solver_exec_results, solver_tools_used = extract_code_and_execute(
                solver_response, detailed
            )
            
            if solver_tools_used:
                solver_response = solver_response_with_output
        
        solver_responses.append(solver_response)
        
        # Extract reasoning and code feedback
        solver_reasoning = extract_reasoning_without_code(solver_response)
        code_exec_feedback = format_code_execution_feedback(solver_exec_results)
        
        # Extract code result
        code_result = None
        if solver_exec_results:
            last_exec = solver_exec_results[-1]
            if last_exec['success'] and last_exec['output'].strip():
                code_output = last_exec['output'].strip()
                last_line = code_output.split('\n')[-1].strip()
                numbers = re.findall(r'[-+]?\d*\.?\d+', last_line)
                if numbers:
                    code_result = numbers[-1]
                    if detailed:
                        print(f"\n[Code Result]: {code_result}")
        
        if detailed:
            print(f"\n[Boxed Answer]: {boxed_answer}")
        
        # ========== VERIFICATION LOGIC (V2: Always verify) ==========
        verifier_response = ""
        verifier_verdict = "NONE"
        verification_feedback = ""
        solver_answer = boxed_answer  # Default to boxed
        
        def run_verifier(answer_to_verify: str) -> Tuple[str, str, str]:
            """Run Verifier to independently solve and compare."""
            v_prompt = format_verifier_prompt(question, answer_to_verify)
            
            if hasattr(verifier_model, 'generate_single'):
                v_response = verifier_model.generate_single(
                    v_prompt,
                    max_new_tokens=CHECKER_MAX_TOKENS,
                    temperature=CHECKER_TEMPERATURE,
                    do_sample=DO_SAMPLE,
                    top_p=CHECKER_TOP_P,
                    repetition_penalty=CHECKER_REPETITION_PENALTY,
                    detailed=detailed
                )
            else:
                v_response = generate_response(
                    verifier_model,
                    verifier_tokenizer,
                    v_prompt,
                    "standard",
                    detailed
                )
            
            v_response_with_output, v_exec_results, _ = extract_code_and_execute(v_response, detailed)
            v_verdict, v_feedback = parse_verification_result(v_exec_results, answer_to_verify)
            return v_response_with_output, v_verdict, v_feedback
        
        if code_result is not None and boxed_answer:
            # Both code and boxed exist
            code_norm = code_result.rstrip('0').rstrip('.') if '.' in code_result else code_result
            boxed_norm = boxed_answer.rstrip('0').rstrip('.') if '.' in boxed_answer else boxed_answer
            is_consistent = (code_norm == boxed_norm or code_result == boxed_answer)
            
            if is_consistent:
                # V2: Even if consistent, run Verifier to check
                if detailed:
                    print(f"\n[Code and boxed consistent: {code_result}]")
                    print(f"[V2: Running Verifier to verify...]")
                
                verifier_response, verifier_verdict, v_feedback = run_verifier(boxed_answer)
                
                # Retry once if ERROR/UNCLEAR
                if verifier_verdict in ["ERROR", "UNCLEAR"]:
                    if detailed:
                        print(f"[Verifier {verifier_verdict}, retrying...]")
                    verifier_response, verifier_verdict, v_feedback = run_verifier(boxed_answer)
                
                if verifier_verdict == "VERIFIED":
                    verifier_verdict = "CONSISTENT_VERIFIED"
                    solver_answer = boxed_answer
                    if detailed:
                        print(f"[VERIFIED, accepting {boxed_answer}]")
                elif verifier_verdict in ["ERROR", "UNCLEAR"]:
                    # Verifier still failed after retry - trust consistent answer
                    verifier_verdict = "CONSISTENT_TRUSTED"
                    solver_answer = boxed_answer
                    if detailed:
                        print(f"[Verifier error after retry, trusting consistent answer: {boxed_answer}]")
                else:
                    # Consistent but Verifier says FAILED - need iteration
                    verifier_verdict = "CONSISTENT_FAILED"
                    verification_feedback = f"""Your previous reasoning:
{solver_reasoning}

{code_exec_feedback}
Your answer: {boxed_answer}

Verification FAILED: {v_feedback}
Your code and reasoning both gave {boxed_answer}, but verification suggests it may be incorrect.

Please re-solve this problem carefully."""
                    if detailed:
                        print(f"[FAILED, will iterate]")
            else:
                # Inconsistent - Verifier arbitrates
                if detailed:
                    print(f"\n[Inconsistent: code={code_result}, boxed={boxed_answer}]")
                    print(f"[Running Verifier to arbitrate...]")
                
                # First verify code_result
                if detailed:
                    print(f"\n[Verifying code result: {code_result}]")
                verifier_response, v1_verdict, v1_feedback = run_verifier(code_result)
                
                if v1_verdict == "VERIFIED":
                    verifier_verdict = "CODE_VERIFIED"
                    solver_answer = code_result
                    if detailed:
                        print(f"[Code result VERIFIED, using {code_result}]")
                else:
                    # Code failed, try boxed
                    if detailed:
                        print(f"[Code result FAILED, verifying boxed: {boxed_answer}]")
                    verifier_response, v2_verdict, v2_feedback = run_verifier(boxed_answer)
                    
                    if v2_verdict == "VERIFIED":
                        verifier_verdict = "BOXED_VERIFIED"
                        solver_answer = boxed_answer
                        if detailed:
                            print(f"[Boxed answer VERIFIED, using {boxed_answer}]")
                    else:
                        # Both failed
                        verifier_verdict = "BOTH_FAILED"
                        verification_feedback = f"""Your previous reasoning:
{solver_reasoning}

{code_exec_feedback}
Your boxed answer: {boxed_answer}

Verification result: Both your code result ({code_result}) and boxed answer ({boxed_answer}) appear incorrect.
Code verification: {v1_feedback}
Boxed verification: {v2_feedback}

Please carefully re-solve this problem."""
                        if detailed:
                            print(f"[Both FAILED, will iterate]")
        
        elif code_result is not None:
            # Only code result, no boxed
            if detailed:
                print(f"\n[No boxed answer, verifying code result: {code_result}]")
            verifier_response, verifier_verdict, v_feedback = run_verifier(code_result)
            
            if verifier_verdict == "VERIFIED":
                solver_answer = code_result
            else:
                verification_feedback = f"""Your previous reasoning:
{solver_reasoning}

{code_exec_feedback}

Verification failed: {v_feedback}

Please provide a clear \\boxed{{}} answer."""
        
        elif solver_exec_results and not solver_exec_results[-1]['success']:
            # Code execution failed with error
            verifier_verdict = "CODE_ERROR"
            verification_feedback = f"""Your previous reasoning:
{solver_reasoning}

{code_exec_feedback}

Please fix the code error and try again."""
            if detailed:
                print(f"\n[Code execution error, will iterate]")
        
        elif boxed_answer:
            # Only boxed answer, no code
            if detailed:
                print(f"\n[No code result, verifying boxed: {boxed_answer}]")
            verifier_response, verifier_verdict, v_feedback = run_verifier(boxed_answer)
            
            if verifier_verdict != "VERIFIED":
                verification_feedback = f"""Your previous reasoning:
{solver_reasoning}

Your answer: {boxed_answer}

Verification failed: {v_feedback}

Please re-check your calculations."""
        
        else:
            verifier_verdict = "ERROR"
            verification_feedback = f"""Your previous reasoning:
{solver_reasoning}

No answer was found. Please provide a clear \\boxed{{}} answer."""
        
        # Store answer
        if solver_answer:
            solver_answers.append(solver_answer)
        
        if detailed:
            print(f"\n[Final Answer]: {solver_answer}")
            print(f"[Verdict]: {verifier_verdict}")
        
        verifier_responses.append(verifier_response)
        verifier_verdicts.append(verifier_verdict)
        
        # Check actual correctness
        is_actually_correct = False
        if solver_answer:
            is_actually_correct = check_answer(solver_answer, ground_truth)
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "boxed_answer": boxed_answer,
            "code_result": code_result,
            "verifier_response": verifier_response,
            "verifier_verdict": verifier_verdict,
            "verification_feedback": verification_feedback,
            "solver_tools_used": solver_tools_used,
            "solver_exec_results": solver_exec_results,
            "is_actually_correct": is_actually_correct
        }
        
        iterations.append(iteration_data)
        
        # ========== DECISION LOGIC ==========
        
        # If verified or trusted, accept the answer
        if verifier_verdict in ["CONSISTENT_VERIFIED", "CONSISTENT_TRUSTED", "CODE_VERIFIED", "BOXED_VERIFIED", "VERIFIED"]:
            if solver_answer:
                predicted_answer = solver_answer
                final_verdict = verifier_verdict
                if detailed:
                    print(f"\n[Answer accepted ({verifier_verdict}), stopping iteration]")
                break
        
        # Answer consistency check (across iterations)
        if consistency_threshold > 0 and len(solver_answers) >= consistency_threshold:
            recent_answers = solver_answers[-consistency_threshold:]
            normalized = [a.strip() if a else "" for a in recent_answers]
            if len(set(normalized)) == 1 and normalized[0]:
                predicted_answer = solver_answer
                final_verdict = f"ANSWER_CONSISTENT_{consistency_threshold}"
                if detailed:
                    print(f"\n[Answer consistency: same answer {consistency_threshold} times, stopping]")
                break
        
        # Max iterations check
        if iteration_num >= max_iterations:
            if detailed:
                print(f"\nReached max iterations ({max_iterations})")
            break
    
    # If no verified verdict, use last valid answer
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
    
    is_verified = final_verdict in ["CONSISTENT_VERIFIED", "CODE_VERIFIED", "BOXED_VERIFIED", "VERIFIED"]
    is_answer_consistent = final_verdict and final_verdict.startswith("ANSWER_CONSISTENT_")
    
    if first_correct and final_correct:
        if len(iterations) == 1 and is_verified:
            case_type = "FIRST_TRY_VERIFIED"
        elif is_answer_consistent:
            case_type = "ANSWER_CONSISTENT_CORRECT"
        elif is_verified:
            case_type = "EVENTUALLY_VERIFIED"
        else:
            case_type = "EVENTUALLY_CONFIRMED"
    elif not first_correct and final_correct:
        case_type = "IMPROVED"
    elif first_correct and not final_correct:
        if is_answer_consistent:
            case_type = "ANSWER_CONSISTENT_WRONG"
        else:
            case_type = "DEGRADED"
    else:
        if is_answer_consistent:
            case_type = "ANSWER_CONSISTENT_WRONG"
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
        "verifier_verdicts": verifier_verdicts,
        "first_answer": first_answer,
        "first_correct": first_correct,
        "case_type": case_type,
        "config": {
            "enable_solver_tools": enable_solver_tools,
            "consistency_threshold": consistency_threshold
        }
    }


# For testing
if __name__ == "__main__":
    print("Solver-Verifier V2 - Quick Test")
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
        
        print(f"\nRunning Solver-Verifier V2 workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_solver_verifier_v2_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            solver_model=model,
            solver_tokenizer=tokenizer,
            max_iterations=3,
            detailed=True,
            dataset_name="gsm8k",
            enable_solver_tools=True
        )
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Expected: {result['ground_truth']}")
        print(f"Correct: {result['final_correct']}")
        print(f"Verdict: {result['final_verdict']}")
        print(f"Case Type: {result['case_type']}")
        print(f"Iterations: {result['total_iterations']}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

