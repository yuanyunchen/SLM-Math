"""
Solver-Verifier (Multi-Agent Classifier - Code Only)

Variant: the verifier only looks at solver-generated code and its execution
output. First check whether the code result matches the boxed answer; if it
matches, accept directly. If not, let the verifier judge the correctness of the
code (CORRECT / INCORRECT / UNCLEAR). Only CORRECT is accepted; otherwise go to
the next round.
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
from utils.python_code_execution import extract_python_code_blocks, execute_python_code


def extract_reasoning_without_code(response: str) -> str:
    """Extract reasoning from response, removing code blocks, boxed answers, and LaTeX."""
    reasoning = re.sub(r'```python.*?```', '[code]', response, flags=re.DOTALL)
    reasoning = re.sub(r'```output.*?```', '', reasoning, flags=re.DOTALL)
    reasoning = re.sub(r'```.*?```', '', reasoning, flags=re.DOTALL)
    reasoning = re.sub(r'\[Code Execution Results\].*', '', reasoning, flags=re.DOTALL)
    reasoning = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '[answer]', reasoning)
    reasoning = re.sub(r'\\\[.*?\\\]', '[math]', reasoning, flags=re.DOTALL)
    reasoning = re.sub(r'\\\(.*?\\\)', '[expr]', reasoning)
    reasoning = re.sub(r'\n\s*\n\s*\n', '\n\n', reasoning)
    max_len = 800
    if len(reasoning) > max_len:
        truncated = reasoning[:max_len]
        last_period = max(truncated.rfind('. '), truncated.rfind('.\n'), truncated.rfind('。'))
        if last_period > max_len // 2:
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


def format_verifier_prompt_code(question: str, given_answer: str, code_snippet: str, code_output: str) -> str:
    """Verifier prompt when checking code-derived answer."""
    return f"""You are a verifier. Decide if the given answer is correct.

Problem:
{question}

Solver code (latest):
```python
{code_snippet.strip() if code_snippet else "(no code provided)"}
```

Code execution output (last line is the candidate answer):
{code_output if code_output else "(no execution output)"}

Candidate answer: {given_answer}

Label with one of:
- CORRECT (confident the answer matches the problem)
- INCORRECT (confident the answer is wrong)
- UNCLEAR (not enough evidence)

Respond with exactly one label: CORRECT, INCORRECT, or UNCLEAR."""


def parse_verifier_label(response: str) -> str:
    """Parse verifier label from text."""
    up = response.upper()
    if "CORRECT" in up and "INCORRECT" not in up:
        return "CORRECT"
    if "INCORRECT" in up:
        return "INCORRECT"
    if "UNCLEAR" in up or "UNSURE" in up or "NOT SURE" in up:
        return "UNCLEAR"
    return "UNCLEAR"


def run_solver_verifier_workflow_multi(
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
    Run Solver-Verifier workflow with classifier-style verifier (code only).
    Other logic follows solver_verifier.py but the verifier only inspects code.
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response

    if verifier_model is None:
        verifier_model = solver_model
        verifier_tokenizer = solver_tokenizer

    def _normalize_simple(val: str) -> str:
        """Lightweight numeric normalization for comparing code vs boxed."""
        s = val.strip()
        return s.rstrip('0').rstrip('.') if '.' in s else s

    def _truncate_for_feedback(text: str, max_len: int = 800) -> str:
        """Truncate long text to keep feedback short."""
        if text is None:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "... [truncated]"

    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-VERIFIER (Classifier Verifier)")
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

        solver_answer = extract_answer(solver_response)
        solver_exec_results = []
        solver_tools_used = False
        solver_response_with_output = solver_response

        if enable_solver_tools:
            if detailed:
                print(f"\n[Solver Code Execution]")
            solver_response_with_output, solver_exec_results, solver_tools_used = extract_code_and_execute(
                solver_response, detailed
            )
            if solver_tools_used:
                solver_response = solver_response_with_output

        solver_responses.append(solver_response)
        solver_reasoning = extract_reasoning_without_code(solver_response)
        solver_code_blocks = extract_python_code_blocks(solver_response)

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

        verifier_response = ""
        verifier_verdict = "NONE"
        verification_feedback = ""

        def run_verifier(answer_to_verify: str, code_snippet: str = "", code_output: str = "") -> Tuple[str, str, str]:
            """Run classifier-style verifier (code only). Returns (response, verdict_label, feedback)."""
            v_prompt = format_verifier_prompt_code(question, answer_to_verify, code_snippet, code_output)

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

            v_label = parse_verifier_label(v_response)
            return v_response, v_label, ""

        code_exec_feedback = format_code_execution_feedback(solver_exec_results)
        latest_code = solver_code_blocks[-1] if solver_code_blocks else ""
        code_output_text = solver_exec_results[-1]['output'] if solver_exec_results else ""

        code_box_match = False
        if code_result is not None and solver_answer:
            code_norm = _normalize_simple(code_result)
            answer_norm = _normalize_simple(solver_answer)
            if code_norm == answer_norm or code_result == solver_answer:
                code_box_match = True
                verifier_verdict = "CODE_BOX_MATCH"
                solver_answer = code_result
                verification_feedback = ""
                if detailed:
                    print(f"\n[Code result matches boxed answer: {code_result}]")

        if code_box_match:
            pass
        elif code_result is not None:
            if detailed:
                print(f"\n[Verifying code result: {code_result}]")
            verifier_response, v_label, _ = run_verifier(
                code_result, code_snippet=latest_code, code_output=code_output_text
            )
            truncated_vresp = _truncate_for_feedback(verifier_response)

            if v_label == "CORRECT":
                verifier_verdict = "CODE_VERIFIED"
                solver_answer = code_result
                if detailed:
                    print(f"[Code VERIFIED, using {code_result}]")
            elif v_label == "INCORRECT":
                verifier_verdict = "CODE_FAILED"
                mismatch_note = ""
                if solver_answer:
                    mismatch_note = f"\n\nCode result ({code_result}) differs from your boxed answer ({solver_answer})."
                reasoning_snippet = solver_reasoning
                code_snippet = latest_code.strip() if latest_code else "(no code)"
                verification_feedback = f"""Your previous code appears incorrect.{mismatch_note}

[Reasoning]
{reasoning_snippet}

[Code]
```python
{code_snippet}
```

[Verifier]
Label: {v_label}
Response: {truncated_vresp}

{code_exec_feedback}

Please fix the code and provide the correct answer."""
            else:
                verifier_verdict = "CODE_UNCLEAR"
                mismatch_note = ""
                if solver_answer:
                    mismatch_note = f"\n\nCode result ({code_result}) differs from your boxed answer ({solver_answer})."
                reasoning_snippet = solver_reasoning
                code_snippet = latest_code.strip() if latest_code else "(no code)"
                verification_feedback = f"""The verifier is unsure about your code result.{mismatch_note}

[Reasoning]
{reasoning_snippet}

[Code]
```python
{code_snippet}
```

[Verifier]
Label: {v_label}
Response: {truncated_vresp}

{code_exec_feedback}

Please refine your code and provide a clear answer."""

        elif solver_exec_results and not solver_exec_results[-1]['success']:
            verifier_verdict = "CODE_ERROR"
            verification_feedback = f"""Your code execution failed.

{code_exec_feedback}

Please fix the code error and try again."""
            if detailed:
                print(f"\n[Code execution error, will iterate]")

        else:
            verifier_verdict = "NO_CODE"
            verification_feedback = """No executable code was found. Please provide Python code to compute the answer."""

        if solver_answer:
            solver_answers.append(solver_answer)

        if detailed:
            print(f"\n[Final Answer]: {solver_answer}")
            print(f"[Verdict]: {verifier_verdict}")

        verifier_responses.append(verifier_response)
        verifier_verdicts.append(verifier_verdict)

        if detailed:
            print(f"\n[Verifier Verdict]: {verifier_verdict}")
            if verification_feedback:
                print(f"[Feedback]: {verification_feedback[:100]}...")

        is_actually_correct = False
        if solver_answer:
            is_actually_correct = check_answer(solver_answer, ground_truth)

        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "boxed_answer": None,
            "code_result": code_result,
            "verifier_response": verifier_response,
            "verifier_verdict": verifier_verdict,
            "verification_feedback": verification_feedback,
            "solver_tools_used": solver_tools_used,
            "solver_exec_results": solver_exec_results,
            "is_actually_correct": is_actually_correct
        }

        iterations.append(iteration_data)

        if verifier_verdict in ["CODE_VERIFIED", "CODE_BOX_MATCH"]:
            if solver_answer:
                predicted_answer = solver_answer
                final_verdict = verifier_verdict
                if detailed:
                    print(f"\n[Answer accepted ({verifier_verdict}), stopping iteration]")
                break

        if consistency_threshold > 0 and len(solver_answers) >= consistency_threshold:
            recent_answers = solver_answers[-consistency_threshold:]
            normalized = [a.strip() if a else "" for a in recent_answers]
            if len(set(normalized)) == 1 and normalized[0]:
                predicted_answer = solver_answer
                final_verdict = f"CONSISTENT_{consistency_threshold}"
                if detailed:
                    print(f"\n[Answer consistency: same answer {consistency_threshold} times, stopping]")
                break

        if iteration_num >= max_iterations:
            if detailed:
                print(f"\nReached max iterations ({max_iterations})")
            break

    if predicted_answer is None:
        for i in range(len(iterations) - 1, -1, -1):
            iter_answer = iterations[i]['solver_answer']
            if iter_answer and iter_answer.strip():
                predicted_answer = iter_answer
                final_verdict = f"LAST_VALID_ITER_{iterations[i]['iteration']}"
                break
        if predicted_answer is None:
            final_verdict = "NO_VALID_ANSWER"

    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)

    first_answer = solver_answers[0] if solver_answers else None
    first_correct = check_answer(first_answer, ground_truth) if first_answer else False

    is_verified = final_verdict in ["CONSISTENT", "CODE_VERIFIED", "CODE_BOX_MATCH", "BOXED_VERIFIED", "VERIFIED"]
    is_answer_consistent = final_verdict and final_verdict.startswith("CONSISTENT_")

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
        if final_verdict == "CODE_VERIFIED":
            case_type = "CODE_CORRECTED"
        elif final_verdict == "BOXED_VERIFIED":
            case_type = "BOXED_CORRECTED"
        else:
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


if __name__ == "__main__":
    print("Solver-Verifier (Classifier) - Quick Test")
    print("=" * 80)

    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_ground_truth = "18"

    try:
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"

        print(f"\nLoading model: {model_name}")
        from models.inference import load_model
        model, tokenizer = load_model(model_name, base_path)

        print(f"\nRunning Solver-Verifier workflow (classifier)...")
        result = run_solver_verifier_workflow_multi(
            question=test_question,
            ground_truth=test_ground_truth,
            solver_model=model,
            solver_tokenizer=tokenizer,
            max_iterations=2,
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

