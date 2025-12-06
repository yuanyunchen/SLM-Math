"""
Solver-Coder: Code-Based Math Problem Solving with Self-Verification

Features:
1. Code generation: Solver writes Python code to solve math problems
2. Self-verification: Solver sees execution result and decides if correct
3. Checker verification: Independent verification code to double-check
4. Debug iterations: Fix errors and refine solutions

Workflow:
1. Code Solver generates Python code
2. Execute code, get result
3. If error -> feed code + error to Solver for fix
4. If success -> Solver self-verifies the result
   - If CORRECT -> accept answer
   - If WRONG -> Solver provides corrected code
5. Checker independently verifies (optional)
6. Iterate until valid answer or max iterations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple, Optional
import re
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY,
    CHECKER_MAX_TOKENS, CHECKER_TEMPERATURE, CHECKER_TOP_P, CHECKER_REPETITION_PENALTY
)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def format_solver_prompt_initial(question: str, dataset_name: str = "") -> str:
    """Initial prompt for Code Solver."""
    return f"""Solve this math problem by writing Python code.

Problem: {question}

Instructions:
1. Write Python code to solve this step by step
2. Use clear variable names with comments
3. The LAST line must be: print(answer)
4. Only print the final numerical answer

```python
"""


def format_solver_prompt_fix_error(
    question: str, 
    previous_code: str, 
    error_message: str,
    iteration: int
) -> str:
    """Prompt when previous code had errors."""
    if len(error_message) > 400:
        error_message = error_message[:400] + "..."
    
    return f"""Your code had an error. Fix it.

Problem: {question}

Your Code:
```python
{previous_code}
```

Error: {error_message}

Write the corrected code:
```python
"""


def format_solver_self_verify_prompt(
    question: str,
    code: str,
    code_output: str,
    iteration: int
) -> str:
    """
    Let solver see the result and self-verify.
    Solver decides if the answer is correct or needs fixing.
    """
    return f"""Check if your solution is correct.

Problem: {question}

Your Code:
```python
{code}
```

Execution Result: {code_output}

Review your solution:
1. Does the code correctly model the problem?
2. Did you use all given information correctly?
3. Is the calculation logic correct?
4. Does the answer make sense (positive for money, reasonable magnitude)?

If the answer is CORRECT, respond with exactly:
CORRECT: [the answer]

If the answer is WRONG, write corrected code:
```python
"""


def format_checker_code_verify_prompt(question: str, answer: str) -> str:
    """
    Checker writes verification code to independently check the answer.
    """
    return f"""Verify this answer by computing it independently.

Problem: {question}
Given Answer: {answer}

Write Python code to:
1. Extract numbers from the problem
2. Calculate the answer step by step
3. Compare with the given answer
4. Print "VERIFIED" if correct, or "FAILED: expected X" if wrong

```python
"""


# =============================================================================
# CODE EXTRACTION AND EXECUTION
# =============================================================================

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from model response."""
    # Try ```python ... ``` block
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    # Try generic code block
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        code_blocks = [m for m in matches if not m.strip().isdigit() and len(m.strip()) > 10]
        if code_blocks:
            return code_blocks[0].strip()
    
    # Handle raw code without markers
    lines = response.strip().split('\n')
    code_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith('output:') or line_lower.startswith('```output'):
            break
        if line_lower.startswith('explanation:') or line_lower.startswith('correct:'):
            break
        if line.strip().startswith('```') and code_lines:
            break
        
        stripped = line.strip()
        if stripped:
            is_code = (
                stripped.startswith('#') or
                stripped.startswith('import ') or
                stripped.startswith('from ') or
                stripped.startswith('def ') or
                stripped.startswith('for ') or
                stripped.startswith('while ') or
                stripped.startswith('if ') or
                stripped.startswith('elif ') or
                stripped.startswith('else:') or
                stripped.startswith('print(') or
                '=' in stripped or
                stripped.endswith(':')
            )
            
            if is_code or code_lines:
                code_lines.append(line)
        elif code_lines:
            code_lines.append(line)
    
    if code_lines:
        code = '\n'.join(code_lines).strip()
        if 'print(' in code or '=' in code:
            if code.endswith('```'):
                code = code[:-3].strip()
            return code
    
    return None


def execute_code(code: str, timeout: int = 10) -> Dict:
    """Execute Python code safely."""
    from utils.python_code_execution import execute_python_code
    return execute_python_code(code, timeout=timeout)


def extract_answer_from_output(output: str) -> Optional[str]:
    """Extract numerical answer from code output."""
    if not output:
        return None
    
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    if not lines:
        return None
    
    last_line = lines[-1]
    numbers = re.findall(r'[-+]?\d*\.?\d+', last_line)
    if numbers:
        return numbers[-1]
    
    return last_line


def parse_self_verify_response(response: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse solver's self-verification response.
    
    Returns:
        (verdict, answer, new_code)
        verdict: "CORRECT", "WRONG", or "UNCLEAR"
    """
    response_upper = response.upper()
    
    # Check if solver says CORRECT
    if "CORRECT:" in response_upper:
        match = re.search(r'CORRECT[:\s]*([^\n]+)', response, re.IGNORECASE)
        if match:
            answer_text = match.group(1).strip()
            numbers = re.findall(r'[-+]?\d*\.?\d+', answer_text)
            if numbers:
                return "CORRECT", numbers[-1], None
        return "CORRECT", None, None
    
    # Check if solver provided new code
    new_code = extract_code_from_response(response)
    if new_code:
        return "WRONG", None, new_code
    
    return "UNCLEAR", None, None


def parse_checker_verify_response(exec_result: Dict) -> Tuple[str, str]:
    """Parse checker's verification code output."""
    if not exec_result['success']:
        return "ERROR", exec_result['error'][:200]
    
    output = exec_result['output'].strip().upper()
    
    if "VERIFIED" in output:
        return "VERIFIED", ""
    elif "FAILED" in output:
        match = re.search(r'FAILED[:\s]*(.+)', exec_result['output'], re.IGNORECASE)
        reason = match.group(1).strip() if match else "Verification failed"
        return "FAILED", reason
    
    return "UNCLEAR", output[:100]


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_solver_coder_workflow(
    question: str,
    ground_truth: str,
    solver_model,
    solver_tokenizer,
    checker_model=None,
    checker_tokenizer=None,
    max_iterations: int = 3,
    detailed: bool = False,
    dataset_name: str = "",
    enable_checker: bool = True,
    enable_self_verify: bool = True,
    enable_checker_verify: bool = True,
    code_timeout: int = 10
) -> Dict:
    """
    Run the Solver-Coder workflow.
    
    Args:
        question: Math problem to solve
        ground_truth: Ground truth answer
        solver_model: Model for code generation
        solver_tokenizer: Tokenizer for solver
        checker_model: Model for verification (optional)
        checker_tokenizer: Tokenizer for checker
        max_iterations: Maximum iterations
        detailed: Verbose output
        dataset_name: Dataset name
        enable_checker: Legacy param (maps to enable_checker_verify)
        enable_self_verify: Enable self-verification phase
        enable_checker_verify: Enable checker verification phase
        code_timeout: Code execution timeout
    
    Returns:
        Dict with workflow results
    """
    from utils.prompt_utils import check_answer
    from models.inference import generate_response
    
    # Handle legacy param
    if not enable_checker:
        enable_checker_verify = False
    
    if checker_model is None:
        checker_model = solver_model
        checker_tokenizer = solver_tokenizer
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-CODER WORKFLOW")
        print(f"{'='*80}")
        print(f"Question: {question[:150]}...")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*80}\n")
    
    iterations = []
    all_codes = []
    all_answers = []
    
    predicted_answer = None
    final_status = None
    
    current_code = None
    current_error = None
    current_output = None
    
    for iteration_num in range(1, max_iterations + 1):
        if detailed:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration_num}/{max_iterations}")
            print(f"{'='*60}")
        
        # ========== PHASE 1: Generate/Fix Code ==========
        if iteration_num == 1:
            solver_prompt = format_solver_prompt_initial(question, dataset_name)
        elif current_error:
            solver_prompt = format_solver_prompt_fix_error(
                question, current_code, current_error, iteration_num
            )
        else:
            solver_prompt = format_solver_prompt_initial(question, dataset_name)
        
        if detailed:
            print(f"\n[Phase 1: Generating code...]")
        
        # Generate code
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
                solver_model, solver_tokenizer, solver_prompt, "standard", detailed
            )
        
        code = extract_code_from_response(solver_response)
        
        if not code:
            if detailed:
                print(f"[No code found in response]")
            current_code = None
            current_error = "No code generated. Please write Python code."
            iterations.append({
                "iteration": iteration_num,
                "phase": "generate",
                "solver_response": solver_response,
                "code": None,
                "exec_success": False,
                "exec_output": None,
                "exec_error": current_error,
                "extracted_answer": None,
                "self_verify_verdict": None,
                "checker_verdict": None
            })
            continue
        
        current_code = code
        all_codes.append(code)
        
        if detailed:
            print(f"\n[Code Generated]")
            print("-" * 40)
            print(code[:400] + ("..." if len(code) > 400 else ""))
            print("-" * 40)
        
        # ========== PHASE 2: Execute Code ==========
        if detailed:
            print(f"\n[Phase 2: Executing code...]")
        
        exec_result = execute_code(code, timeout=code_timeout)
        
        if not exec_result['success']:
            current_error = exec_result['error']
            current_output = None
            
            if detailed:
                print(f"[Execution FAILED]: {current_error[:200]}")
            
            iterations.append({
                "iteration": iteration_num,
                "phase": "execute",
                "solver_response": solver_response,
                "code": code,
                "exec_success": False,
                "exec_output": None,
                "exec_error": current_error,
                "extracted_answer": None,
                "self_verify_verdict": "ERROR",
                "checker_verdict": None
            })
            continue
        
        # Execution successful
        current_error = None
        current_output = exec_result['output']
        extracted_answer = extract_answer_from_output(current_output)
        all_answers.append(extracted_answer)
        
        if detailed:
            print(f"[Execution SUCCESS]")
            print(f"Output: {current_output}")
            print(f"Extracted Answer: {extracted_answer}")
        
        # ========== PHASE 3: Self-Verification ==========
        self_verify_verdict = "SKIPPED"
        new_code = None
        
        if enable_self_verify and extracted_answer:
            if detailed:
                print(f"\n[Phase 3: Self-verification...]")
            
            verify_prompt = format_solver_self_verify_prompt(
                question, code, current_output, iteration_num
            )
            
            if hasattr(solver_model, 'generate_single'):
                verify_response = solver_model.generate_single(
                    verify_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=CHECKER_TEMPERATURE,
                    do_sample=DO_SAMPLE,
                    top_p=CHECKER_TOP_P,
                    repetition_penalty=CHECKER_REPETITION_PENALTY,
                    detailed=detailed
                )
            else:
                verify_response = generate_response(
                    solver_model, solver_tokenizer, verify_prompt, "standard", detailed
                )
            
            self_verify_verdict, verified_answer, new_code = parse_self_verify_response(verify_response)
            
            if detailed:
                print(f"[Self-verify verdict]: {self_verify_verdict}")
            
            if self_verify_verdict == "CORRECT":
                if verified_answer:
                    extracted_answer = verified_answer
                if detailed:
                    print(f"[Solver confirms answer: {extracted_answer}]")
            
            elif self_verify_verdict == "WRONG" and new_code:
                if detailed:
                    print(f"[Solver provided corrected code, executing...]")
                
                new_exec_result = execute_code(new_code, timeout=code_timeout)
                
                if new_exec_result['success']:
                    new_answer = extract_answer_from_output(new_exec_result['output'])
                    if new_answer:
                        extracted_answer = new_answer
                        all_answers.append(extracted_answer)
                        all_codes.append(new_code)
                        code = new_code
                        current_output = new_exec_result['output']
                        if detailed:
                            print(f"[Corrected answer: {extracted_answer}]")
        
        # ========== PHASE 4: Checker Verification ==========
        checker_verdict = "SKIPPED"
        checker_reason = ""
        
        if enable_checker_verify and extracted_answer:
            if detailed:
                print(f"\n[Phase 4: Checker verification...]")
            
            checker_prompt = format_checker_code_verify_prompt(question, extracted_answer)
            
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
                checker_response = generate_response(
                    checker_model, checker_tokenizer, checker_prompt, "standard", detailed
                )
            
            checker_code = extract_code_from_response(checker_response)
            
            if checker_code:
                checker_exec = execute_code(checker_code, timeout=code_timeout)
                checker_verdict, checker_reason = parse_checker_verify_response(checker_exec)
                
                if detailed:
                    print(f"[Checker verdict]: {checker_verdict}")
                    if checker_reason:
                        print(f"[Reason]: {checker_reason}")
            else:
                checker_verdict = "NO_CODE"
        
        # Store iteration data
        iterations.append({
            "iteration": iteration_num,
            "phase": "complete",
            "solver_response": solver_response,
            "code": code,
            "exec_success": True,
            "exec_output": current_output,
            "exec_error": None,
            "extracted_answer": extracted_answer,
            "self_verify_verdict": self_verify_verdict,
            "checker_verdict": checker_verdict,
            "checker_reason": checker_reason
        })
        
        # ========== Decision: Accept or Continue? ==========
        should_accept = False
        
        if self_verify_verdict == "CORRECT":
            should_accept = True
            final_status = f"SELF_VERIFIED_ITER_{iteration_num}"
        elif checker_verdict == "VERIFIED":
            should_accept = True
            final_status = f"CHECKER_VERIFIED_ITER_{iteration_num}"
        elif checker_verdict == "FAILED":
            if detailed:
                print(f"[Checker failed, will iterate...]")
            current_error = f"Checker verification failed: {checker_reason}"
        elif self_verify_verdict == "WRONG" and not new_code:
            if detailed:
                print(f"[Self-verify failed, will iterate...]")
            current_error = "Self-verification indicated the answer may be wrong."
        else:
            # Unclear or skipped - accept if we have an answer
            if extracted_answer:
                should_accept = True
                final_status = f"ACCEPTED_ITER_{iteration_num}"
        
        if should_accept:
            predicted_answer = extracted_answer
            if detailed:
                print(f"\n[Answer accepted: {predicted_answer}]")
            break
    
    # ========== FINAL RESULT ==========
    if predicted_answer is None:
        for i in range(len(iterations) - 1, -1, -1):
            if iterations[i].get('extracted_answer'):
                predicted_answer = iterations[i]['extracted_answer']
                final_status = f"LAST_VALID_ITER_{iterations[i]['iteration']}"
                break
        
        if predicted_answer is None:
            final_status = "NO_VALID_ANSWER"
    
    # Check correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    first_answer = all_answers[0] if all_answers else None
    first_correct = check_answer(first_answer, ground_truth) if first_answer else False
    
    # Determine case type
    if first_correct and final_correct:
        case_type = "FIRST_TRY_SUCCESS"
    elif not first_correct and final_correct:
        case_type = "FIXED_SUCCESS"
    elif first_correct and not final_correct:
        case_type = "DEGRADED"
    else:
        case_type = "FAILED"
    
    error_count = sum(1 for it in iterations if not it.get('exec_success', False))
    success_count = sum(1 for it in iterations if it.get('exec_success', False))
    
    result = {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "final_correct": final_correct,
        "final_status": final_status,
        "final_verdict": final_status,
        "total_iterations": len(iterations),
        "iterations": iterations,
        "all_codes": all_codes,
        "all_answers": all_answers,
        "first_answer": first_answer,
        "first_correct": first_correct,
        "case_type": case_type,
        "error_count": error_count,
        "success_count": success_count,
        "config": {
            "max_iterations": max_iterations,
            "enable_self_verify": enable_self_verify,
            "enable_checker_verify": enable_checker_verify,
            "code_timeout": code_timeout
        }
    }
    
    if detailed:
        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        print(f"Predicted: {predicted_answer}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correct: {final_correct}")
        print(f"Status: {final_status}")
        print(f"Case: {case_type}")
        print(f"Iterations: {len(iterations)}")
        print(f"{'='*80}")
    
    return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Solver-Coder - Quick Test")
    print("="*80)
    
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_ground_truth = "18"
    
    try:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        from models.inference import load_model
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning Solver-Coder workflow...")
        
        result = run_solver_coder_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            solver_model=model,
            solver_tokenizer=tokenizer,
            max_iterations=3,
            detailed=True,
            dataset_name="gsm8k",
            enable_self_verify=True,
            enable_checker_verify=True
        )
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Answer: {result['predicted_answer']}")
        print(f"Correct: {result['final_correct']}")
        print(f"Case: {result['case_type']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
