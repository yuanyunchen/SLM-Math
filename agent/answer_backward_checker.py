"""
Answer Backward Checker

Given a problem and a candidate numeric answer, ask the model to verify the
answer by reconstructing/validating the quantities (forward/backward check).
Outputs a simple verdict (VERIFIED/FAILED) plus the model's reasoning.
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
from models.generation_config import MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY
from utils.python_code_execution import extract_python_code_blocks, execute_python_code


def run_answer_backward_checker(
    question: str,
    proposed_answer: str,
    ground_truth: str,
    model,
    tokenizer,
    detailed: bool = False,
    dataset_name: str = "",
    greedy: bool = True,
    apply_chat_template: bool = False,
) -> Dict:
    """
    Verify a proposed numeric answer by asking the model to check consistency.

    Args:
        question: Original problem text.
        proposed_answer: Candidate numeric/string answer to verify.
        ground_truth: Ground truth answer (for scoring).
        model: Loaded model.
        tokenizer: Corresponding tokenizer.
        detailed: Whether to print verbose logs.
        dataset_name: Dataset name (unused but kept for API symmetry).
        greedy: Greedy decoding flag.
        apply_chat_template: Whether to wrap prompt with chat template.

    Returns:
        Dict with verdict, correctness vs ground truth, and raw response.
    """
    from utils.prompt_utils import extract_answer, check_answer
    from models.inference import generate_response

    # Build a concise checker prompt
    base_prompt = f"""You are a math answer checker.
Given the problem and a proposed answer, verify correctness by doing a forward
recompute and a backward consistency check (use the proposed answer to see if
the original quantities/operations still hold).

Problem:
{question}

Proposed answer: {proposed_answer}

Instructions:
- Briefly explain the key steps.
- Write ONE Python code block that:
    * extracts the needed numbers,
    * recomputes the expected answer (forward),
    * sets `given_answer = {proposed_answer}`,
    * checks that `expected` and `given_answer` match,
    * optionally does a simple backward sanity check (e.g., plug the answer
      back into the operation to recover the known inputs),
    * prints only:
        VERIFIED
      or
        FAILED: expected <expected_value>, got <given_answer>
- Do not print anything else.

Format:
Reasoning: <short steps>
```python
# your verification code
```
"""

    prompt = base_prompt
    if apply_chat_template and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if detailed:
        print(f"\n{'='*80}")
        print("ANSWER BACKWARD CHECKER")
        print(f"Question: {question[:100]}...")
        print(f"Proposed answer: {proposed_answer}")
        print(f"Greedy: {greedy}, Chat template: {apply_chat_template}")
        print(f"{'='*80}\n")

    # Generate response
    response = generate_response(
        model,
        tokenizer,
        prompt,
        "standard",
        detailed,
        greedy=greedy,
    )

    # Execute any python code blocks for verification
    checker_exec_results = []
    checker_code_used = False
    code_blocks = extract_python_code_blocks(response)
    for i, code in enumerate(code_blocks, 1):
        result = execute_python_code(code, timeout=10)
        checker_exec_results.append(result)
        if detailed:
            status = "ok" if result["success"] else "error"
            preview = result["output"][:80] if result["output"] else result["error"][:80]
            print(f"[Checker code #{i} {status}]: {preview}")
    if checker_exec_results:
        checker_code_used = True

    # Parse verdict (prefer code execution output)
    verdict = "UNCLEAR"
    code_reason = ""

    for res in reversed(checker_exec_results):
        if not res["success"]:
            continue
        out = res["output"].strip()
        up = out.upper()
        if "VERIFIED" in up:
            verdict = "VERIFIED"
            code_reason = out
            break
        if "FAILED" in up:
            verdict = "FAILED"
            reason_match = re.search(r'FAILED[:\s]*(.+)', out, re.IGNORECASE)
            code_reason = reason_match.group(1).strip() if reason_match else out
            break

    # Fallback to text verdict if code output unclear
    if verdict == "UNCLEAR":
        m = re.search(r"VERDICT[:\s]+(VERIFIED|FAILED)", response, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()

    # If still unclear and code failed, mark error
    if verdict == "UNCLEAR" and any(not r["success"] for r in checker_exec_results):
        verdict = "ERROR"

    # Extract model-extracted answer (may differ from proposed)
    model_answer = extract_answer(response)

    # Use proposed answer as final_checked_answer for consistency check
    final_checked_answer = proposed_answer
    final_correct = check_answer(final_checked_answer, ground_truth)

    if detailed:
        print(f"[Model response preview]: {response[:300]}...")
        print(f"[Verdict]: {verdict}")
        print(f"[Ground truth match]: {final_correct}")

    return {
        "question": question,
        "proposed_answer": proposed_answer,
        "ground_truth": ground_truth,
        "verdict": verdict,
        "final_correct": final_correct,
        "model_answer_extracted": model_answer,
        "response": response,
        "checker_code_used": checker_code_used,
        "checker_exec_results": checker_exec_results,
        "checker_reason": code_reason,
    }


if __name__ == "__main__":
    # Quick self-test (uses base model without checkpoint)
    try:
        test_q = "16 eggs per day, eat 3, bake 4, sell rest at $2 each. How many dollars from selling?"
        test_a = "18"
        base_path = Path(__file__).parent.parent
        model_name = "pretrained_models/Qwen2.5-Math-1.5B"
        from models.inference import load_model

        print("\n[Loading model for quick test...]")
        model, tokenizer = load_model(model_name, base_path)

        result = run_answer_backward_checker(
            question=test_q,
            proposed_answer=test_a,
            ground_truth=test_a,
            model=model,
            tokenizer=tokenizer,
            detailed=True,
            greedy=True,
            apply_chat_template=False,
        )

        print("\nResult:")
        print(result)
    except Exception as e:
        print(f"Self-test failed: {e}")

