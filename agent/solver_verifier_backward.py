"""
Solver-BackwardVerifier Agent

Workflow (single-pass):
1) Solver generates reasoning (+ code) → extract boxed / numeric answer as candidate.
2) Backward checker performs a consistency check against the original question using the candidate answer, returning VERDICT (VERIFIED/FAILED).
3) If VERIFIED → accept the answer; otherwise mark as FAILED.

说明：
- Single round only; no iteration. Keep aligned with the default prompt (format_prompt_standard) without adding an Answer section.
- Verification uses agent/answer_backward_checker.py; defaults to greedy decoding without a chat template.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY
)
from agent.answer_backward_checker import run_answer_backward_checker


def run_solver_verifier_backward(
    question: str,
    ground_truth: str,
    solver_model,
    solver_tokenizer,
    detailed: bool = False,
    dataset_name: str = "",
    enable_solver_tools: bool = True,
) -> Dict:
    """
    Single-pass solver + backward verifier.

    Args:
        question: Problem text
        ground_truth: Reference answer (for scoring)
        solver_model/tokenizer: Model/tokenizer used for generation
        detailed: Whether to print verbose logs
        dataset_name: Dataset name (for prompts)
        enable_solver_tools: Whether solver is allowed to output code (hint only, not enforced)

    Returns:
        Dict: Contains predicted_answer, final_correct, verdict, response, etc.
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response

    # Solver prompt
    tool_instruction = ""
    if enable_solver_tools:
        tool_instruction = "\n\nYou may use Python code to help with calculations. Show your reasoning step by step."
    solver_prompt = format_prompt_standard(question, dataset_name) + tool_instruction

    if detailed:
        print(f"\n{'='*80}")
        print("SOLVER + BACKWARD VERIFIER (single pass)")
        print(f"Question: {question[:120]}...")
        print(f"{'='*80}")

    # Generate solver response
    solver_response = generate_response(
        solver_model,
        solver_tokenizer,
        solver_prompt,
        "standard",
        detailed,
        greedy=True,
    )

    # Extract candidate answer
    solver_answer = extract_answer(solver_response)

    if detailed:
        print(f"[Solver answer]: {solver_answer}")

    # Backward verify
    if solver_answer:
        verify_result = run_answer_backward_checker(
            question=question,
            proposed_answer=solver_answer,
            ground_truth=ground_truth,
            model=solver_model,  # reuse solver as verifier
            tokenizer=solver_tokenizer,
            detailed=detailed,
            dataset_name=dataset_name,
            greedy=True,
            apply_chat_template=False,
        )
        verdict = verify_result.get("verdict", "UNCLEAR")
    else:
        verify_result = {"verdict": "NO_ANSWER", "response": ""}
        verdict = "NO_ANSWER"

    final_answer = solver_answer
    final_correct = check_answer(final_answer, ground_truth) if final_answer else False

    if detailed:
        print(f"[Backward verdict]: {verdict}")
        print(f"[Final correct]: {final_correct}")

    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": final_answer,
        "final_correct": final_correct,
        "verdict": verdict,
        "solver_response": solver_response,
        "verify_response": verify_result.get("response", ""),
        "first_answer": solver_answer,
    }


if __name__ == "__main__":
    # Quick self-test
    try:
        base_path = Path(__file__).parent.parent
        model_name = "pretrained_models/Qwen2.5-Math-1.5B"
        from models.inference import load_model

        test_q = "16 eggs per day, eat 3, bake 4, sell rest at $2 each. How many dollars from selling?"
        test_gt = "18"

        print("\n[Loading model for quick test...]")
        model, tokenizer = load_model(model_name, base_path)

        result = run_solver_verifier_backward(
            question=test_q,
            ground_truth=test_gt,
            solver_model=model,
            solver_tokenizer=tokenizer,
            detailed=True,
            dataset_name="gsm8k",
            enable_solver_tools=True,
        )

        print("\nResult summary:")
        for k, v in result.items():
            if isinstance(v, str) and len(v) > 200:
                print(f"{k}: {v[:200]}...")
            else:
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Self-test failed: {e}")



