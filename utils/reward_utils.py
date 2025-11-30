"""
Reward computation utilities for RL training.
Implements rule-based verifier for mathematical reasoning.
"""

import re
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \boxed{} format.
    
    Args:
        text: Generated text containing \boxed{answer}
    
    Returns:
        Extracted answer string or None if not found
    """
    if not text or not isinstance(text, str):
        return None
    
    # Find the last occurrence of \boxed{...}
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed")
        if idx != -1 and idx + 6 < len(text) and text[idx + 6] == '{':
            idx = idx
        else:
            return None
    
    # Extract content with balanced braces
    start = text.find("{", idx)
    if start == -1:
        return None
    
    brace_count = 1
    end = start + 1
    
    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
        end += 1
    
    if brace_count == 0:
        answer = text[start + 1:end - 1].strip()
        return answer
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Handles:
    - Whitespace
    - LaTeX formatting
    - Number formatting
    - Case sensitivity
    
    Args:
        answer: Raw answer string
    
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    # Convert to lowercase
    answer = answer.lower().strip()
    
    # Remove common LaTeX commands
    answer = answer.replace("\\text{", "").replace("}", "")
    answer = answer.replace("\\mathrm{", "")
    answer = answer.replace("\\mathbf{", "")
    answer = answer.replace("\\textbf{", "")
    answer = answer.replace("\\$", "$")
    
    # Remove dollar signs
    answer = answer.replace("$", "")
    
    # Remove spaces around operators
    answer = re.sub(r'\s*([+\-*/=<>])\s*', r'\1', answer)
    
    # Remove commas in numbers
    answer = answer.replace(",", "")
    
    # Normalize whitespace
    answer = " ".join(answer.split())
    
    return answer


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """
    Compare predicted answer with ground truth.
    Uses the same logic as prompt_utils.check_answer for consistency.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
    
    Returns:
        True if answers match, False otherwise
    """
    # Import check_answer from prompt_utils to ensure consistency
    # between training rewards and evaluation metrics
    from utils.prompt_utils import check_answer
    return check_answer(predicted, ground_truth)


def compute_reward(
    generated_text: str,
    ground_truth: str,
    reward_correct: float = 1.0,
    reward_wrong: float = -1.0,
    reward_no_answer: float = -0.5
) -> Tuple[float, dict]:
    """
    Compute reward for generated text based on answer correctness.
    
    Args:
        generated_text: Generated response text
        ground_truth: Ground truth answer
        reward_correct: Reward for correct answer (default: +1.0)
        reward_wrong: Reward for wrong answer (default: -1.0)
        reward_no_answer: Reward for no valid answer (default: -0.5)
    
    Returns:
        Tuple of (reward, info_dict)
        - reward: Scalar reward value
        - info_dict: Dictionary with diagnostic information
    """
    # Extract predicted answer
    pred_answer = extract_boxed_answer(generated_text)
    
    # No valid answer found
    if pred_answer is None or pred_answer == "":
        return reward_no_answer, {
            "predicted": None,
            "ground_truth": ground_truth,
            "correct": False,
            "has_answer": False
        }
    
    # Extract ground truth answer
    gt_answer = extract_boxed_answer(ground_truth)
    if gt_answer is None:
        # If ground truth is not in \boxed{}, use it directly
        gt_answer = ground_truth
    
    # Compare answers
    is_correct = compare_answers(pred_answer, gt_answer)
    
    reward = reward_correct if is_correct else reward_wrong
    
    info = {
        "predicted": pred_answer,
        "ground_truth": gt_answer,
        "correct": is_correct,
        "has_answer": True
    }
    
    return reward, info


def batch_compute_rewards(
    generated_texts: list,
    ground_truths: list,
    **kwargs
) -> Tuple[list, list]:
    """
    Compute rewards for a batch of generated texts.
    
    Args:
        generated_texts: List of generated response texts
        ground_truths: List of ground truth answers
        **kwargs: Additional arguments passed to compute_reward
    
    Returns:
        Tuple of (rewards, infos)
        - rewards: List of reward values
        - infos: List of info dictionaries
    """
    assert len(generated_texts) == len(ground_truths), \
        f"Mismatch: {len(generated_texts)} texts vs {len(ground_truths)} truths"
    
    rewards = []
    infos = []
    
    for text, truth in zip(generated_texts, ground_truths):
        reward, info = compute_reward(text, truth, **kwargs)
        rewards.append(reward)
        infos.append(info)
    
    return rewards, infos


if __name__ == "__main__":
    # Test the reward computation
    test_cases = [
        {
            "generated": "The answer is \\boxed{42}",
            "truth": "42",
            "expected": 1.0
        },
        {
            "generated": "We can conclude that \\boxed{3.14}",
            "truth": "3.14",
            "expected": 1.0
        },
        {
            "generated": "Therefore \\boxed{1/2}",
            "truth": "0.5",
            "expected": 1.0
        },
        {
            "generated": "The solution is \\boxed{100}",
            "truth": "200",
            "expected": -1.0
        },
        {
            "generated": "I don't know the answer",
            "truth": "42",
            "expected": -0.5
        },
    ]
    
    print("Testing reward computation:")
    print("=" * 80)
    
    for i, case in enumerate(test_cases):
        reward, info = compute_reward(case["generated"], case["truth"])
        passed = abs(reward - case["expected"]) < 1e-6
        status = "✓" if passed else "✗"
        
        print(f"\nTest {i+1} {status}")
        print(f"  Generated: {case['generated']}")
        print(f"  Truth: {case['truth']}")
        print(f"  Predicted: {info['predicted']}")
        print(f"  Correct: {info['correct']}")
        print(f"  Reward: {reward} (expected: {case['expected']})")
    
    print("\n" + "=" * 80)
    print("All tests completed!")

