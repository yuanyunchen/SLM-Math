#!/usr/bin/env python3
"""
Reasoning + Code Data Generation Script (V2 - Improved Prompt)
--------------------------------------------------------------

Key changes from v1:
- Removed forced <think><code><final> format
- Encourages detailed step-by-step reasoning
- Emphasizes code quality and correctness
- Uses natural markdown format like the original Qwen model

Two-Round Architecture with Multi-Model Support:
- Round 1: Teacher models with retries
- Round 2: Expert models retry failed samples

Usage:
    python -m dataset.build_reasoning_code_data_v2 \\
        --teacher "x-ai/grok-4-1-fast-reasoning,deepseek/deepseek-reasoner-v3.1" \\
        --expert "alibaba/qwen3-235b-a22b-thinking-2507,minimax/m2" \\
        --dataset "gsm8k-train,math-train" \\
        --round1-attempts 3 \\
        --round2-attempts 5 \\
        --count 100 \\
        --workers 8 \\
        --api-key YOUR_KEY \\
        --output-dir data/reasoning_code/generated
"""

import sys
import json
import time
import random
import argparse
import io
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
from contextlib import redirect_stdout
import math
import itertools
import fractions
import decimal
from fractions import Fraction
from decimal import Decimal
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dataset.dataloader import load_dataset_for_eval
from utils.prompt_utils import extract_answer, check_answer

# Import openai after path setup
import openai


# ============================================================================
# API Helpers
# ============================================================================

def initialize_api(api_key: str, api_base: str):
    """Initialize OpenAI API client."""
    openai.api_key = api_key
    openai.base_url = api_base


# ============================================================================
# Prompt & Parsing Helpers (V2 - Improved)
# ============================================================================

def get_system_prompt_v2(thinking_effort: str) -> str:
    """
    Construct system prompt for reasoning + code generation.
    
    V2 Changes:
    - Uses natural step-by-step format instead of forced <think><code><final> tags
    - Encourages detailed reasoning before coding
    - Emphasizes code quality and valid Python syntax
    - Follows the original Qwen-Math style that achieved 83.6% accuracy
    """
    effort_configs = {
        "low": {
            "instruction": (
                "Solve the math problem. Show your reasoning clearly, then write "
                "Python code to verify your answer."
            ),
            "detail_level": "concise",
            "thinking_tokens": 300
        },
        "medium": {
            "instruction": (
                "Solve the math problem step by step. First explain your approach, "
                "then write Python code that computes and verifies the answer. "
                "Make sure your reasoning is clear and your code is correct."
            ),
            "detail_level": "clear",
            "thinking_tokens": 700
        },
        "high": {
            "instruction": (
                "Solve the math problem with detailed step-by-step reasoning. "
                "Break down the problem, explain each step of your solution, "
                "then write well-structured Python code that computes and prints "
                "the final answer. Double-check your work."
            ),
            "detail_level": "detailed",
            "thinking_tokens": 1400
        },
        "very_high": {
            "instruction": (
                "Provide comprehensive step-by-step reasoning for this math problem. "
                "Analyze the problem carefully, consider different approaches, "
                "explain your solution method in detail, then write robust Python "
                "code that computes the answer. Verify your result."
            ),
            "detail_level": "comprehensive",
            "thinking_tokens": 2500
        }
    }

    config = effort_configs.get(thinking_effort, effort_configs["medium"])

    return f"""You are an expert mathematics problem solver and Python programmer.
{config['instruction']}

Instructions:
1. First, analyze the problem and explain your reasoning step by step.
2. Then, write Python code to compute the answer.
3. Finally, state your answer in \\boxed{{}}.

Code requirements:
- Use only Python 3.9 standard library (math, fractions, decimal, itertools are allowed)
- Use valid Python variable names (no spaces, no special characters)
- The code must be executable as-is and deterministically compute the answer
- Store the final result in a variable called `final_answer` and print it
- Do not read/write files or make network calls

Example format:
To solve this problem, I need to...

[Step-by-step reasoning explaining your approach]

Let me write code to compute this:
```python
# Clear, well-commented code
# Use proper variable names (e.g., total_cost, not "total cost")
...
final_answer = ...
print(final_answer)
```

The code outputs X, so the answer is \\boxed{{X}}"""


def extract_python_code(text: str) -> Optional[str]:
    """Extract the first Python code block from a response."""
    patterns = [
        r"```python\s*(.*?)```",
        r"```py\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            if code:
                return code
    return None


# ============================================================================
# Safe Code Execution
# ============================================================================

def execute_python_code(code: str, timeout: int) -> Dict:
    """Execute code with timeout; return status, answer, stdout.
    
    Uses direct execution (no subprocess) to avoid nested process issues
    when running inside Pool workers.
    """
    import threading
    
    result_holder = [None]
    
    def run_code():
        """Run code in a restricted environment."""
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            allowed = {"math", "fractions", "decimal", "itertools"}
            if name in allowed:
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of {name} is not allowed")

        allowed_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "range": range, "enumerate": enumerate, "zip": zip, "round": round,
            "int": int, "float": float, "pow": pow, "sorted": sorted,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "str": str, "bool": bool, "map": map, "filter": filter,
            "any": any, "all": all, "reversed": reversed,
            "print": print, "__import__": safe_import,
        }

        safe_globals = {
            "__builtins__": allowed_builtins,
            "math": math,
            "Fraction": Fraction,
            "fractions": fractions,
            "Decimal": Decimal,
            "decimal": decimal,
            "itertools": itertools,
        }
        local_env: Dict = {}
        stdout_buffer = io.StringIO()

        try:
            with redirect_stdout(stdout_buffer):
                exec(code, safe_globals, local_env)

            answer = None
            for key in ("final_answer", "answer", "result", "res"):
                if key in local_env:
                    answer = local_env[key]
                    break

            stdout_text = stdout_buffer.getvalue().strip()
            if answer is None and stdout_text:
                answer = stdout_text.splitlines()[-1].strip()

            if isinstance(answer, Fraction):
                answer_str = f"{answer.numerator}/{answer.denominator}"
            elif isinstance(answer, Decimal):
                answer_str = format(answer.normalize(), "f")
            elif answer is not None:
                answer_str = str(answer)
            else:
                answer_str = None

            result_holder[0] = {
                "status": "ok",
                "answer": answer_str,
                "stdout": stdout_text
            }
        except Exception as e:
            result_holder[0] = {
                "status": "error",
                "answer": None,
                "stdout": stdout_buffer.getvalue().strip(),
                "error": str(e)
            }

    thread = threading.Thread(target=run_code)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Thread is still running (timeout) - we can't forcibly kill it,
        # but we return timeout status. The thread will eventually finish.
        return {"status": "timeout", "answer": None, "stdout": ""}

    if result_holder[0] is None:
        return {"status": "no_result", "answer": None, "stdout": ""}

    return result_holder[0]


# ============================================================================
# Model Call
# ============================================================================

def call_api_single_attempt(
    question: str,
    teacher_model: str,
    thinking_effort: str,
    max_tokens: int,
    temperature: float,
    dataset_name: str,
    code_timeout: int,
) -> Optional[Dict]:
    """Call API once and return reasoning + code result."""
    try:
        system_prompt = get_system_prompt_v2(thinking_effort)

        client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)
        response = client.chat.completions.create(
            model=teacher_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        raw_output = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        tokens_used = response.usage.total_tokens

        # V2: Extract thinking process more flexibly
        # Look for content before the code block as "thinking"
        thinking_process = ""
        solution = raw_output
        
        # Try to find thinking in <think> tags if present
        if "<think>" in raw_output and "</think>" in raw_output:
            start_idx = raw_output.find("<think>") + len("<think>")
            end_idx = raw_output.find("</think>")
            thinking_process = raw_output[start_idx:end_idx].strip()
            solution = raw_output[end_idx + len("</think>"):].strip()
        else:
            # Otherwise, everything before ```python is thinking
            code_start = raw_output.find("```python")
            if code_start == -1:
                code_start = raw_output.find("```")
            if code_start > 0:
                thinking_process = raw_output[:code_start].strip()

        code_block = extract_python_code(raw_output)
        execution = {"status": "no_code", "answer": None, "stdout": ""}
        if code_block:
            execution = execute_python_code(code_block, code_timeout)

        # Extract both answers separately
        code_answer = execution.get("answer")
        boxed_answer = extract_answer(raw_output)
        
        # Primary answer: prefer code result, fallback to boxed
        predicted_answer = code_answer or boxed_answer
        
        # Check if both answers match (when both exist)
        answers_match = None
        if code_answer and boxed_answer:
            answers_match = check_answer(code_answer, boxed_answer)

        return {
            "raw_output": raw_output,
            "thinking_process": thinking_process,
            "solution": solution,
            "code": code_block or "",
            "predicted_answer": predicted_answer,
            "code_answer": code_answer,
            "boxed_answer": boxed_answer,
            "answers_match": answers_match,
            "tokens_used": tokens_used,
            "finish_reason": finish_reason,
            "execution": execution,
            "answer_source": "code" if code_answer else "text",
            "status": "success"
        }

    except Exception as e:
        return {
            "raw_output": "",
            "thinking_process": "",
            "solution": "",
            "code": "",
            "predicted_answer": None,
            "code_answer": None,
            "boxed_answer": None,
            "answers_match": None,
            "tokens_used": 0,
            "finish_reason": "error",
            "execution": {"status": "error", "answer": None, "stdout": "", "error": str(e)},
            "answer_source": "error",
            "status": f"error: {str(e)}"
        }


# ============================================================================
# Sampling Helpers
# ============================================================================

def select_thinking_effort(efforts: List[str], ratios: List[float]) -> str:
    """Randomly select thinking effort based on ratios."""
    return random.choices(efforts, weights=ratios, k=1)[0]


def extract_question_and_ground_truth(example: dict, dataset_name: str) -> Tuple[str, str]:
    """Extract question and ground truth for supported datasets."""
    if dataset_name == "gsm8k":
        question = example["question"]
        ground_truth = example["answer"].split("####")[-1].strip()
    elif dataset_name == "math":
        question = example["problem"]
        if "answer" in example:
            ground_truth = example["answer"].strip()
        else:
            solution = example.get("solution", "")
            match = re.search(r"\\boxed\{([^}]+)\}", solution)
            if match:
                ground_truth = match.group(1)
            else:
                numbers = re.findall(r"[+-]?\d+\.?\d*", solution)
                ground_truth = numbers[-1] if numbers else solution
    else:
        question = example.get("question", example.get("problem", ""))
        ground_truth = example.get("answer", example.get("solution", ""))

    return question, ground_truth


# ============================================================================
# Core Processing
# ============================================================================

def process_sample_with_model(
    sample_data: Tuple
) -> Dict:
    """Process a single sample with one model (with retry)."""
    (
        index,
        question,
        ground_truth,
        model_name,
        round_num,
        max_attempts,
        thinking_efforts,
        effort_ratios,
        max_tokens,
        temperatures,
        dataset_name,
        code_timeout,
    ) = sample_data

    thinking_effort = select_thinking_effort(thinking_efforts, effort_ratios)

    attempts = []
    final_correct = False
    final_result = None

    for attempt in range(max_attempts):
        temp = temperatures[attempt % len(temperatures)]

        result = call_api_single_attempt(
            question,
            model_name,
            thinking_effort,
            max_tokens,
            temp,
            dataset_name,
            code_timeout,
        )

        if result is None:
            continue

        predicted = result["predicted_answer"]
        is_correct = False
        if predicted is not None:
            is_correct = check_answer(predicted, ground_truth)

        attempts.append({
            "attempt": attempt + 1,
            "temperature": temp,
            "predicted_answer": predicted,
            "correct": is_correct,
            "tokens": result["tokens_used"],
            "status": result["status"],
            "execution_status": result["execution"].get("status") if result.get("execution") else "no_code",
        })

        if is_correct:
            final_correct = True
            final_result = result
            break

    if final_result is None:
        final_result = {
            "raw_output": "",
            "thinking_process": "",
            "solution": "",
            "code": "",
            "predicted_answer": None,
            "code_answer": None,
            "boxed_answer": None,
            "answers_match": None,
            "tokens_used": 0,
            "finish_reason": "no_attempts",
            "execution": {"status": "failed", "answer": None, "stdout": ""},
            "answer_source": "none",
            "status": "failed"
        }

    return {
        "index": index,
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": final_result["predicted_answer"],
        "code_answer": final_result.get("code_answer"),
        "boxed_answer": final_result.get("boxed_answer"),
        "answers_match": final_result.get("answers_match"),
        "thinking_process": final_result["thinking_process"],
        "solution": final_result["solution"],
        "raw_output": final_result["raw_output"],
        "code": final_result.get("code", ""),
        "execution": final_result.get("execution", {}),
        "answer_source": final_result.get("answer_source", "unknown"),
        "correct": final_correct,
        "teacher_model": model_name,
        "thinking_effort": thinking_effort,
        "round": round_num,
        "attempt_count": len(attempts),
        "attempts_history": attempts,
        "tokens_used": sum(a["tokens"] for a in attempts),
        "status": "success" if final_correct else "failed"
    }


def run_round_multi_model(
    samples: List[Dict],
    models: List[str],
    round_num: int,
    max_attempts: int,
    thinking_efforts: List[str],
    effort_ratios: List[float],
    max_tokens: int,
    temperatures: List[float],
    workers: int,
    dataset_name: str,
    code_timeout: int,
    dataset_full: str,
    output_jsonl: Optional[Path] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run one round with multiple models in parallel.
    Writes results incrementally to output_jsonl if provided.
    Returns: (successful_results, failed_samples)
    """
    print(f"\n{'='*80}")
    print(f"ROUND {round_num}: {len(samples)} samples x {len(models)} models")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models)}")
    print(f"Max Attempts: {max_attempts}")

    effort_display = []
    for effort, ratio in zip(thinking_efforts, effort_ratios):
        effort_display.append(f"{effort}({ratio*100:.0f}%)")
    print(f"Thinking Effort: {', '.join(effort_display)}")
    print(f"{'='*80}\n")

    if not samples:
        return [], []

    tasks = []
    for sample in samples:
        for model in models:
            tasks.append((
                sample["index"],
                sample["question"],
                sample["ground_truth"],
                model,
                round_num,
                max_attempts,
                thinking_efforts,
                effort_ratios,
                max_tokens,
                temperatures,
                dataset_name,
                code_timeout,
            ))

    print(f"Processing {len(tasks)} tasks ({len(samples)} samples x {len(models)} models)...")
    start_time = time.time()
    results = []
    written_count = 0
    correct_count = 0

    # Open file for incremental writing
    jsonl_file = None
    if output_jsonl:
        jsonl_file = open(output_jsonl, "a", encoding="utf-8")

    try:
        pbar = tqdm(total=len(tasks), desc=f"Round {round_num}", unit="task")
        if workers > 1:
            with Pool(processes=workers) as pool:
                for result in pool.imap_unordered(process_sample_with_model, tasks):
                    result["dataset"] = dataset_full
                    results.append(result)
                    # Write correct results immediately
                    if result["correct"]:
                        correct_count += 1
                        if jsonl_file:
                            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                            jsonl_file.flush()
                            written_count += 1
                    pbar.update(1)
                    pbar.set_postfix(correct=correct_count, written=written_count)
        else:
            for task in tasks:
                result = process_sample_with_model(task)
                result["dataset"] = dataset_full
                results.append(result)
                if result["correct"]:
                    correct_count += 1
                    if jsonl_file:
                        jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        jsonl_file.flush()
                        written_count += 1
                pbar.update(1)
                pbar.set_postfix(correct=correct_count, written=written_count)
        pbar.close()
    finally:
        if jsonl_file:
            jsonl_file.close()

    elapsed = time.time() - start_time

    sample_results: Dict[int, List[Dict]] = {}
    for r in results:
        idx = r["index"]
        if idx not in sample_results:
            sample_results[idx] = []
        sample_results[idx].append(r)

    successful = []
    failed_indices = set()

    for idx, sample_results_list in sample_results.items():
        correct_results = [r for r in sample_results_list if r["correct"]]

        if correct_results:
            successful.extend(correct_results)
        else:
            failed_indices.add(idx)

    failed_samples = [s for s in samples if s["index"] in failed_indices]

    total_correct = len(set(r["index"] for r in successful))
    total_failed = len(failed_samples)

    print(f"\n[OK] Round {round_num} Complete:")
    print(f"  Samples Processed: {len(samples)}")
    print(f"  Samples Solved: {total_correct} ({total_correct/len(samples)*100:.2f}%)")
    print(f"  Samples Failed: {total_failed} ({total_failed/len(samples)*100:.2f}%)")
    print(f"  Total Solutions: {len(successful)} (including multiple correct per sample)")
    print(f"  Written to disk: {written_count}")
    print(f"  Time: {elapsed:.2f}s")

    for model in models:
        model_results = [r for r in results if r["teacher_model"] == model]
        model_correct = len([r for r in model_results if r["correct"]])
        print(f"    {model}: {model_correct}/{len(model_results)} correct")

    return successful, failed_samples


# ============================================================================
# Dataset Processing
# ============================================================================

def process_datasets(
    dataset_names: List[str],
    teacher_models: List[str],
    expert_models: List[str],
    round1_attempts: int,
    round2_attempts: int,
    round1_efforts: List[str],
    round1_ratios: List[float],
    round2_efforts: List[str],
    round2_ratios: List[float],
    max_tokens: int,
    round1_temps: List[float],
    round2_temps: List[float],
    workers: int,
    count: int,
    start_idx: int,
    output_dir: Path,
    code_timeout: int,
) -> Dict:
    """Process all datasets through 2-round pipeline with incremental saving."""

    # Create output directory and JSONL file for incremental writes
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_str = "_".join([d.replace("-", "_") for d in dataset_names])
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base_name = f"reasoning_code_v2_{dataset_str}_{timestamp}"
    output_jsonl = output_dir / f"{base_name}.jsonl"

    print(f"\n[Incremental Output] Writing results to: {output_jsonl}")

    all_results = []
    dataset_stats = {}

    for dataset_full in dataset_names:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset_full}")
        print(f"{'#'*80}\n")

        dataset_name = dataset_full
        split_name = "train"
        if "-" in dataset_full:
            parts = dataset_full.split("-", 1)
            dataset_name, split_name = parts[0], parts[1]

        base_path = Path(__file__).parent.parent
        dataset = load_dataset_for_eval(dataset_name, str(base_path), split_name=split_name)

        total = len(dataset)
        remaining = total - start_idx
        num_samples = remaining if count == 0 else min(count, remaining)

        print(f"Dataset: {dataset_name} (split: {split_name})")
        print(f"Total samples: {total}")
        print(f"Processing: {num_samples} samples (index {start_idx} to {start_idx + num_samples - 1})\n")

        samples = []
        for i in range(start_idx, start_idx + num_samples):
            example = dataset[i]
            question, answer = extract_question_and_ground_truth(example, dataset_name)

            samples.append({
                "index": i,
                "question": question,
                "ground_truth": answer,
                "dataset": dataset_full
            })

        # Round 1
        round1_results, round1_failed = run_round_multi_model(
            samples,
            teacher_models,
            1,
            round1_attempts,
            round1_efforts,
            round1_ratios,
            max_tokens,
            round1_temps,
            workers,
            dataset_name,
            code_timeout,
            dataset_full,
            output_jsonl,
        )

        # Round 2
        round2_results = []
        if round1_failed and expert_models:
            round2_results, _ = run_round_multi_model(
                round1_failed,
                expert_models,
                2,
                round2_attempts,
                round2_efforts,
                round2_ratios,
                max_tokens,
                round2_temps,
                workers,
                dataset_name,
                code_timeout,
                dataset_full,
                output_jsonl,
            )

        all_round_results = round1_results + round2_results
        all_results.extend(all_round_results)

        unique_solved = set(r["index"] for r in all_round_results)
        r1_solved = set(r["index"] for r in round1_results)
        r2_solved = set(r["index"] for r in round2_results)

        dataset_stats[dataset_full] = {
            "total_samples": num_samples,
            "solved_total": len(unique_solved),
            "solved_round1": len(r1_solved),
            "solved_round2": len(r2_solved),
            "unsolved": num_samples - len(unique_solved),
            "success_rate": len(unique_solved) / num_samples * 100 if num_samples else 0.0,
            "solutions_count": len(all_round_results)
        }

        print(f"\n{'='*80}")
        print(f"Dataset {dataset_full} Summary:")
        print(f"  Total: {num_samples}")
        print(f"  Solved: {len(unique_solved)} ({dataset_stats[dataset_full]['success_rate']:.2f}%)")
        print(f"    Round 1: {len(r1_solved)}")
        print(f"    Round 2: {len(r2_solved)}")
        print(f"  Unsolved: {dataset_stats[dataset_full]['unsolved']}")
        print(f"  Total Solutions: {len(all_round_results)}")
        print(f"{'='*80}")

    return {
        "results": all_results,
        "dataset_stats": dataset_stats,
        "output_jsonl": output_jsonl,
        "base_name": base_name,
    }


# ============================================================================
# Saving
# ============================================================================

def save_final_results(
    results: List[Dict],
    dataset_stats: Dict,
    output_dir: Path,
    teacher_models: List[str],
    expert_models: List[str],
    config: Dict,
    base_name: str,
    output_jsonl: Optional[Path] = None,
):
    """Save final results in clean format (JSON, CSV, stats).
    
    Results are already incrementally written to JSONL. This function
    generates sorted JSON, CSV, and statistics summary.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load from JSONL if available (more reliable than in-memory)
    if output_jsonl and output_jsonl.exists():
        results = []
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        print(f"\n[Info] Loaded {len(results)} results from {output_jsonl}")

    results_sorted = sorted(results, key=lambda x: (x["dataset"], x["index"]))

    json_file = output_dir / f"{base_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved JSON: {json_file}")
    print(f"  Total solutions: {len(results_sorted)}")

    csv_file = output_dir / f"{base_name}.csv"

    df_data = []
    for r in results_sorted:
        df_data.append({
            "index": r["index"],
            "dataset": r["dataset"],
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "predicted_answer": r["predicted_answer"],
            "code_answer": r.get("code_answer"),
            "boxed_answer": r.get("boxed_answer"),
            "answers_match": r.get("answers_match"),
            "correct": r["correct"],
            "thinking_process": r["thinking_process"],
            "solution": r["solution"],
            "code": r.get("code", ""),
            "execution_status": r.get("execution", {}).get("status"),
            "execution_stdout": r.get("execution", {}).get("stdout"),
            "answer_source": r.get("answer_source", "unknown"),
            "teacher_model": r["teacher_model"],
            "round": r["round"],
            "attempt_count": r["attempt_count"],
            "tokens_used": r["tokens_used"],
            "status": r["status"]
        })

    import pandas as pd  # Local import to keep top import section small

    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False, encoding="utf-8")

    print(f"[OK] Saved CSV: {csv_file}")

    stats_file = output_dir / f"{base_name}_statistics.txt"

    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("Reasoning + Code Data Generation Statistics (V2)\n")
        f.write("="*80 + "\n\n")

        f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Configuration:\n")
        f.write("-"*80 + "\n")
        f.write(f"Teacher Models: {', '.join(teacher_models)}\n")
        f.write(f"Expert Models: {', '.join(expert_models)}\n")
        f.write(f"Round 1 Attempts: {config['round1_attempts']}\n")
        f.write(f"Round 2 Attempts: {config['round2_attempts']}\n")
        f.write(f"Workers: {config['workers']}\n")
        f.write(f"Code Timeout: {config['code_timeout']}s\n\n")

        f.write("Overall Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Solutions Generated: {len(results_sorted)}\n")
        f.write(f"Unique Samples Solved: {len(set((r['dataset'], r['index']) for r in results_sorted))}\n\n")

        f.write("Per-Dataset Statistics:\n")
        f.write("-"*80 + "\n")
        for dataset, stats in dataset_stats.items():
            f.write(f"\n{dataset}:\n")
            f.write(f"  Total Samples: {stats['total_samples']}\n")
            f.write(f"  Solved: {stats['solved_total']} ({stats['success_rate']:.2f}%)\n")
            f.write(f"    Round 1: {stats['solved_round1']}\n")
            f.write(f"    Round 2: {stats['solved_round2']}\n")
            f.write(f"  Unsolved: {stats['unsolved']}\n")
            f.write(f"  Total Solutions: {stats['solutions_count']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Per-Model Statistics:\n")
        f.write("-"*80 + "\n")

        for model in teacher_models + expert_models:
            model_results = [r for r in results_sorted if r["teacher_model"] == model]
            if model_results:
                correct_count = len([r for r in model_results if r["correct"]])
                f.write(f"\n{model}:\n")
                f.write(f"  Solutions: {len(model_results)}\n")
                f.write(f"  Correct: {correct_count}\n")
                f.write(f"  Unique Samples: {len(set(r['index'] for r in model_results))}\n")

                r1 = [r for r in model_results if r["round"] == 1]
                r2 = [r for r in model_results if r["round"] == 2]
                if r1:
                    f.write(f"  Round 1: {len([r for r in r1 if r['correct']])}/{len(r1)} correct\n")
                if r2:
                    f.write(f"  Round 2: {len([r for r in r2 if r['correct']])}/{len(r2)} correct\n")

        f.write("\n" + "="*80 + "\n")

    print(f"[OK] Saved Statistics: {stats_file}\n")


# ============================================================================
# CLI
# ============================================================================

def parse_efforts_and_ratios(efforts_str: str, ratios_str: str) -> Tuple[List[str], List[float]]:
    efforts = [e.strip() for e in efforts_str.split(",") if e.strip()]
    if ratios_str:
        ratios = [float(r) for r in ratios_str.split(",")]
    else:
        ratios = [1.0 / len(efforts)] * len(efforts)

    if len(efforts) != len(ratios):
        print(f"Error: efforts ({len(efforts)}) and ratios ({len(ratios)}) must match")
        sys.exit(1)
    if abs(sum(ratios) - 1.0) > 0.01:
        total = sum(ratios)
        ratios = [r / total for r in ratios]
    return efforts, ratios


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning + Code Data Generation V2 (Improved Prompt)"
    )

    parser.add_argument("--teacher", type=str, required=True,
                        help="Comma-separated list of teacher models")
    parser.add_argument("--expert", type=str, default="",
                        help="Comma-separated list of expert models (optional)")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Comma-separated list of datasets (e.g., gsm8k-train,math-train)")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of samples per dataset (0 = all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting index")

    parser.add_argument("--round1-attempts", type=int, default=3,
                        help="Max attempts for Round 1")
    parser.add_argument("--round1-efforts", type=str, default="medium",
                        help="Comma-separated thinking efforts for Round 1")
    parser.add_argument("--round1-effort-ratios", type=str, default="",
                        help="Comma-separated ratios for Round 1 efforts")
    parser.add_argument("--round1-temps", type=str, default="0.7,0.5,0.9",
                        help="Comma-separated temperatures for Round 1")

    parser.add_argument("--round2-attempts", type=int, default=5,
                        help="Max attempts for Round 2")
    parser.add_argument("--round2-efforts", type=str, default="high",
                        help="Comma-separated thinking efforts for Round 2")
    parser.add_argument("--round2-effort-ratios", type=str, default="",
                        help="Comma-separated ratios for Round 2 efforts")
    parser.add_argument("--round2-temps", type=str, default="0.7,0.5,0.9,0.3,1.0",
                        help="Comma-separated temperatures for Round 2")

    # V2: Default to 4096 tokens for more detailed reasoning
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens for generation (default: 4096 for detailed reasoning)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--api-base", type=str, default="https://api.aimlapi.com/v1")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--code-timeout", type=int, default=8,
                        help="Seconds allowed for executing generated code")

    args = parser.parse_args()

    teacher_models = [m.strip() for m in args.teacher.split(",") if m.strip()]
    expert_models = [m.strip() for m in args.expert.split(",") if m.strip()] if args.expert else []
    dataset_names = [d.strip() for d in args.dataset.split(",") if d.strip()]
    round1_temps = [float(t) for t in args.round1_temps.split(",")]
    round2_temps = [float(t) for t in args.round2_temps.split(",")]

    round1_efforts, round1_ratios = parse_efforts_and_ratios(args.round1_efforts, args.round1_effort_ratios)
    round2_efforts, round2_ratios = parse_efforts_and_ratios(args.round2_efforts, args.round2_effort_ratios)

    initialize_api(args.api_key, args.api_base)

    print("\n" + "="*80)
    print("REASONING + CODE DATA GENERATION (V2 - Improved Prompt)")
    print("="*80)
    print(f"\nTeacher Models ({len(teacher_models)}):")
    for m in teacher_models:
        print(f"  - {m}")
    if expert_models:
        print(f"\nExpert Models ({len(expert_models)}):")
        for m in expert_models:
            print(f"  - {m}")
    print(f"\nDatasets ({len(dataset_names)}):")
    for d in dataset_names:
        print(f"  - {d}")

    print(f"\nRound 1: {args.round1_attempts} attempts")
    for effort, ratio in zip(round1_efforts, round1_ratios):
        print(f"  - {effort}: {ratio*100:.1f}%")

    print(f"\nRound 2: {args.round2_attempts} attempts")
    for effort, ratio in zip(round2_efforts, round2_ratios):
        print(f"  - {effort}: {ratio*100:.1f}%")

    print(f"\nMax Tokens: {args.max_tokens}")
    print(f"Workers: {args.workers}")
    print(f"Code Timeout: {args.code_timeout}s")
    print("="*80 + "\n")

    output_dir = Path(args.output_dir)

    config = {
        "datasets": dataset_names,
        "round1_attempts": args.round1_attempts,
        "round2_attempts": args.round2_attempts,
        "workers": args.workers,
        "code_timeout": args.code_timeout,
    }

    result = process_datasets(
        dataset_names=dataset_names,
        teacher_models=teacher_models,
        expert_models=expert_models,
        round1_attempts=args.round1_attempts,
        round2_attempts=args.round2_attempts,
        round1_efforts=round1_efforts,
        round1_ratios=round1_ratios,
        round2_efforts=round2_efforts,
        round2_ratios=round2_ratios,
        max_tokens=args.max_tokens,
        round1_temps=round1_temps,
        round2_temps=round2_temps,
        workers=args.workers,
        count=args.count,
        start_idx=args.start,
        output_dir=output_dir,
        code_timeout=args.code_timeout,
    )

    save_final_results(
        result["results"],
        result["dataset_stats"],
        output_dir,
        teacher_models,
        expert_models,
        config,
        result["base_name"],
        result.get("output_jsonl"),
    )

    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"\nOutput Directory: {output_dir}")
    print(f"Total Solutions: {len(result['results'])}")
    print(f"Unique Samples: {len(set((r['dataset'], r['index']) for r in result['results']))}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

