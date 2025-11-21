#!/usr/bin/env python3
"""
Standardized Evaluation Script for SLM-Math
Usage: python -m evaluation.eval_pipeline --model MODEL_NAME --round ROUND_NAME --dataset DATASET --count COUNT --mode MODE
"""

import sys
import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm

from utils.prompt_utils import (
    extract_answer,
    check_answer,
    format_prompt_standard,
    format_prompt_thinking,
    parse_thinking_output,
    load_dataset_for_eval,
    extract_question_and_answer,
    # Implementation for multi-agent: import multi-agent prompt helpers
    format_prompt_solver,
    format_prompt_checker,
    # Implementation for multi-agent: parse checker verdict and reasoning
    parse_checker_verdict,
    parse_checker_reasoning,
    parse_checker_tip,
)
from models.inference import load_model, generate_response


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation script")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Qwen3-0.6B)")
    parser.add_argument("--round", type=str, required=True, help="Test round name (e.g., round1_standard)")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or dataset-split (e.g., 'gsm8k' or 'gsm8k-train')",
    )
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate (overrides suffix)")
    parser.add_argument("--count", type=int, required=True, help="Number of test cases (0 = run entire dataset)")
    parser.add_argument("--start", type=int, default=0, help="Zero-based index to start evaluation from")
    # Implementation for multi-agent: add new evaluation mode choice
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["standard", "thinking", "multi_agent"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--detailed", type=str, default="false", choices=["true", "false"], help="Detailed output (true/false)"
    )
    parser.add_argument(
        "--checker_model",
        type=str,
        default=None,
        help="Optional checker model name (e.g., Qwen2.5-Math-1.5B). If provided, the checker will load and use this model instead of the solver model.",
    )
    return parser.parse_args()


def run_evaluation(args):
    """Main evaluation function"""
    detailed = args.detailed.lower() == "true"

    print("\n" + "=" * 80)
    print("SLM-Math Evaluation")
    print("=" * 80)

    # Handle dataset name and split suffix (e.g., gsm8k-test)
    dataset_name = args.dataset
    split_name = args.split
    if "-" in dataset_name:
        base, suffix = dataset_name.split("-", 1)
        dataset_name = base
        if split_name is None:
            split_name = suffix
    if split_name is None:
        split_name = "test"
    args.dataset = dataset_name
    args.split = split_name

    print(f"Model: {args.model}")
    print(f"Round: {args.round}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Split: {split_name}")
    print(f"Count: {args.count}")
    print(f"Mode: {args.mode}")
    print(f"Detailed Output: {detailed}")
    print("=" * 80 + "\n")

    base_path = Path(__file__).parent.parent

    # Load model
    try:
        model, tokenizer = load_model(args.model, base_path)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

    # Optionally load a separate checker model
    checker_model = None
    checker_tokenizer = None
    try:
        if args.checker_model and args.checker_model != args.model:
            print(f"Loading checker model: {args.checker_model}")
            checker_model, checker_tokenizer = load_model(args.checker_model, base_path)
        else:
            # Use the solver model for checker by default
            checker_model, checker_tokenizer = model, tokenizer
    except Exception as e:
        print(f"ERROR loading checker model: {e}")
        print("Falling back to using the solver model for checker.")
        checker_model, checker_tokenizer = model, tokenizer

    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    try:
        test_data = load_dataset_for_eval(dataset_name, str(base_path), split_name=split_name)
        total_samples = len(test_data)
        print(f"Dataset split '{split_name}' loaded: {total_samples} total samples")

        start_index = max(0, args.start)
        if start_index >= total_samples:
            print(f"ERROR: start index {start_index} is outside the dataset (size {total_samples}).")
            return None

        remaining = total_samples - start_index
        if args.count == 0:
            num_samples = remaining
            print(f"Testing on remaining {num_samples} samples starting at index {start_index} (sample #{start_index+1})\n")
        else:
            num_samples = min(args.count, remaining)
            print(f"Testing on {num_samples} samples starting at index {start_index} (sample #{start_index+1})\n")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None

    # Results directory with timestamp-based naming
    now = datetime.now()
    month_date = now.strftime("%m%d")
    minute_second = now.strftime("%H%M")

    base_dir_name = f"{args.round}_{args.model}_{args.dataset}_{num_samples}_{month_date}"
    results_dir = base_path / "results" / base_dir_name
    if results_dir.exists():
        base_dir_name = f"{args.round}_{args.model}_{args.dataset}_{num_samples}_{month_date}_{minute_second}"
        results_dir = base_path / "results" / base_dir_name

    log_dir = results_dir / "log"
    answers_dir = results_dir / "answers"
    log_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}\n")

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "mode": args.mode,
        "round": args.round,
        "num_samples": num_samples,
        "start_index": start_index,
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
        "total_time": 0.0,
        "avg_time_per_sample": 0.0,
        "timestamp": datetime.now().isoformat(),
        "predictions": [],
    }

    start_time = time.time()

    log_file = log_dir / f"{args.model}_{args.dataset}_{args.mode}.log"

    def log_and_print(message: str, to_console: bool = True):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        if to_console:
            print(message)

    log_and_print("=" * 80)
    log_and_print(f"Starting evaluation loop - {num_samples} samples")
    log_and_print("=" * 80 + "\n")

    progress_bar = None
    if not detailed:
        progress_bar = tqdm(
            total=num_samples,
            desc="Progress",
            unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    # Run evaluation
    for local_idx, dataset_idx in enumerate(range(start_index, start_index + num_samples)):
        question = None
        ground_truth = None

        log_and_print("\n" + "=" * 80, to_console=detailed)
        log_and_print(f"[Sample {local_idx+1}/{num_samples} | Dataset Index: {dataset_idx}]", to_console=detailed)

        example = test_data[dataset_idx]
        question, ground_truth = extract_question_and_answer(example, args.dataset)

        question_preview = question[:150] + ("..." if len(question) > 150 else "")
        log_and_print(f"Question: {question_preview}", to_console=detailed)
        log_and_print(f"Ground Truth: {ground_truth}", to_console=detailed)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Full Question: {question}\n")

        try:
            # Implementation for multi-agent: iterative solver + checker loop
            if args.mode == "multi_agent":
                sample_start_time = time.time()
                
                # Store all attempts for majority voting if needed
                solver_responses = []
                solver_answers = []
                checker_responses = []
                checker_verdicts = []
                
                checker_feedback = ""  # Initially empty, will be filled by checker when INCORRECT
                max_iterations = 5
                predicted_answer = None
                
                for iteration in range(max_iterations):
                    iteration_num = iteration + 1
                    log_and_print("\n" + "=" * 40, to_console=detailed)
                    log_and_print(f"[Iteration {iteration_num}/{max_iterations}]", to_console=detailed)
                    log_and_print("=" * 40, to_console=detailed)
                    
                    # Step 1: Solver generates response
                    solver_prompt = format_prompt_solver(question, checker_feedback, args.dataset)
                    log_and_print("\n" + "-" * 40, to_console=detailed)
                    log_and_print(f"[Iteration {iteration_num}] Generating Solver response...", to_console=detailed)
                    if checker_feedback:
                        log_and_print(f"Checker feedback: {checker_feedback}", to_console=detailed)
                    log_and_print("-" * 40, to_console=detailed)
                    
                    solver_response = generate_response(model, tokenizer, solver_prompt, "solver", detailed)
                    solver_answer = extract_answer(solver_response)
                    
                    # Fallback if solver produces empty response
                    if not solver_response.strip():
                        log_and_print("Solver response empty, falling back to standard prompt.", to_console=detailed)
                        fallback_prompt = format_prompt_standard(question, args.dataset)
                        solver_response = generate_response(model, tokenizer, fallback_prompt, "standard", detailed)
                        solver_answer = extract_answer(solver_response)
                    
                    solver_responses.append(solver_response)
                    if solver_answer:
                        solver_answers.append(solver_answer)
                    
                    # Step 2: Checker evaluates solver response
                    checker_prompt = format_prompt_checker(question, solver_response, args.dataset)
                    log_and_print("\n" + "-" * 40, to_console=detailed)
                    log_and_print(f"[Iteration {iteration_num}] Generating Checker response...", to_console=detailed)
                    log_and_print("-" * 40, to_console=detailed)
                    
                    checker_response = generate_response(checker_model, checker_tokenizer, checker_prompt, "checker", detailed)
                    
                    # If checker responded with nothing, retry with an even simpler prompt
                    if not checker_response.strip():
                        log_and_print(
                            "Checker produced empty response; retrying with simpler prompt...",
                            to_console=detailed,
                        )
                        # Try a very minimal prompt
                        simple_checker_prompt = f"""Q: {question}\nA: {solver_answer}\n\nVERDICT:"""
                        checker_response = generate_response(checker_model, checker_tokenizer, simple_checker_prompt, "checker", detailed)
                        
                        # If still empty, force a default verdict
                        if not checker_response.strip():
                            log_and_print(
                                "Checker still empty after retry; defaulting to VERDICT: UNCLEAR",
                                to_console=detailed,
                            )
                            checker_response = "VERDICT: UNCLEAR"
                    
                    checker_responses.append(checker_response)
                    checker_verdict = parse_checker_verdict(checker_response)
                    
                    # Ensure we always have a valid verdict
                    if checker_verdict not in ["CORRECT", "INCORRECT", "UNCLEAR"]:
                        checker_verdict = "UNCLEAR"
                    
                    checker_verdicts.append(checker_verdict)
                    
                    log_and_print(f"\nChecker Verdict: {checker_verdict}", to_console=detailed)
                    
                    # Condition 1: If checker says CORRECT, use this answer and break
                    if checker_verdict == "CORRECT":
                        if solver_answer:
                            predicted_answer = solver_answer
                            log_and_print(f"\n✓ Checker confirmed solution is CORRECT. Using answer: {predicted_answer}", to_console=detailed)
                            break
                        else:
                            log_and_print("Warning: Checker says CORRECT but no answer extracted. Continuing...", to_console=detailed)
                    
                    # Condition 2: If checker says INCORRECT, extract tip and loop back
                    elif checker_verdict == "INCORRECT":
                        # Extract tip from checker response
                        checker_feedback = parse_checker_tip(checker_response)
                        
                        if checker_feedback:
                            log_and_print(f"\n✗ Checker says INCORRECT. Tip: {checker_feedback}", to_console=detailed)
                        else:
                            log_and_print("\n✗ Checker says INCORRECT. No tip provided.", to_console=detailed)
                            checker_feedback = "The previous solution was incorrect. Please reconsider the problem."
                        
                        # Continue to next iteration if not at max
                        if iteration_num < max_iterations:
                            log_and_print(f"Looping back to solver with feedback (iteration {iteration_num + 1}/{max_iterations})...", to_console=detailed)
                            continue
                        else:
                            log_and_print("\nReached maximum iterations (5). Proceeding to majority vote...", to_console=detailed)
                            break
                    
                    # Condition 3: If UNCLEAR, continue to next iteration or break if at max
                    else:  # UNCLEAR
                        log_and_print(f"\n? Checker verdict is UNCLEAR. ", to_console=detailed)
                        if iteration_num < max_iterations:
                            log_and_print(f"Continuing to next iteration...", to_console=detailed)
                            checker_feedback = "The previous solution was unclear. Please try again with clearer reasoning."
                            continue
                        else:
                            log_and_print("Reached maximum iterations. Proceeding to majority vote...", to_console=detailed)
                            break
                
                # If we've exhausted all iterations without CORRECT verdict, use majority vote
                if predicted_answer is None:
                    log_and_print("\n" + "=" * 40, to_console=detailed)
                    log_and_print("Using majority vote from all iterations", to_console=detailed)
                    log_and_print("=" * 40, to_console=detailed)
                    
                    if solver_answers:
                        # Count occurrences of each answer
                        from collections import Counter
                        answer_counts = Counter(solver_answers)
                        most_common = answer_counts.most_common(1)[0]
                        predicted_answer = most_common[0]
                        log_and_print(f"Majority vote: {predicted_answer} (appeared {most_common[1]} times out of {len(solver_answers)} attempts)", to_console=detailed)
                    else:
                        # Fallback: use last answer if no answers extracted
                        predicted_answer = solver_answers[-1] if solver_answers else None
                        log_and_print(f"No answers extracted. Using last attempt.", to_console=detailed)

                sample_time = time.time() - sample_start_time

                # Log all iterations
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 40 + "\n")
                    f.write(f"Total Iterations: {len(solver_responses)}\n")
                    f.write("=" * 40 + "\n")
                    
                    for i, (solver_resp, checker_resp, checker_v) in enumerate(zip(solver_responses, checker_responses, checker_verdicts), 1):
                        f.write(f"\n--- Iteration {i} ---\n")
                        f.write("Solver Response:\n")
                        f.write(solver_resp + "\n")
                        f.write("\nChecker Response:\n")
                        f.write(checker_resp + "\n")
                        f.write(f"Checker Verdict: {checker_v}\n")
                        if checker_v == "INCORRECT" and i < len(solver_responses):
                            tip = parse_checker_tip(checker_resp)
                            if tip:
                                f.write(f"Checker Tip: {tip}\n")
                    
                    f.write("\n" + "-" * 40 + "\n")
                    f.write(f"Final Predicted Answer: {predicted_answer}\n")
                    f.write(f"Total multi-agent generation time: {sample_time:.2f}s\n")
                    f.write("-" * 40 + "\n")

                if detailed:
                    print("\n" + "-" * 40)
                    print(f"Total Iterations: {len(solver_responses)}")
                    print(f"Final Answer: {predicted_answer}")
                    print(f"Multi-agent generation time: {sample_time:.2f}s")
                    print("-" * 40)

                prediction_entry = {
                    "question_id": dataset_idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "num_iterations": len(solver_responses),
                    "solver_responses": [resp[:500] for resp in solver_responses],
                    "solver_answers": solver_answers,
                    "checker_responses": [resp[:500] for resp in checker_responses],
                    "checker_verdicts": checker_verdicts,
                    "predicted_answer": predicted_answer,
                    "correct": False,
                }

            else:
                # Original single-agent behavior
                if args.mode == "thinking":
                    prompt = format_prompt_thinking(question, args.dataset)
                else:
                    prompt = format_prompt_standard(question, args.dataset)

                log_and_print("\n" + "-" * 40, to_console=detailed)
                log_and_print(f"Generating ({args.mode} mode)...", to_console=detailed)
                log_and_print("-" * 40, to_console=detailed)

                sample_start_time = time.time()
                response = generate_response(model, tokenizer, prompt, args.mode, detailed)
                sample_time = time.time() - sample_start_time

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("\n" + "-" * 40 + "\n")
                    f.write("Full Response:\n")
                    f.write(response + "\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Generation time: {sample_time:.2f}s\n")
                    f.write("-" * 40 + "\n")

                if detailed:
                    print("\n" + "-" * 40)
                    print(f"Generation time: {sample_time:.2f}s")
                    print("-" * 40)

                if args.mode == "thinking":
                    parsed = parse_thinking_output(response)
                    predicted_answer = parsed["final_answer"]
                    prediction_entry = {
                        "question_id": dataset_idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "analysis": parsed["analysis"][:300],
                        "chain_of_thought": parsed["chain_of_thought"][:500],
                        "predicted_answer": predicted_answer,
                        "full_response": response[:1000],
                        "correct": False,
                    }
                else:
                    predicted_answer = extract_answer(response)
                    prediction_entry = {
                        "question_id": dataset_idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "predicted_answer": predicted_answer,
                        "response": response[:500],
                        "correct": False,
                    }

            # Check correctness and update results (common across modes)
            is_correct = check_answer(predicted_answer, ground_truth)
            prediction_entry["correct"] = is_correct

            if is_correct:
                results["correct"] += 1

            results["total"] += 1
            results["predictions"].append(prediction_entry)

            current_accuracy = results["correct"] / results["total"]
            status_symbol = "✓" if is_correct else "✗"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{status_symbol} Predicted: {predicted_answer}\n")
                f.write(f"{status_symbol} Expected: {ground_truth}\n")
                f.write(f"{status_symbol} Result: {'CORRECT' if is_correct else 'WRONG'}\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"Running Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)\n"
                )
                f.write("=" * 80 + "\n")

            if detailed:
                print(f"\n{status_symbol} Predicted: {predicted_answer}")
                print(f"{status_symbol} Expected: {ground_truth}")
                print(f"{status_symbol} Result: {'CORRECT' if is_correct else 'WRONG'}")
                print("-" * 40)

            if detailed:
                print(
                    f"[{local_idx+1}/{num_samples}] {status_symbol} {predicted_answer} "
                    f"(Expected: {ground_truth}) | "
                    f"Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)"
                )
                print("=" * 80)
            else:
                progress_bar.set_postfix(
                    {"Accuracy": f"{current_accuracy*100:.1f}%", "Correct": f"{results['correct']}/{results['total']}"}
                )
                progress_bar.update(1)

        except Exception as e:
            error_msg = f"ERROR processing sample {dataset_idx}: {e}"
            log_and_print("\n✗ " + error_msg)

            results["total"] += 1
            results["predictions"].append(
                {
                    "question_id": dataset_idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted_answer": None,
                    "response": f"Error: {e}",
                    "correct": False,
                }
            )
            current_accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0.0

            if detailed:
                print(
                    f"[{local_idx+1}/{num_samples}] ✗ ERROR | "
                    f"Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)"
                )
                print("=" * 80)
            else:
                progress_bar.set_postfix(
                    {"Accuracy": f"{current_accuracy*100:.1f}%", "Correct": f"{results['correct']}/{results['total']}"}
                )
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    end_time = time.time()

    results["total_time"] = end_time - start_time
    results["avg_time_per_sample"] = results["total_time"] / results["total"] if results["total"] > 0 else 0.0
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0

    print("\n" + "=" * 80)
    print("Results Summary:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"  Total Time: {results['total_time']:.2f} seconds")
    print(f"  Avg Time/Sample: {results['avg_time_per_sample']:.2f} seconds")
    print("=" * 80 + "\n")

    # Save answers JSON
    answer_file = answers_dir / f"{args.model}_{args.dataset}_{args.mode}_answers.json"
    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved answers to: {answer_file}")

    # Save metrics CSV
    metrics_csv = results_dir / f"metrics_{args.mode}.csv"
    metrics_data = {
        "model": [args.model],
        "dataset": [args.dataset],
        "mode": [args.mode],
        "accuracy": [results["accuracy"]],
        "correct": [results["correct"]],
        "total": [results["total"]],
        "avg_time_per_sample": [results["avg_time_per_sample"]],
        "total_time": [results["total_time"]],
    }
    df = pd.DataFrame(metrics_data)
    if metrics_csv.exists():
        existing_df = pd.read_csv(metrics_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(metrics_csv, index=False)
    print(f"Updated metrics: {metrics_csv}")

    # Save metrics TXT
    metrics_txt = results_dir / f"metrics_{args.mode}.txt"
    with open(metrics_txt, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Samples: {results['total']}\n")
        f.write(f"Correct: {results['correct']}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Total Time: {results['total_time']:.2f}s\n")
        f.write(f"Avg Time/Sample: {results['avg_time_per_sample']:.2f}s\n")
        f.write("=" * 80 + "\n\n")
    print(f"Updated metrics: {metrics_txt}\n")

    # Cleanup
    # If a separate checker model was loaded, try to free it as well
    try:
        if 'checker_model' in locals() and checker_model is not None and checker_model is not model:
            del checker_model
            del checker_tokenizer
    except Exception:
        pass

    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    args = parse_args()
    result = run_evaluation(args)
    if result is None:
        sys.exit(1)
    sys.exit(0)
