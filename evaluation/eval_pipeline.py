#!/usr/bin/env python3
"""
Standardized Evaluation Script for SLM-Math
Usage: python evaluate_batch.py --model MODEL_NAME --round ROUND_NAME --dataset DATASET --count COUNT --mode MODE
"""

import sys
import json
import time
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from utils.prompt_utils import (
    extract_answer, 
    check_answer,
    format_prompt_standard, 
    format_prompt_thinking,
    parse_thinking_output,
    load_dataset_for_eval,
    extract_question_and_answer
)
from models.inference import load_model, generate_response

def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation script')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., Qwen3-0.6B)')
    parser.add_argument('--round', type=str, required=True, help='Test round name (e.g., round1_standard)')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name or dataset-split (e.g., 'gsm8k' or 'gsm8k-train')")
    parser.add_argument('--split', type=str, default=None, help='Dataset split to evaluate (overrides suffix)')
    parser.add_argument('--count', type=int, required=True, help='Number of test cases (0 = run entire dataset)')
    parser.add_argument('--start', type=int, default=0, help='Zero-based index to start evaluation from')
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'thinking'], help='Evaluation mode')
    parser.add_argument('--detailed', type=str, default='false', choices=['true', 'false'], help='Detailed output (true/false)')
    return parser.parse_args()

def run_evaluation(args):
    """Main evaluation function"""
    # Convert detailed arg to boolean
    detailed = args.detailed.lower() == 'true'
    
    print(f"\n{'='*80}")
    print(f"SLM-Math Evaluation")
    print(f"{'='*80}")
    dataset_name = args.dataset
    split_name = args.split
    if '-' in dataset_name:
        parts = dataset_name.split('-', 1)
        dataset_name, suffix = parts[0], parts[1]
        if split_name is None:
            split_name = suffix
    if split_name is None:
        split_name = 'test'
    args.dataset = dataset_name
    args.split = split_name

    print(f"Model: {args.model}")
    print(f"Round: {args.round}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Split: {split_name}")
    print(f"Count: {args.count}")
    print(f"Mode: {args.mode}")
    print(f"Detailed Output: {detailed}")
    print(f"{'='*80}\n")
    
    # Setup paths
    base_path = Path(__file__).parent.parent
    
    # Load model
    try:
        model, tokenizer = load_model(args.model, base_path)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None
    
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

        # Handle count=0 (run entire remaining dataset)
        if args.count == 0:
            num_samples = remaining
            print(f"Testing on remaining {num_samples} samples starting at index {start_index} (sample #{start_index+1})\n")
        else:
            num_samples = min(args.count, remaining)
            print(f"Testing on {num_samples} samples starting at index {start_index} (sample #{start_index+1})\n")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None
    
    # Create results directory with timestamp-based naming
    # Format: roundName_ModelName_Dataset_Count_MonthDate[_MinuteSecond]
    now = datetime.now()
    month_date = now.strftime("%m%d")
    minute_second = now.strftime("%H%M")
    
    base_dir_name = f"{args.round}_{args.model}_{args.dataset}_{num_samples}_{month_date}"
    results_dir = base_path / 'results' / base_dir_name
    
    # If directory already exists, add minute_second
    if results_dir.exists():
        base_dir_name = f"{args.round}_{args.model}_{args.dataset}_{num_samples}_{month_date}_{minute_second}"
        results_dir = base_path / 'results' / base_dir_name
    
    log_dir = results_dir / 'log'
    answers_dir = results_dir / 'answers'
    
    # Create directories
    log_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results directory: {results_dir}\n")
    
    # Prepare results
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
        "predictions": []
    }
    
    start_time = time.time()
    
    # Setup logging to file
    log_file = log_dir / f"{args.model}_{args.dataset}_{args.mode}.log"
    
    def log_and_print(message, to_console=True):
        """Log to file and optionally print to console"""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        if to_console:
            print(message)
    
    log_and_print(f"{'='*80}")
    log_and_print(f"Starting evaluation loop - {num_samples} samples")
    log_and_print(f"{'='*80}\n")
    
    # Initialize progress bar for non-detailed mode
    progress_bar = None
    if not detailed:
        progress_bar = tqdm(total=num_samples, desc="Progress", unit="sample", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    # Run evaluation
    for local_idx, dataset_idx in enumerate(range(start_index, start_index + num_samples)):
        # Initialize variables to avoid undefined variable errors
        question = None
        ground_truth = None
        
        log_and_print(f"\n{'─'*80}", to_console=detailed)
        log_and_print(f"[Sample {local_idx+1}/{num_samples} | Dataset Index: {dataset_idx}]", to_console=detailed)
        example = test_data[dataset_idx]
        
        # Get question and ground truth
        question, ground_truth = extract_question_and_answer(example, args.dataset)
        
        question_preview = question[:150] + ('...' if len(question) > 150 else '')
        log_and_print(f"Question: {question_preview}", to_console=detailed)
        log_and_print(f"Ground Truth: {ground_truth}", to_console=detailed)
        
        # Always log full question to file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Full Question: {question}\n")
        
        # Format prompt based on mode
        if args.mode == "thinking":
            prompt = format_prompt_thinking(question, args.dataset)
        else:
            prompt = format_prompt_standard(question, args.dataset)
        
        # Generate response
        try:
            log_and_print(f"\n{'─'*40}", to_console=detailed)
            log_and_print(f"🤖 Generating ({args.mode} mode)...", to_console=detailed)
            log_and_print(f"{'─'*40}", to_console=detailed)
            
            sample_start_time = time.time()
            response = generate_response(model, tokenizer, prompt, args.mode, detailed)
            sample_time = time.time() - sample_start_time
            
            # Log full response to file always
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'─'*40}\n")
                f.write(f"Full Response:\n{response}\n")
                f.write(f"{'─'*40}\n")
                f.write(f"Generation time: {sample_time:.2f}s\n")
                f.write(f"{'─'*40}\n")
            
            if detailed:
                print(f"\n{'─'*40}")
                print(f"⏱️  Generation time: {sample_time:.2f}s")
                print(f"{'─'*40}")
            
            # Parse response based on mode
            if args.mode == "thinking":
                parsed = parse_thinking_output(response)
                predicted_answer = parsed['final_answer']
                
                prediction_entry = {
                    "question_id": dataset_idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "analysis": parsed['analysis'][:300],
                    "chain_of_thought": parsed['chain_of_thought'][:500],
                    "predicted_answer": predicted_answer,
                    "full_response": response[:1000],
                    "correct": False
                }
            else:
                predicted_answer = extract_answer(response)
                prediction_entry = {
                    "question_id": dataset_idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted_answer": predicted_answer,
                    "response": response[:500],
                    "correct": False
                }
            
            # Check correctness
            is_correct = check_answer(predicted_answer, ground_truth)
            prediction_entry["correct"] = is_correct
            
            if is_correct:
                results["correct"] += 1
            
            results["total"] += 1
            results["predictions"].append(prediction_entry)
            
            # Print immediate result
            current_accuracy = results["correct"] / results["total"]
            status_symbol = "✓" if is_correct else "✗"
            
            # Log to file (always)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{status_symbol} Predicted: {predicted_answer}\n")
                f.write(f"{status_symbol} Expected: {ground_truth}\n")
                f.write(f"{status_symbol} Result: {'CORRECT' if is_correct else 'WRONG'}\n")
                f.write(f"{'─'*40}\n")
                f.write(f"Running Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)\n")
                f.write(f"{'─'*80}\n")
            
            # Print to console (detailed mode or summary)
            if detailed:
                print(f"\n{status_symbol} Predicted: {predicted_answer}")
                print(f"{status_symbol} Expected: {ground_truth}")
                print(f"{status_symbol} Result: {'CORRECT' if is_correct else 'WRONG'}")
                print(f"{'─'*40}")
            
            # Update progress display
            if detailed:
                print(f"[{local_idx+1}/{num_samples}] {status_symbol} {predicted_answer} (Expected: {ground_truth}) | Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)")
                print(f"{'─'*80}")
            else:
                # Update progress bar with accuracy info
                progress_bar.set_postfix({'Accuracy': f'{current_accuracy*100:.1f}%', 'Correct': f'{results["correct"]}/{results["total"]}'})
                progress_bar.update(1)
            
        except Exception as e:
            error_msg = f"ERROR processing sample {dataset_idx}: {e}"
            log_and_print(f"\n✗ {error_msg}")
            
            results["total"] += 1
            results["predictions"].append({
                "question_id": dataset_idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": None,
                "response": f"Error: {e}",
                "correct": False
            })
            current_accuracy = results["correct"] / results["total"]
            
            # Update progress display for error
            if detailed:
                print(f"[{local_idx+1}/{num_samples}] ✗ ERROR | Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)")
                print(f"{'─'*80}")
            else:
                progress_bar.set_postfix({'Accuracy': f'{current_accuracy*100:.1f}%', 'Correct': f'{results["correct"]}/{results["total"]}'})
                progress_bar.update(1)
    
    # Close progress bar if used
    if progress_bar is not None:
        progress_bar.close()
    
    end_time = time.time()
    
    # Calculate metrics
    results["total_time"] = end_time - start_time
    results["avg_time_per_sample"] = results["total_time"] / results["total"] if results["total"] > 0 else 0
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Results Summary:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"  Total Time: {results['total_time']:.2f} seconds")
    print(f"  Avg Time/Sample: {results['avg_time_per_sample']:.2f} seconds")
    print(f"{'='*80}\n")
    
    # Save results
    answer_file = answers_dir / f"{args.model}_{args.dataset}_{args.mode}_answers.json"
    with open(answer_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved answers to: {answer_file}")
    
    # Save metrics CSV
    metrics_csv = results_dir / f"metrics_{args.mode}.csv"
    metrics_data = {
        'model': [args.model],
        'dataset': [args.dataset],
        'mode': [args.mode],
        'accuracy': [results['accuracy']],
        'correct': [results['correct']],
        'total': [results['total']],
        'avg_time_per_sample': [results['avg_time_per_sample']],
        'total_time': [results['total_time']]
    }
    df = pd.DataFrame(metrics_data)
    
    if metrics_csv.exists():
        existing_df = pd.read_csv(metrics_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(metrics_csv, index=False)
    print(f"Updated metrics: {metrics_csv}")
    
    # Save metrics TXT
    metrics_txt = results_dir / f"metrics_{args.mode}.txt"
    with open(metrics_txt, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Samples: {results['total']}\n")
        f.write(f"Correct: {results['correct']}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Total Time: {results['total_time']:.2f}s\n")
        f.write(f"Avg Time/Sample: {results['avg_time_per_sample']:.2f}s\n")
        f.write(f"{'='*80}\n\n")
    print(f"Updated metrics: {metrics_txt}\n")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    args = parse_args()
    result = run_evaluation(args)
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)

