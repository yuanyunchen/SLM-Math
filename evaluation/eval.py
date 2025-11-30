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
    extract_question_and_answer
)
from dataset.dataloader import load_dataset_for_eval
from models.inference import load_model, generate_response, generate_response_batch, load_inference_engine_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation script')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., Qwen3-0.6B)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (LoRA adapter or fine-tuned model). Can be relative to project root or absolute path.')
    parser.add_argument('--round', type=str, required=True, help='Test round name (e.g., round1_standard)')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name or dataset-split (e.g., 'gsm8k' or 'gsm8k-train')")
    parser.add_argument('--split', type=str, default=None, help='Dataset split to evaluate (overrides suffix)')
    parser.add_argument('--count', type=int, required=True, help='Number of test cases (0 = run entire dataset)')
    parser.add_argument('--start', type=int, default=0, help='Zero-based index to start evaluation from')
    parser.add_argument('--mode', type=str, required=True, choices=['standard'], help='Evaluation mode')
    parser.add_argument('--detailed', type=str, default='false', choices=['true', 'false'], help='Streaming console output for each sample (true/false)')
    parser.add_argument('--log_samples', type=str, default='true', choices=['true', 'false'], help='Log full sample details to file (true/false)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from existing results directory (e.g., results/round1_model_gsm8k_1000_0101)')
    parser.add_argument('--save_interval', type=int, default=10, help='Save intermediate results every N samples (default: 10)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (default: 1, recommended: 8-32)')
    parser.add_argument('--inference_backend', type=str, default='transformers', choices=['transformers', 'vllm'], help='Inference backend (default: transformers)')
    parser.add_argument('--greedy', type=str, default='true', choices=['true', 'false'], help='Use greedy decoding (default: true). When true, ignores temperature/top_p/etc.')
    return parser.parse_args()

def run_evaluation(args):
    """Main evaluation function"""
    # Convert args to boolean
    detailed = args.detailed.lower() == 'true'      # Streaming console output
    log_samples = args.log_samples.lower() == 'true'  # Log full sample details to file
    greedy = args.greedy.lower() == 'true'
    
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
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Round: {args.round}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Split: {split_name}")
    print(f"Count: {args.count}")
    print(f"Mode: {args.mode}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Inference Backend: {args.inference_backend}")
    print(f"Greedy Decoding: {greedy}")
    print(f"Streaming Output: {detailed}")
    print(f"Log Samples: {log_samples}")
    print(f"{'='*80}\n")
    
    # Setup paths
    base_path = Path(__file__).parent.parent
    
    # Load model/inference engine
    try:
        if args.batch_size > 1 or args.inference_backend == 'vllm':
            # Use new inference engine for batch processing or vLLM
            (model, tokenizer), inference_engine = load_inference_engine_wrapper(
                args.model, 
                base_path, 
                backend=args.inference_backend,
                checkpoint_path=args.checkpoint
            )
            # If inference_engine is None (e.g., checkpoint fallback), disable batch inference
            # unless batch_size > 1 (in which case we can use transformers batch generation)
            if inference_engine is None and args.batch_size == 1:
                use_batch_inference = False
            else:
                use_batch_inference = (args.batch_size > 1) or (inference_engine is not None)
        else:
            # Use original load_model for backward compatibility
            model, tokenizer = load_model(args.model, base_path, checkpoint_path=args.checkpoint)
            inference_engine = None
            use_batch_inference = False
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
    
    # Handle resume or create new results directory
    resume_mode = args.resume is not None
    processed_ids = set()
    
    if resume_mode:
        # Resume from existing directory
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = base_path / args.resume
        
        if not resume_path.exists():
            print(f"ERROR: Resume directory not found: {resume_path}")
            return None
        
        results_dir = resume_path
        log_dir = results_dir / 'log'
        answers_dir = results_dir / 'answers'
        
        # Load existing results
        answer_file = answers_dir / f"{args.model}_{args.dataset}_{args.mode}_answers.json"
        if answer_file.exists():
            print(f"Resuming from existing results: {answer_file}")
            with open(answer_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Extract processed question IDs
            processed_ids = {pred['question_id'] for pred in results.get('predictions', [])}
            print(f"Found {len(processed_ids)} already processed samples")
            
            # Calculate elapsed time from previous run
            if 'total_time' in results:
                start_time = time.time() - results['total_time']
            else:
                start_time = time.time()
        else:
            print(f"WARNING: No existing results file found at {answer_file}, starting fresh")
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
    else:
        # Create new results directory with timestamp-based naming
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
    
    print(f"Results directory: {results_dir}")
    if resume_mode:
        print(f"Resume mode: ON (will skip {len(processed_ids)} already processed samples)\n")
    else:
        print(f"Resume mode: OFF (starting fresh)\n")
    
    # Setup logging to file
    log_file = log_dir / f"{args.model}_{args.dataset}_{args.mode}.log"
    samples_log_file = log_dir / f"{args.model}_{args.dataset}_{args.mode}_samples.log"
    
    def log_and_print(message, to_console=True):
        """Log to file and optionally print to console"""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        if to_console:
            print(message)
    
    def log_sample(sample_idx, question, ground_truth, predicted_answer, response, is_correct):
        """Log full sample details to samples log file"""
        if not log_samples:
            return
        with open(samples_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[Sample {sample_idx}] {'CORRECT' if is_correct else 'WRONG'}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Question:\n{question}\n")
            f.write(f"\n{'─'*40}\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Predicted: {predicted_answer}\n")
            f.write(f"\n{'─'*40}\n")
            f.write(f"Full Response:\n{response[:2000]}{'...(truncated)' if len(response) > 2000 else ''}\n")
            f.write(f"{'='*80}\n")
    
    log_and_print(f"{'='*80}")
    if resume_mode:
        remaining_count = num_samples - len(processed_ids)
        log_and_print(f"Resuming evaluation - {remaining_count} remaining samples (skipping {len(processed_ids)} already processed)")
    else:
        log_and_print(f"Starting evaluation loop - {num_samples} samples")
    log_and_print(f"{'='*80}\n")
    
    # Calculate remaining samples to process
    remaining_samples = [idx for idx in range(start_index, start_index + num_samples) if idx not in processed_ids]
    remaining_count = len(remaining_samples)
    
    # Check if all samples are already processed
    if remaining_count == 0:
        print(f"\n{'='*80}")
        print(f"All samples have already been processed!")
        print(f"Total samples: {len(results['predictions'])}")
        print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
        print(f"{'='*80}\n")
        return results
    
    # Initialize progress bar for non-detailed mode
    progress_bar = None
    if not detailed:
        initial_processed = len(processed_ids)
        progress_bar = tqdm(total=num_samples, initial=initial_processed, desc="Progress", unit="sample", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    # Helper function to save intermediate results
    answer_file = answers_dir / f"{args.model}_{args.dataset}_{args.mode}_answers.json"
    def save_intermediate_results():
        """Save current results to file"""
        results["total_time"] = time.time() - start_time
        results["avg_time_per_sample"] = results["total_time"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
        with open(answer_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Run evaluation
    processed_count = 0
    
    # Choose evaluation loop based on batch size
    if use_batch_inference and args.batch_size > 1:
        # Batch processing mode
        batch_size = args.batch_size
        num_batches = (remaining_count + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, remaining_count)
            batch_indices = remaining_samples[batch_start:batch_end]
            
            # Collect batch data
            batch_data = []
            for dataset_idx in batch_indices:
                example = test_data[dataset_idx]
                question, ground_truth = extract_question_and_answer(example, args.dataset)
                prompt = format_prompt_standard(question, args.dataset)
                batch_data.append({
                    'dataset_idx': dataset_idx,
                    'question': question,
                    'ground_truth': ground_truth,
                    'prompt': prompt
                })
            
            # Batch generation
            try:
                batch_prompts = [item['prompt'] for item in batch_data]
                
                if detailed:
                    print(f"\n{'='*80}")
                    print(f"Batch {batch_idx + 1}/{num_batches} - Processing {len(batch_prompts)} samples")
                    print(f"{'='*80}")
                
                batch_start_time = time.time()
                
                if inference_engine:
                    # Use inference engine
                    if greedy:
                        batch_responses = inference_engine.generate_batch(
                            batch_prompts,
                            max_new_tokens=4096,
                            temperature=1.0,  # Ignored when do_sample=False
                            do_sample=False,
                            detailed=detailed
                        )
                    else:
                        batch_responses = inference_engine.generate_batch(
                            batch_prompts,
                            max_new_tokens=4096,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.95,
                            repetition_penalty=1.15,
                            detailed=detailed
                        )
                else:
                    # Use transformers batch generation
                    batch_responses = generate_response_batch(
                        model, tokenizer, batch_prompts, args.mode, detailed, greedy=greedy
                    )
                
                batch_time = time.time() - batch_start_time
                
                # Process each response in batch
                for item, response in zip(batch_data, batch_responses):
                    processed_count += 1
                    dataset_idx = item['dataset_idx']
                    question = item['question']
                    ground_truth = item['ground_truth']
                    
                    sample_time = batch_time / len(batch_responses)
                    
                    # Log to file
                    log_and_print(f"\n{'─'*80}", to_console=detailed)
                    log_and_print(f"[Sample {processed_count}/{remaining_count} | Index: {dataset_idx}]", to_console=detailed)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Full Question: {question}\n")
                        f.write(f"Full Response:\n{response}\n")
                        f.write(f"{'─'*40}\n")
                    
                    # Extract and check answer
                    predicted_answer = extract_answer(response)
                    is_correct = check_answer(predicted_answer, ground_truth)
                    
                    prediction_entry = {
                        "question_id": dataset_idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "predicted_answer": predicted_answer,
                        "response": response[:500],
                        "correct": is_correct
                    }
                    
                    if is_correct:
                        results["correct"] += 1
                    
                    results["total"] += 1
                    results["predictions"].append(prediction_entry)
                    
                    # Update display
                    current_accuracy = results["correct"] / results["total"]
                    status_symbol = "✓" if is_correct else "✗"
                    
                    if detailed:
                        print(f"{status_symbol} Predicted: {predicted_answer} (Expected: {ground_truth})")
                        print(f"Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)")
                    else:
                        progress_bar.set_postfix({'Accuracy': f'{current_accuracy*100:.1f}%', 'Correct': f'{results["correct"]}/{results["total"]}'})
                        progress_bar.update(1)
                
                # Save periodically
                if (batch_idx + 1) % max(1, args.save_interval // batch_size) == 0:
                    save_intermediate_results()
                    if detailed:
                        print(f"Saved intermediate results")
            
            except Exception as e:
                error_msg = f"ERROR processing batch {batch_idx}: {e}"
                log_and_print(f"\n{error_msg}")
                # Mark all samples in batch as errors
                for item in batch_data:
                    processed_count += 1
                    results["total"] += 1
                    results["predictions"].append({
                        "question_id": item['dataset_idx'],
                        "question": item['question'],
                        "ground_truth": item['ground_truth'],
                        "predicted_answer": None,
                        "response": f"Error: {e}",
                        "correct": False
                    })
                    if not detailed:
                        current_accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
                        progress_bar.set_postfix({'Accuracy': f'{current_accuracy*100:.1f}%', 'Correct': f'{results["correct"]}/{results["total"]}'})
                        progress_bar.update(1)
    
    else:
        # Original single-sample processing mode
        for local_idx, dataset_idx in enumerate(remaining_samples):
            question = None
            ground_truth = None
            
            processed_count += 1
            log_and_print(f"\n{'─'*80}", to_console=detailed)
            log_and_print(f"[Sample {processed_count}/{remaining_count} remaining | Dataset Index: {dataset_idx} | Total: {len(processed_ids) + processed_count}/{num_samples}]", to_console=detailed)
            example = test_data[dataset_idx]
            
            question, ground_truth = extract_question_and_answer(example, args.dataset)
            
            question_preview = question[:150] + ('...' if len(question) > 150 else '')
            log_and_print(f"Question: {question_preview}", to_console=detailed)
            log_and_print(f"Ground Truth: {ground_truth}", to_console=detailed)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Full Question: {question}\n")
            
            prompt = format_prompt_standard(question, args.dataset)
            
            try:
                log_and_print(f"\n{'─'*40}", to_console=detailed)
                log_and_print(f"Generating ({args.mode} mode)...", to_console=detailed)
                log_and_print(f"{'─'*40}", to_console=detailed)
                
                sample_start_time = time.time()
                # Use inference_engine if available (for vLLM with batch_size=1), otherwise use model
                model_to_use = inference_engine if inference_engine is not None else model
                response = generate_response(model_to_use, tokenizer, prompt, args.mode, detailed, greedy=greedy)
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
                    print(f"Generation time: {sample_time:.2f}s")
                    print(f"{'─'*40}")
                
                # Extract answer from response
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
                
                # Log to file (summary)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{status_symbol} Predicted: {predicted_answer}\n")
                    f.write(f"{status_symbol} Expected: {ground_truth}\n")
                    f.write(f"{status_symbol} Result: {'CORRECT' if is_correct else 'WRONG'}\n")
                    f.write(f"{'─'*40}\n")
                    f.write(f"Running Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)\n")
                    f.write(f"{'─'*80}\n")
                
                # Log full sample details to separate file
                log_sample(dataset_idx, question, ground_truth, predicted_answer, response, is_correct)
                
                # Print to console (detailed mode only)
                if detailed:
                    print(f"\n{status_symbol} Predicted: {predicted_answer}")
                    print(f"{status_symbol} Expected: {ground_truth}")
                    print(f"{status_symbol} Result: {'CORRECT' if is_correct else 'WRONG'}")
                    print(f"{'─'*40}")
                
                # Update progress display
                if detailed:
                    print(f"[{processed_count}/{remaining_count} remaining | Total: {len(processed_ids) + processed_count}/{num_samples}] {status_symbol} {predicted_answer} (Expected: {ground_truth}) | Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)")
                    print(f"{'─'*80}")
                else:
                    # Update progress bar with accuracy info
                    progress_bar.set_postfix({'Accuracy': f'{current_accuracy*100:.1f}%', 'Correct': f'{results["correct"]}/{results["total"]}'})
                    progress_bar.update(1)
                
                # Save intermediate results periodically
                if processed_count % args.save_interval == 0:
                    save_intermediate_results()
                    if detailed:
                        print(f"Saved intermediate results ({processed_count} new samples processed)")
                
            except Exception as e:
                error_msg = f"ERROR processing sample {dataset_idx}: {e}"
                log_and_print(f"\n{error_msg}")
                
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
                    print(f"[{processed_count}/{remaining_count} remaining | Total: {len(processed_ids) + processed_count}/{num_samples}] ERROR | Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)")
                    print(f"{'─'*80}")
                else:
                    progress_bar.set_postfix({'Accuracy': f'{current_accuracy*100:.1f}%', 'Correct': f'{results["correct"]}/{results["total"]}'})
                    progress_bar.update(1)
                
                # Save intermediate results periodically
                if processed_count % args.save_interval == 0:
                    save_intermediate_results()
                    if detailed:
                        print(f"Saved intermediate results ({processed_count} new samples processed)")
    
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
    if resume_mode:
        print(f"  Mode: Resume (processed {processed_count} new samples)")
    print(f"  Total Samples: {len(results['predictions'])}")
    print(f"  Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"  Total Time: {results['total_time']:.2f} seconds")
    print(f"  Avg Time/Sample: {results['avg_time_per_sample']:.2f} seconds")
    print(f"{'='*80}\n")
    
    # Save final results
    save_intermediate_results()
    print(f"Saved final results to: {answer_file}")
    
    # Save metrics CSV (unified format)
    metrics_csv = results_dir / "metrics.csv"
    metrics_data = {
        'model': [args.model],
        'agent': ['base_direct'],
        'dataset': [args.dataset],
        'total_samples': [results['total']],
        'correct': [results['correct']],
        'accuracy': [results['accuracy']],
        'first_try_correct': [results['correct']],  # For base direct, first=final
        'first_try_accuracy': [results['accuracy']],
        'avg_iterations': [1.0],  # Base direct always 1 iteration
        'improved_cases': [0],  # N/A for base direct
        'degraded_cases': [0],
        'failed_cases': [results['total'] - results['correct']],
        'avg_time_per_sample': [results['avg_time_per_sample']],
        'total_time': [results['total_time']],
        'timestamp': [results['timestamp']]
    }
    df = pd.DataFrame(metrics_data)
    df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics: {metrics_csv}")
    
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
    if model is not None:
        del model
    if inference_engine is not None:
        del inference_engine
    torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    args = parse_args()
    result = run_evaluation(args)
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)

