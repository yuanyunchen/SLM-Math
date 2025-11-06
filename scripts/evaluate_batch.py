#!/usr/bin/env python3
"""
Standardized Evaluation Script for SLM-Math
Usage: python evaluate_batch.py --model MODEL_NAME --round ROUND_NAME --dataset DATASET --count COUNT --mode MODE
"""

import os
import sys
import json
import time
import argparse
import torch
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from tqdm import tqdm
from datetime import datetime
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation script')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., Qwen3-0.6B)')
    parser.add_argument('--round', type=str, required=True, help='Test round name (e.g., round1_standard)')
    parser.add_argument('--dataset', type=str, required=True, choices=['gsm8k', 'math'], help='Dataset name')
    parser.add_argument('--count', type=int, required=True, help='Number of test cases (0 = run entire dataset)')
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'thinking'], help='Evaluation mode')
    parser.add_argument('--detailed', type=str, default='false', choices=['true', 'false'], help='Detailed output (true/false)')
    return parser.parse_args()

def extract_answer(text: str) -> str:
    """Extract numerical answer from model output"""
    patterns = [
        r'####\s*([+-]?\d+\.?\d*)',
        r'(?:answer|Answer|ANSWER)[\s:=]+\$?([+-]?\d+\.?\d*)',
        r'\\boxed\{([^}]+)\}',
        r'\$([+-]?\d+\.?\d*)\s*$',
        r'([+-]?\d+\.?\d*)\s*(?:dollars?|eggs?|meters?|bolts?|people?|students?|items?)\s*(?:\.|$)',
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).strip()
            if answer:
                return normalize_answer(answer)
    
    last_numbers = re.findall(r'([+-]?\d+\.?\d*)', text[-200:])
    if last_numbers:
        return normalize_answer(last_numbers[-1])
    
    return None

def normalize_answer(answer: str) -> str:
    """Normalize answer to standardized format"""
    answer = answer.strip()
    answer = answer.replace(',', '').replace('$', '').replace('%', '')
    answer = answer.replace(' ', '')
    
    try:
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                if den != 0:
                    answer = str(num / den)
        
        float_val = float(answer)
        if float_val.is_integer():
            answer = str(int(float_val))
        else:
            answer = str(float_val)
    except:
        pass
    
    return answer.lower()

def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth"""
    if not predicted or not ground_truth:
        return False
    
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)
    
    if pred_normalized == gt_normalized:
        return True
    
    try:
        pred_num = float(pred_normalized)
        gt_num = float(gt_normalized)
        return abs(pred_num - gt_num) < 1e-3
    except:
        pass
    
    return pred_normalized in gt_normalized or gt_normalized in pred_normalized

def format_prompt_standard(question: str, dataset_name: str) -> str:
    """Format prompt for standard (non-thinking) mode - using common benchmark format"""
    if dataset_name == "gsm8k":
        return f"""{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""
    else:  # math dataset
        return f"""{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""

def format_prompt_thinking(question: str, dataset_name: str) -> str:
    """Format prompt for thinking mode - simple step-by-step format"""
    return f"""{question}

Let me solve this step by step and put the final answer in \\boxed{{}}.

Step 1:"""

def parse_thinking_output(response: str) -> dict:
    """Parse thinking mode output to extract analysis, CoT, and answer"""
    result = {
        'analysis': '',
        'chain_of_thought': '',
        'final_answer': ''
    }
    
    # Try to extract Analysis section (multiple formats)
    analysis_patterns = [
        r'\*\*Analysis\*\*:?(.*?)(?:\*\*|$)',
        r'Analysis:?(.*?)(?:Chain of Thought|Step|$)',
        r'<Analysis>(.*?)(?:</Analysis>|<|$)',
    ]
    for pattern in analysis_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result['analysis'] = match.group(1).strip()[:500]
            break
    
    # If no analysis found, use first part of response
    if not result['analysis']:
        lines = response.split('\n')
        result['analysis'] = ' '.join(lines[:3])[:500]
    
    # Try to extract Chain of Thought
    cot_patterns = [
        r'\*\*Chain of Thought\*\*:?(.*?)(?:\*\*Final Answer\*\*|\\boxed|$)',
        r'Chain of Thought:?(.*?)(?:Final Answer|\\boxed|$)',
        r'Step-by-step.*?:?(.*?)(?:Final Answer|\\boxed|$)',
    ]
    for pattern in cot_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result['chain_of_thought'] = match.group(1).strip()[:1000]
            break
    
    # If no CoT found, use full response
    if not result['chain_of_thought']:
        result['chain_of_thought'] = response[:1000]
    
    # Extract final answer
    answer = extract_answer(response)
    result['final_answer'] = answer if answer else ''
    
    return result

def load_dataset_for_eval(dataset_name: str, base_path: str):
    """Load dataset from disk"""
    dataset_path = os.path.join(base_path, 'data', dataset_name)
    dataset = load_from_disk(dataset_path)
    
    if 'test' in dataset:
        return dataset['test']
    elif 'train' in dataset:
        return dataset['train']
    else:
        raise ValueError(f"No valid split found in dataset {dataset_name}")

def run_evaluation(args):
    """Main evaluation function"""
    # Convert detailed arg to boolean
    detailed = args.detailed.lower() == 'true'
    
    print(f"\n{'='*80}")
    print(f"SLM-Math Evaluation")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Round: {args.round}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Count: {args.count}")
    print(f"Mode: {args.mode}")
    print(f"Detailed Output: {detailed}")
    print(f"{'='*80}\n")
    
    # Setup paths
    base_path = Path(__file__).parent.parent
    model_dir = base_path / 'models' / args.model
    
    # Check model exists
    if not model_dir.exists():
        print(f"ERROR: Model directory {model_dir} not found!")
        return None
    
    model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not model_files:
        print(f"ERROR: No model files found in {model_dir}!")
        return None
    
    # Load model
    print(f"Loading model from {model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model.eval()
        print(f"Model loaded successfully on CPU\n")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    try:
        test_data = load_dataset_for_eval(args.dataset, str(base_path))
        print(f"Dataset loaded: {len(test_data)} total samples")
        
        # Handle count=0 (run entire dataset)
        if args.count == 0:
            num_samples = len(test_data)
            print(f"Testing on ENTIRE dataset: {num_samples} samples\n")
        else:
            num_samples = min(args.count, len(test_data))
            print(f"Testing on {num_samples} samples\n")
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
    for idx in range(num_samples):
        # Initialize variables to avoid undefined variable errors
        question = None
        ground_truth = None
        
        log_and_print(f"\n{'─'*80}", to_console=detailed)
        log_and_print(f"[Sample {idx+1}/{num_samples}]", to_console=detailed)
        example = test_data[idx]
        
        # Get question and ground truth
        if args.dataset == "gsm8k":
            question = example['question']
            ground_truth = example['answer'].split('####')[-1].strip()
        elif args.dataset == "math":
            question = example['problem']
            solution = example['solution']
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
            if boxed_match:
                ground_truth = boxed_match.group(1)
            else:
                numbers = re.findall(r'[+-]?\d+\.?\d*', solution)
                ground_truth = numbers[-1] if numbers else solution
        
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
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            log_and_print(f"\n{'─'*40}", to_console=detailed)
            log_and_print(f"🤖 Generating ({args.mode} mode)...", to_console=detailed)
            log_and_print(f"{'─'*40}", to_console=detailed)
            
            sample_start_time = time.time()
            
            # Create streamer for real-time token-by-token output (only if detailed)
            if detailed:
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            else:
                streamer = None
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024 if args.mode == "thinking" else 512,
                    temperature=0.1,  # Lower temperature to reduce repetition
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Add repetition penalty
                    streamer=streamer  # Enable token-by-token streaming only if detailed
                )
            
            sample_time = time.time() - sample_start_time
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
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
                    "question_id": idx,
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
                    "question_id": idx,
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
                print(f"[{idx+1}/{num_samples}] {status_symbol} {predicted_answer} (Expected: {ground_truth}) | Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)")
                print(f"{'─'*80}")
            else:
                # Update progress bar with accuracy info
                progress_bar.set_postfix({'Accuracy': f'{current_accuracy*100:.1f}%', 'Correct': f'{results["correct"]}/{results["total"]}'})
                progress_bar.update(1)
            
        except Exception as e:
            error_msg = f"ERROR processing sample {idx}: {e}"
            log_and_print(f"\n✗ {error_msg}")
            
            results["total"] += 1
            results["predictions"].append({
                "question_id": idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": None,
                "response": f"Error: {e}",
                "correct": False
            })
            current_accuracy = results["correct"] / results["total"]
            
            # Update progress display for error
            if detailed:
                print(f"[{idx+1}/{num_samples}] ✗ ERROR | Score: {results['correct']}/{results['total']} ({current_accuracy*100:.1f}%)")
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

