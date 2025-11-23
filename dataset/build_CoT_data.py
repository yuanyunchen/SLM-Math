#!/usr/bin/env python3
"""
Optimized End-to-End CoT Data Generation Script

Two-Round Architecture with Multi-Model Support:
- Round 1: Teacher model(s) with retry
- Round 2: Expert model(s) with retry for failures

Features:
- Multiple models per round (parallel testing)
- Keep all correct solutions from same round
- Multiple datasets support
- Clean final output (only successful samples)

Usage:
    python -m dataset.build_CoT_data \\
        --teacher "x-ai/grok-4-1-fast-reasoning,deepseek/deepseek-reasoner-v3.1" \\
        --expert "alibaba/qwen3-235b-a22b-thinking-2507,minimax/m2" \\
        --dataset "gsm8k-train,math-train" \\
        --round1-attempts 3 \\
        --round2-attempts 5 \\
        --count 100 \\
        --workers 8 \\
        --api-key YOUR_KEY \\
        --output-dir data/cot_generated/multi_model_test
"""

import sys
import json
import time
import random
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dataset.dataloader import load_dataset_for_eval
from utils.prompt_utils import extract_answer, check_answer

# Import openai after path setup
import openai


def initialize_api(api_key: str, api_base: str):
    """Initialize OpenAI API client"""
    openai.api_key = api_key
    openai.base_url = api_base


def get_system_prompt(thinking_effort: str) -> str:
    """Get system prompt based on thinking effort"""
    # effort_configs = {
    #     'low': {
    #         'instruction': 'Solve this math problem. Show your work briefly.',
    #         'thinking_tokens': 400
    #     },
    #     'medium': {
    #         'instruction': 'Solve this math problem step by step. Show clear reasoning.',
    #         'thinking_tokens': 800
    #     },
    #     'high': {
    #         'instruction': 'Solve this math problem with detailed reasoning. Consider multiple approaches and verify your answer.',
    #         'thinking_tokens': 1600
    #     },
    #     'very_high': {
    #         'instruction': 'Solve this math problem with comprehensive analysis. Explore multiple solution paths, verify results, and explain your reasoning thoroughly.',
    #         'thinking_tokens': 3200
    #     }
    # }
    effort_configs = {
    'low': {
        'instruction': (
            "Solve the math problem. Provide a short chain of thought: "
            "explain only the essential steps needed to reach the answer."
        ),
        'thinking_tokens': 300
    },
    'medium': {
        'instruction': (
            "Solve the math problem step by step. Break the problem into logical sub-steps, "
            "explain intermediate calculations, and ensure the reasoning is coherent. "
            "Avoid unnecessary elaboration."
        ),
        'thinking_tokens': 700
    },
    'high': {
        'instruction': (
            "Solve the math problem with detailed reasoning. Explicitly identify assumptions, "
            "justify each transformation, and check for potential pitfalls or edge cases. "
            "Include a verification step to confirm the final answer is correct."
        ),
        'thinking_tokens': 1400
    },
    'very_high': {
        'instruction': (
            "Solve the math problem with comprehensive and rigorous mathematical reasoning. "
            "Decompose the problem into precise subproblems, explore alternative solution paths "
            "when applicable, justify every step with clear mathematical arguments, "
            "perform sanity checks and error analysis, and verify the final result through "
            "multiple independent methods."
        ),
        'thinking_tokens': 2500
    }
    }

    
    config = effort_configs.get(thinking_effort, effort_configs['medium'])
    
    return f"""You are an expert mathematics problem solver. {config['instruction']}

Present your solution in the following format:
1. Use <think></think> tags for your detailed reasoning process
2. After your thinking, write out the solution steps clearly
3. End with: Therefore, the answer is \\boxed{{your_answer}}

The thinking process should be approximately {config['thinking_tokens']} tokens."""


def call_api_single_attempt(
    question: str,
    teacher_model: str,
    thinking_effort: str,
    max_tokens: int,
    temperature: float,
    dataset_name: str
) -> Optional[Dict]:
    """Call API once and return result"""
    try:
        system_prompt = get_system_prompt(thinking_effort)
        
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
        
        # Extract thinking process and solution
        thinking_process = ""
        solution = raw_output
        
        if "<think>" in raw_output and "</think>" in raw_output:
            start_idx = raw_output.find("<think>") + 7
            end_idx = raw_output.find("</think>")
            thinking_process = raw_output[start_idx:end_idx].strip()
            solution = raw_output[end_idx + 8:].strip()
        
        # Extract answer
        predicted_answer = extract_answer(raw_output)
        
        return {
            'raw_output': raw_output,
            'thinking_process': thinking_process,
            'solution': solution,
            'predicted_answer': predicted_answer,
            'tokens_used': tokens_used,
            'finish_reason': finish_reason,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'raw_output': '',
            'thinking_process': '',
            'solution': '',
            'predicted_answer': None,
            'tokens_used': 0,
            'finish_reason': 'error',
            'status': f'error: {str(e)}'
        }


def select_thinking_effort(efforts: List[str], ratios: List[float]) -> str:
    """Randomly select thinking effort based on ratios"""
    return random.choices(efforts, weights=ratios, k=1)[0]


def process_sample_with_model(
    sample_data: Tuple
) -> Dict:
    """Process a single sample with one model (with retry)"""
    (index, question, ground_truth, model_name, round_num, 
     max_attempts, thinking_efforts, effort_ratios, max_tokens, temperatures, dataset_name) = sample_data
    
    # Select thinking effort for this sample
    thinking_effort = select_thinking_effort(thinking_efforts, effort_ratios)
    
    attempts = []
    final_correct = False
    final_result = None
    
    for attempt in range(max_attempts):
        temp = temperatures[attempt % len(temperatures)]
        
        result = call_api_single_attempt(
            question, model_name, thinking_effort, 
            max_tokens, temp, dataset_name
        )
        
        if result is None:
            continue
        
        # Check correctness
        predicted = result['predicted_answer']
        is_correct = False
        if predicted is not None:
            is_correct = check_answer(predicted, ground_truth)
        
        attempts.append({
            'attempt': attempt + 1,
            'temperature': temp,
            'predicted_answer': predicted,
            'correct': is_correct,
            'tokens': result['tokens_used'],
            'status': result['status']
        })
        
        if is_correct:
            final_correct = True
            final_result = result
            break
    
    if final_result is None:
        # Use default empty result
        final_result = {
            'raw_output': '',
            'thinking_process': '',
            'solution': '',
            'predicted_answer': None,
            'tokens_used': 0,
            'finish_reason': 'no_attempts',
            'status': 'failed'
        }
    
    return {
        'index': index,
        'question': question,
        'ground_truth': ground_truth,
        'predicted_answer': final_result['predicted_answer'] if final_result else None,
        'thinking_process': final_result['thinking_process'] if final_result else '',
        'solution': final_result['solution'] if final_result else '',
        'raw_output': final_result['raw_output'] if final_result else '',
        'correct': final_correct,
        'teacher_model': model_name,
        'thinking_effort': thinking_effort,
        'round': round_num,
        'attempt_count': len(attempts),
        'attempts_history': attempts,
        'tokens_used': sum(a['tokens'] for a in attempts),
        'status': 'success' if final_correct else 'failed'
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
    dataset_name: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run one round with multiple models in parallel.
    Returns: (successful_results, failed_samples)
    """
    print(f"\n{'='*80}")
    print(f"ROUND {round_num}: {len(samples)} samples × {len(models)} models")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models)}")
    print(f"Max Attempts: {max_attempts}")
    
    # Display thinking effort distribution
    effort_display = []
    for effort, ratio in zip(thinking_efforts, effort_ratios):
        effort_display.append(f"{effort}({ratio*100:.0f}%)")
    print(f"Thinking Effort: {', '.join(effort_display)}")
    print(f"{'='*80}\n")
    
    if not samples:
        return [], []
    
    # Prepare tasks: for each sample, test all models
    tasks = []
    for sample in samples:
        for model in models:
            tasks.append((
                sample['index'],
                sample['question'],
                sample['ground_truth'],
                model,
                round_num,
                max_attempts,
                thinking_efforts,
                effort_ratios,
                max_tokens,
                temperatures,
                dataset_name
            ))
    
    # Process all tasks
    print(f"Processing {len(tasks)} tasks ({len(samples)} samples × {len(models)} models)...")
    start_time = time.time()
    results = []
    
    if workers > 1:
        with Pool(processes=workers) as pool:
            with tqdm(total=len(tasks), desc=f"Round {round_num}", unit="task") as pbar:
                for result in pool.imap_unordered(process_sample_with_model, tasks):
                    results.append(result)
                    pbar.update(1)
    else:
        for task in tqdm(tasks, desc=f"Round {round_num}", unit="task"):
            result = process_sample_with_model(task)
            results.append(result)
    
    elapsed = time.time() - start_time
    
    # Group results by sample index
    sample_results = {}
    for r in results:
        idx = r['index']
        if idx not in sample_results:
            sample_results[idx] = []
        sample_results[idx].append(r)
    
    # Determine which samples succeeded and which failed
    successful = []
    failed_indices = set()
    
    for idx, sample_results_list in sample_results.items():
        # Check if any model succeeded
        correct_results = [r for r in sample_results_list if r['correct']]
        
        if correct_results:
            # Keep ALL correct results from this round
            successful.extend(correct_results)
        else:
            # All models failed
            failed_indices.add(idx)
    
    # Prepare failed samples for next round
    failed_samples = [s for s in samples if s['index'] in failed_indices]
    
    # Statistics
    total_correct = len(set(r['index'] for r in successful))
    total_failed = len(failed_samples)
    
    print(f"\n✓ Round {round_num} Complete:")
    print(f"  Samples Processed: {len(samples)}")
    print(f"  Samples Solved: {total_correct} ({total_correct/len(samples)*100:.2f}%)")
    print(f"  Samples Failed: {total_failed} ({total_failed/len(samples)*100:.2f}%)")
    print(f"  Total Solutions: {len(successful)} (including multiple correct per sample)")
    print(f"  Time: {elapsed:.2f}s")
    
    # Per-model statistics
    for model in models:
        model_results = [r for r in results if r['teacher_model'] == model]
        model_correct = len([r for r in model_results if r['correct']])
        print(f"    {model}: {model_correct}/{len(model_results)} correct")
    
    return successful, failed_samples


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
    output_dir: Path
) -> Dict:
    """Process all datasets through 2-round pipeline"""
    
    all_results = []
    dataset_stats = {}
    
    for dataset_full in dataset_names:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset_full}")
        print(f"{'#'*80}\n")
        
        # Parse dataset name and split
        dataset_name = dataset_full
        split_name = 'train'
        if '-' in dataset_full:
            parts = dataset_full.split('-', 1)
            dataset_name, split_name = parts[0], parts[1]
        
        # Load dataset
        base_path = Path(__file__).parent.parent
        dataset = load_dataset_for_eval(dataset_name, str(base_path))
        
        # Determine samples to process
        total = len(dataset)
        remaining = total - start_idx
        num_samples = remaining if count == 0 else min(count, remaining)
        
        print(f"Dataset: {dataset_name} (split: {split_name})")
        print(f"Total samples: {total}")
        print(f"Processing: {num_samples} samples (index {start_idx} to {start_idx + num_samples - 1})\n")
        
        # Prepare initial samples
        samples = []
        for i in range(start_idx, start_idx + num_samples):
            example = dataset[i]
            question = example.get('question', example.get('problem', ''))
            answer = example.get('answer', example.get('solution', ''))
            
            samples.append({
                'index': i,
                'question': question,
                'ground_truth': answer,
                'dataset': dataset_full
            })
        
        # ====================================================================
        # ROUND 1: Teacher Models
        # ====================================================================
        
        round1_results, round1_failed = run_round_multi_model(
            samples, teacher_models, 1, round1_attempts,
            round1_efforts, round1_ratios, max_tokens, round1_temps, workers, dataset_name
        )
        
        # ====================================================================
        # ROUND 2: Expert Models (only for failed samples)
        # ====================================================================
        
        round2_results = []
        if round1_failed and expert_models:
            round2_results, _ = run_round_multi_model(
                round1_failed, expert_models, 2, round2_attempts,
                round2_efforts, round2_ratios, max_tokens, round2_temps, workers, dataset_name
            )
        
        # ====================================================================
        # Combine Results
        # ====================================================================
        
        all_round_results = round1_results + round2_results
        
        # Add dataset field
        for r in all_round_results:
            r['dataset'] = dataset_full
        
        all_results.extend(all_round_results)
        
        # Statistics for this dataset
        unique_solved = set(r['index'] for r in all_round_results)
        r1_solved = set(r['index'] for r in round1_results)
        r2_solved = set(r['index'] for r in round2_results)
        
        dataset_stats[dataset_full] = {
            'total_samples': num_samples,
            'solved_total': len(unique_solved),
            'solved_round1': len(r1_solved),
            'solved_round2': len(r2_solved),
            'unsolved': num_samples - len(unique_solved),
            'success_rate': len(unique_solved) / num_samples * 100,
            'solutions_count': len(all_round_results)  # May be > solved if multiple models
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
        'results': all_results,
        'dataset_stats': dataset_stats
    }


def save_final_results(
    results: List[Dict],
    dataset_stats: Dict,
    output_dir: Path,
    teacher_models: List[str],
    expert_models: List[str],
    config: Dict
):
    """Save final results in clean format"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by dataset and index
    results_sorted = sorted(results, key=lambda x: (x['dataset'], x['index']))
    
    # Generate output filename
    dataset_str = '_'.join([d.replace('-', '_') for d in config['datasets']])
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base_name = f"cot_data_{dataset_str}_{timestamp}"
    
    # ========================================================================
    # Save JSON
    # ========================================================================
    json_file = output_dir / f"{base_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved JSON: {json_file}")
    print(f"  Total solutions: {len(results_sorted)}")
    
    # ========================================================================
    # Save CSV
    # ========================================================================
    csv_file = output_dir / f"{base_name}.csv"
    
    # Prepare dataframe
    df_data = []
    for r in results_sorted:
        df_data.append({
            'index': r['index'],
            'dataset': r['dataset'],
            'question': r['question'],
            'ground_truth': r['ground_truth'],
            'predicted_answer': r['predicted_answer'],
            'correct': r['correct'],
            'thinking_process': r['thinking_process'],
            'solution': r['solution'],
            'teacher_model': r['teacher_model'],
            'round': r['round'],
            'attempt_count': r['attempt_count'],
            'tokens_used': r['tokens_used'],
            'status': r['status']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"✓ Saved CSV: {csv_file}")
    
    # ========================================================================
    # Save Statistics
    # ========================================================================
    stats_file = output_dir / f"{base_name}_statistics.txt"
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CoT Data Generation Statistics\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration:\n")
        f.write("-"*80 + "\n")
        f.write(f"Teacher Models: {', '.join(teacher_models)}\n")
        f.write(f"Expert Models: {', '.join(expert_models)}\n")
        f.write(f"Round 1 Attempts: {config['round1_attempts']}\n")
        f.write(f"Round 2 Attempts: {config['round2_attempts']}\n")
        f.write(f"Workers: {config['workers']}\n\n")
        
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
            model_results = [r for r in results_sorted if r['teacher_model'] == model]
            if model_results:
                correct_count = len([r for r in model_results if r['correct']])
                f.write(f"\n{model}:\n")
                f.write(f"  Solutions: {len(model_results)}\n")
                f.write(f"  Correct: {correct_count}\n")
                f.write(f"  Unique Samples: {len(set(r['index'] for r in model_results))}\n")
                
                # Per round
                r1 = [r for r in model_results if r['round'] == 1]
                r2 = [r for r in model_results if r['round'] == 2]
                if r1:
                    f.write(f"  Round 1: {len([r for r in r1 if r['correct']])}/{len(r1)} correct\n")
                if r2:
                    f.write(f"  Round 2: {len([r for r in r2 if r['correct']])}/{len(r2)} correct\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved Statistics: {stats_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Optimized End-to-End CoT Data Generation with Multi-Model Support'
    )
    
    # Model configuration
    parser.add_argument('--teacher', type=str, required=True,
                       help='Comma-separated list of teacher models')
    parser.add_argument('--expert', type=str, default='',
                       help='Comma-separated list of expert models (optional)')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True,
                       help='Comma-separated list of datasets (e.g., gsm8k-train,math-train)')
    parser.add_argument('--count', type=int, default=0,
                       help='Number of samples per dataset (0 = all)')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting index')
    
    # Round 1 configuration
    parser.add_argument('--round1-attempts', type=int, default=3,
                       help='Max attempts for Round 1')
    parser.add_argument('--round1-efforts', type=str, default='medium',
                       help='Comma-separated thinking efforts for Round 1 (e.g., medium,high)')
    parser.add_argument('--round1-effort-ratios', type=str, default='',
                       help='Comma-separated ratios for efforts (e.g., 0.7,0.3). Default: equal distribution')
    parser.add_argument('--round1-temps', type=str, default='0.7,0.5,0.9',
                       help='Comma-separated temperatures for Round 1')
    
    # Round 2 configuration
    parser.add_argument('--round2-attempts', type=int, default=5,
                       help='Max attempts for Round 2')
    parser.add_argument('--round2-efforts', type=str, default='high',
                       help='Comma-separated thinking efforts for Round 2 (e.g., high,very_high)')
    parser.add_argument('--round2-effort-ratios', type=str, default='',
                       help='Comma-separated ratios for efforts (e.g., 0.6,0.4). Default: equal distribution')
    parser.add_argument('--round2-temps', type=str, default='0.7,0.5,0.9,0.3,1.0',
                       help='Comma-separated temperatures for Round 2')
    
    # General configuration
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--api-key', type=str, required=True)
    parser.add_argument('--api-base', type=str, default='https://api.aimlapi.com/v1')
    parser.add_argument('--output-dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # Parse lists
    teacher_models = [m.strip() for m in args.teacher.split(',') if m.strip()]
    expert_models = [m.strip() for m in args.expert.split(',') if m.strip()] if args.expert else []
    dataset_names = [d.strip() for d in args.dataset.split(',') if d.strip()]
    round1_temps = [float(t) for t in args.round1_temps.split(',')]
    round2_temps = [float(t) for t in args.round2_temps.split(',')]
    
    # Parse thinking efforts and ratios
    round1_efforts = [e.strip() for e in args.round1_efforts.split(',') if e.strip()]
    if args.round1_effort_ratios:
        round1_ratios = [float(r) for r in args.round1_effort_ratios.split(',')]
    else:
        round1_ratios = [1.0 / len(round1_efforts)] * len(round1_efforts)
    
    # Validate ratios
    if len(round1_efforts) != len(round1_ratios):
        print(f"Error: Round 1 efforts ({len(round1_efforts)}) and ratios ({len(round1_ratios)}) must have same length")
        sys.exit(1)
    if abs(sum(round1_ratios) - 1.0) > 0.01:
        print(f"Warning: Round 1 ratios sum to {sum(round1_ratios):.2f}, normalizing to 1.0")
        total = sum(round1_ratios)
        round1_ratios = [r/total for r in round1_ratios]
    
    round2_efforts = [e.strip() for e in args.round2_efforts.split(',') if e.strip()]
    if args.round2_effort_ratios:
        round2_ratios = [float(r) for r in args.round2_effort_ratios.split(',')]
    else:
        round2_ratios = [1.0 / len(round2_efforts)] * len(round2_efforts)
    
    if len(round2_efforts) != len(round2_ratios):
        print(f"Error: Round 2 efforts ({len(round2_efforts)}) and ratios ({len(round2_ratios)}) must have same length")
        sys.exit(1)
    if abs(sum(round2_ratios) - 1.0) > 0.01:
        print(f"Warning: Round 2 ratios sum to {sum(round2_ratios):.2f}, normalizing to 1.0")
        total = sum(round2_ratios)
        round2_ratios = [r/total for r in round2_ratios]
    
    # Initialize API
    initialize_api(args.api_key, args.api_base)
    
    # Print configuration
    print("\n" + "="*80)
    print("OPTIMIZED CoT DATA GENERATION - Multi-Model Multi-Dataset")
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
    
    # Display thinking effort distribution
    print(f"\nRound 1: {args.round1_attempts} attempts")
    print("  Thinking Efforts:")
    for effort, ratio in zip(round1_efforts, round1_ratios):
        print(f"    - {effort}: {ratio*100:.1f}%")
    
    print(f"\nRound 2: {args.round2_attempts} attempts")
    print("  Thinking Efforts:")
    for effort, ratio in zip(round2_efforts, round2_ratios):
        print(f"    - {effort}: {ratio*100:.1f}%")
    
    print(f"\nWorkers: {args.workers}")
    print("="*80 + "\n")
    
    # Process all datasets
    output_dir = Path(args.output_dir)
    
    config = {
        'datasets': dataset_names,
        'round1_attempts': args.round1_attempts,
        'round2_attempts': args.round2_attempts,
        'workers': args.workers
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
        output_dir=output_dir
    )
    
    # Save final results
    save_final_results(
        result['results'],
        result['dataset_stats'],
        output_dir,
        teacher_models,
        expert_models,
        config
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
