#!/usr/bin/env python3
"""
Parallel Comparison Test: Baseline vs Agents (8 GPU Version)
充分利用8个GPU并行运行baseline和所有agents

Usage:
    python scripts/compare_baseline_agents_parallel.py --count 50 --dataset gsm8k
"""

import sys
import os
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='Compare baseline vs agents (parallel)')
    parser.add_argument('--model', type=str, default='Qwen2.5-Math-1.5B', help='Model name')
    parser.add_argument('--dataset', type=str, default='gsm8k', help='Dataset name')
    parser.add_argument('--count', type=int, default=50, help='Number of samples')
    parser.add_argument('--detailed', action='store_true', help='Show detailed output')
    return parser.parse_args()


def run_baseline(gpu_id: int, model_name: str, dataset_name: str, 
                 samples: List[Dict], results_queue: mp.Queue):
    """Run baseline evaluation on specified GPU"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from models.inference import load_model, generate_response
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from agent.unified_config import FIRST_ROUND_SOLVER_CONFIG
    
    print(f"[GPU {gpu_id}] Starting BASELINE...")
    model, tokenizer = load_model(model_name, PROJECT_ROOT)
    
    results = []
    correct = 0
    
    for i, sample in enumerate(samples):
        question = sample['question']
        ground_truth = sample['ground_truth']
        
        prompt = format_prompt_standard(question, dataset_name)
        
        try:
            response = generate_response(
                model, tokenizer, prompt, "standard", False,
                temperature=FIRST_ROUND_SOLVER_CONFIG['temperature'],
                do_sample=FIRST_ROUND_SOLVER_CONFIG['do_sample'],
                top_p=FIRST_ROUND_SOLVER_CONFIG['top_p']
            )
            predicted = extract_answer(response)
            is_correct = check_answer(predicted, ground_truth) if predicted else False
        except Exception as e:
            predicted = None
            is_correct = False
        
        if is_correct:
            correct += 1
        
        results.append({
            'index': sample['index'],
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': is_correct,
        })
        
        if (i + 1) % 10 == 0:
            print(f"[BASELINE GPU {gpu_id}] {i+1}/{len(samples)} | Acc: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")
    
    print(f"[BASELINE GPU {gpu_id}] DONE | Final: {correct}/{len(samples)} ({100*correct/len(samples):.1f}%)")
    
    results_queue.put({
        'agent': 'baseline',
        'gpu': gpu_id,
        'results': results,
        'correct': correct,
        'total': len(samples),
        'accuracy': correct / len(samples) if samples else 0
    })
    
    del model
    torch.cuda.empty_cache()


def run_agent(gpu_id: int, agent_name: str, model_name: str, dataset_name: str,
              samples: List[Dict], results_queue: mp.Queue):
    """Run agent evaluation on specified GPU"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from models.inference import load_model
    from utils.prompt_utils import check_answer
    
    print(f"[GPU {gpu_id}] Starting {agent_name}...")
    model, tokenizer = load_model(model_name, PROJECT_ROOT)
    
    # Get workflow function
    workflow_func = get_agent_workflow(agent_name)
    if workflow_func is None:
        print(f"[ERROR GPU {gpu_id}] Unknown agent: {agent_name}")
        results_queue.put({
            'agent': agent_name,
            'gpu': gpu_id,
            'error': f'Unknown agent: {agent_name}'
        })
        return
    
    results = []
    correct = 0
    first_correct = 0
    improved = 0
    degraded = 0
    
    for i, sample in enumerate(samples):
        question = sample['question']
        ground_truth = sample['ground_truth']
        
        try:
            result = workflow_func(
                question=question,
                ground_truth=ground_truth,
                model=model,
                tokenizer=tokenizer,
                max_iterations=3,
                detailed=False,
                dataset_name=dataset_name
            )
            
            predicted = result.get('predicted_answer')
            first_answer = result.get('first_answer')
            is_correct = result.get('final_correct', False)
            is_first_correct = result.get('first_correct', False)
            case_type = result.get('case_type', 'UNKNOWN')
            
        except Exception as e:
            predicted = None
            first_answer = None
            is_correct = False
            is_first_correct = False
            case_type = f'ERROR'
        
        if is_correct:
            correct += 1
        if is_first_correct:
            first_correct += 1
        if case_type == 'IMPROVED':
            improved += 1
        elif case_type == 'DEGRADED':
            degraded += 1
        
        results.append({
            'index': sample['index'],
            'ground_truth': ground_truth,
            'first_answer': first_answer,
            'predicted': predicted,
            'first_correct': is_first_correct,
            'final_correct': is_correct,
            'case_type': case_type
        })
        
        if (i + 1) % 10 == 0:
            print(f"[{agent_name} GPU {gpu_id}] {i+1}/{len(samples)} | 1st: {first_correct}/{i+1} | Final: {correct}/{i+1}")
    
    print(f"[{agent_name} GPU {gpu_id}] DONE | 1st: {first_correct}/{len(samples)} | Final: {correct}/{len(samples)}")
    
    results_queue.put({
        'agent': agent_name,
        'gpu': gpu_id,
        'results': results,
        'correct': correct,
        'first_correct': first_correct,
        'improved': improved,
        'degraded': degraded,
        'total': len(samples),
        'accuracy': correct / len(samples) if samples else 0,
        'first_accuracy': first_correct / len(samples) if samples else 0
    })
    
    del model
    torch.cuda.empty_cache()


def get_agent_workflow(agent_name: str):
    """Get workflow function for specified agent"""
    if agent_name == 'solver_checker_stateless':
        from agent.solver_checker_stateless import run_solver_checker_workflow
        def wrapper(**kwargs):
            return run_solver_checker_workflow(
                solver_model=kwargs['model'],
                solver_tokenizer=kwargs['tokenizer'],
                checker_model=kwargs['model'],
                checker_tokenizer=kwargs['tokenizer'],
                question=kwargs['question'],
                ground_truth=kwargs['ground_truth'],
                max_iterations=kwargs.get('max_iterations', 3),
                detailed=kwargs.get('detailed', False),
                dataset_name=kwargs.get('dataset_name', '')
            )
        return wrapper
    elif agent_name == 'solver_checker_chat':
        from agent.solver_checker_chat import run_solver_checker_chat_workflow
        return run_solver_checker_chat_workflow
    elif agent_name == 'solver_checker_with_tools':
        from agent.solver_checker_with_tools import run_solver_checker_with_tools_workflow
        def wrapper(**kwargs):
            return run_solver_checker_with_tools_workflow(
                solver_model=kwargs['model'],
                solver_tokenizer=kwargs['tokenizer'],
                checker_model=kwargs['model'],
                checker_tokenizer=kwargs['tokenizer'],
                question=kwargs['question'],
                ground_truth=kwargs['ground_truth'],
                max_iterations=kwargs.get('max_iterations', 3),
                detailed=kwargs.get('detailed', False),
                dataset_name=kwargs.get('dataset_name', '')
            )
        return wrapper
    elif agent_name == 'majority_vote':
        from agent.majority_vote import run_majority_vote_workflow
        return run_majority_vote_workflow
    elif agent_name == 'plan_and_reflection':
        from agent.plan_and_reflection import run_plan_and_reflection_workflow
        return run_plan_and_reflection_workflow
    elif agent_name == 'agent_with_python_tools':
        from agent.agent_with_python_tools import run_agent_with_python_tools
        return run_agent_with_python_tools
    elif agent_name == 'solver_checker_summarizer':
        from agent.solver_checker_summarizer import run_solver_checker_summarizer_workflow
        def wrapper(**kwargs):
            return run_solver_checker_summarizer_workflow(
                solver_model=kwargs['model'],
                solver_tokenizer=kwargs['tokenizer'],
                checker_model=kwargs['model'],
                checker_tokenizer=kwargs['tokenizer'],
                question=kwargs['question'],
                ground_truth=kwargs['ground_truth'],
                max_iterations=kwargs.get('max_iterations', 3),
                detailed=kwargs.get('detailed', False),
                dataset_name=kwargs.get('dataset_name', '')
            )
        return wrapper
    else:
        return None


def load_samples(dataset_name: str, count: int) -> List[Dict]:
    """Load dataset samples"""
    from dataset.dataloader import load_dataset_for_eval
    from utils.prompt_utils import extract_question_and_answer
    
    data = load_dataset_for_eval(dataset_name, str(PROJECT_ROOT))
    
    samples = []
    for i in range(min(count, len(data))):
        example = data[i]
        question, ground_truth = extract_question_and_answer(example, dataset_name)
        samples.append({
            'index': i,
            'question': question,
            'ground_truth': ground_truth
        })
    
    return samples


def main():
    args = parse_args()
    
    print("=" * 80)
    print("PARALLEL COMPARISON: BASELINE VS AGENTS (8 GPUs)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Count: {args.count}")
    print("=" * 80)
    
    # Load samples
    print("\nLoading dataset...")
    samples = load_samples(args.dataset, args.count)
    print(f"Loaded {len(samples)} samples\n")
    
    # Setup results directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    results_dir = PROJECT_ROOT / 'results' / f'parallel_comparison_{args.model}_{args.dataset}_{args.count}_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Results queue
    results_queue = mp.Queue()
    
    # Define all tasks: (gpu_id, agent_name)
    # GPU 0: baseline
    # GPU 1-7: agents
    tasks = [
        (0, 'baseline'),
        (1, 'solver_checker_stateless'),
        (2, 'solver_checker_chat'),
        (3, 'majority_vote'),
        (4, 'plan_and_reflection'),
        (5, 'agent_with_python_tools'),
        (6, 'solver_checker_with_tools'),
        (7, 'solver_checker_summarizer'),
    ]
    
    print(f"\nStarting {len(tasks)} parallel processes on 8 GPUs...")
    print("-" * 80)
    for gpu_id, agent_name in tasks:
        print(f"  GPU {gpu_id}: {agent_name}")
    print("-" * 80)
    
    # Start all processes in parallel
    processes = []
    start_time = time.time()
    
    for gpu_id, agent_name in tasks:
        if agent_name == 'baseline':
            p = mp.Process(
                target=run_baseline,
                args=(gpu_id, args.model, args.dataset, samples, results_queue)
            )
        else:
            p = mp.Process(
                target=run_agent,
                args=(gpu_id, agent_name, args.model, args.dataset, samples, results_queue)
            )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Collect all results
    all_results = {}
    while not results_queue.empty():
        result = results_queue.get()
        all_results[result['agent']] = result
    
    # Generate summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total Time: {total_time:.1f}s\n")
    
    summary_data = []
    
    # Baseline first
    if 'baseline' in all_results:
        result = all_results['baseline']
        summary_data.append({
            'Agent': 'baseline',
            'GPU': result['gpu'],
            'Total': result['total'],
            'Correct': result['correct'],
            'Accuracy': f"{result['accuracy']*100:.2f}%",
            'First_Correct': result['correct'],
            'First_Accuracy': f"{result['accuracy']*100:.2f}%",
            'Improved': '-',
            'Degraded': '-'
        })
    
    # Then agents
    for agent_name in ['solver_checker_stateless', 'solver_checker_chat', 'majority_vote', 
                       'plan_and_reflection', 'agent_with_python_tools', 
                       'solver_checker_with_tools', 'solver_checker_summarizer']:
        if agent_name in all_results:
            result = all_results[agent_name]
            if 'error' in result:
                summary_data.append({
                    'Agent': agent_name,
                    'GPU': result['gpu'],
                    'Total': '-',
                    'Correct': '-',
                    'Accuracy': 'ERROR',
                    'First_Correct': '-',
                    'First_Accuracy': '-',
                    'Improved': '-',
                    'Degraded': '-'
                })
            else:
                summary_data.append({
                    'Agent': agent_name,
                    'GPU': result['gpu'],
                    'Total': result['total'],
                    'Correct': result['correct'],
                    'Accuracy': f"{result['accuracy']*100:.2f}%",
                    'First_Correct': result.get('first_correct', '-'),
                    'First_Accuracy': f"{result.get('first_accuracy', 0)*100:.2f}%",
                    'Improved': result.get('improved', 0),
                    'Degraded': result.get('degraded', 0)
                })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv(results_dir / 'summary.csv', index=False)
    
    with open(results_dir / 'detailed_results.json', 'w') as f:
        serializable_results = {}
        for k, v in all_results.items():
            serializable_results[k] = {
                'agent': v['agent'],
                'gpu': v['gpu'],
                'total': v.get('total'),
                'correct': v.get('correct'),
                'accuracy': v.get('accuracy'),
                'first_correct': v.get('first_correct'),
                'first_accuracy': v.get('first_accuracy'),
                'improved': v.get('improved'),
                'degraded': v.get('degraded'),
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    if 'baseline' in all_results:
        baseline_acc = all_results['baseline']['accuracy']
        print(f"Baseline Accuracy: {baseline_acc*100:.2f}%\n")
        
        for agent_name in ['solver_checker_stateless', 'solver_checker_chat', 'majority_vote',
                          'plan_and_reflection', 'agent_with_python_tools',
                          'solver_checker_with_tools', 'solver_checker_summarizer']:
            if agent_name in all_results and 'error' not in all_results[agent_name]:
                result = all_results[agent_name]
                first_acc = result.get('first_accuracy', 0)
                final_acc = result['accuracy']
                improved = result.get('improved', 0)
                degraded = result.get('degraded', 0)
                
                first_match = abs(first_acc - baseline_acc) < 0.02
                
                print(f"{agent_name}:")
                print(f"  First-try: {first_acc*100:.2f}% {'(=baseline)' if first_match else '(DIFFERS!)'}")
                print(f"  Final: {final_acc*100:.2f}%")
                print(f"  +{improved} improved / -{degraded} degraded")
                print(f"  Net: {(final_acc - first_acc)*100:+.2f}%")
                print()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()








