#!/usr/bin/env python3
"""
Comparison Test Script: Baseline vs Agents
在不同GPU上并行运行baseline和各个agent，对比第一轮生成结果

Usage:
    python scripts/compare_baseline_agents.py --count 50 --dataset gsm8k
"""

import sys
import os
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='Compare baseline vs agents')
    parser.add_argument('--model', type=str, default='Qwen2.5-Math-1.5B', help='Model name')
    parser.add_argument('--dataset', type=str, default='gsm8k', help='Dataset name')
    parser.add_argument('--count', type=int, default=50, help='Number of samples')
    parser.add_argument('--gpus', type=str, default='0,1', help='GPU IDs to use (comma-separated)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed output')
    return parser.parse_args()


def run_baseline_on_gpu(gpu_id: int, model_name: str, dataset_name: str, 
                         samples: List[Dict], results_queue: mp.Queue):
    """Run baseline evaluation on specified GPU"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from models.inference import load_model, generate_response
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from agent.unified_config import FIRST_ROUND_SOLVER_CONFIG
    
    print(f"[GPU {gpu_id}] Loading model for BASELINE...")
    model, tokenizer = load_model(model_name, PROJECT_ROOT)
    
    results = []
    correct = 0
    
    for i, sample in enumerate(samples):
        question = sample['question']
        ground_truth = sample['ground_truth']
        
        # Use unified config (same as agents' first round)
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
            response = f"Error: {e}"
        
        if is_correct:
            correct += 1
        
        results.append({
            'index': sample['index'],
            'question': question[:100],
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': is_correct,
            'response': response[:200]
        })
        
        print(f"[BASELINE GPU {gpu_id}] {i+1}/{len(samples)} | Acc: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")
    
    results_queue.put({
        'agent': 'baseline',
        'gpu': gpu_id,
        'results': results,
        'correct': correct,
        'total': len(samples),
        'accuracy': correct / len(samples) if samples else 0
    })
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


def run_agent_on_gpu(gpu_id: int, agent_name: str, model_name: str, dataset_name: str,
                     samples: List[Dict], results_queue: mp.Queue):
    """Run agent evaluation on specified GPU (only first round answer)"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from models.inference import load_model
    from utils.prompt_utils import extract_answer, check_answer
    
    print(f"[GPU {gpu_id}] Loading model for {agent_name}...")
    model, tokenizer = load_model(model_name, PROJECT_ROOT)
    
    # Import agent workflow
    workflow_func = get_agent_workflow(agent_name)
    if workflow_func is None:
        print(f"[ERROR] Unknown agent: {agent_name}")
        results_queue.put({
            'agent': agent_name,
            'gpu': gpu_id,
            'results': [],
            'correct': 0,
            'total': len(samples),
            'accuracy': 0,
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
            # Run agent workflow
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
            case_type = f'ERROR: {str(e)[:50]}'
        
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
            'question': question[:100],
            'ground_truth': ground_truth,
            'first_answer': first_answer,
            'predicted': predicted,
            'first_correct': is_first_correct,
            'final_correct': is_correct,
            'case_type': case_type
        })
        
        print(f"[{agent_name} GPU {gpu_id}] {i+1}/{len(samples)} | "
              f"1st: {first_correct}/{i+1} ({100*first_correct/(i+1):.1f}%) | "
              f"Final: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")
    
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
    
    # Cleanup
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
    print("BASELINE VS AGENTS COMPARISON TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Count: {args.count}")
    print(f"GPUs: {args.gpus}")
    print("=" * 80)
    
    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    if len(gpu_ids) < 2:
        print("WARNING: Only 1 GPU specified. Running sequentially.")
        gpu_ids = [gpu_ids[0], gpu_ids[0]]
    
    # Load samples
    print("\nLoading dataset...")
    samples = load_samples(args.dataset, args.count)
    print(f"Loaded {len(samples)} samples\n")
    
    # Setup results directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    results_dir = PROJECT_ROOT / 'results' / f'comparison_{args.model}_{args.dataset}_{args.count}_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Results queue
    results_queue = mp.Queue()
    
    # List of agents to test
    agents_to_test = [
        'solver_checker_stateless',
        'solver_checker_chat',
        'majority_vote',
        'plan_and_reflection',
        'agent_with_python_tools',
    ]
    
    all_results = {}
    
    # Run baseline first (on GPU 0)
    print("\n" + "=" * 80)
    print("PHASE 1: Running BASELINE")
    print("=" * 80)
    
    baseline_proc = mp.Process(
        target=run_baseline_on_gpu,
        args=(gpu_ids[0], args.model, args.dataset, samples, results_queue)
    )
    baseline_proc.start()
    baseline_proc.join()
    
    # Get baseline results
    baseline_result = results_queue.get()
    all_results['baseline'] = baseline_result
    print(f"\n[BASELINE COMPLETE] Accuracy: {baseline_result['accuracy']*100:.2f}%")
    
    # Run agents sequentially (each needs full GPU memory)
    print("\n" + "=" * 80)
    print("PHASE 2: Running AGENTS")
    print("=" * 80)
    
    for agent_name in agents_to_test:
        print(f"\n--- Running {agent_name} ---")
        
        agent_proc = mp.Process(
            target=run_agent_on_gpu,
            args=(gpu_ids[1 % len(gpu_ids)], agent_name, args.model, args.dataset, samples, results_queue)
        )
        agent_proc.start()
        agent_proc.join()
        
        agent_result = results_queue.get()
        all_results[agent_name] = agent_result
        
        print(f"\n[{agent_name} COMPLETE] "
              f"First: {agent_result.get('first_accuracy', 0)*100:.2f}% | "
              f"Final: {agent_result['accuracy']*100:.2f}%")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for name, result in all_results.items():
        row = {
            'Agent': name,
            'Total': result['total'],
            'Correct': result['correct'],
            'Accuracy': f"{result['accuracy']*100:.2f}%",
        }
        if name != 'baseline':
            row['First_Correct'] = result.get('first_correct', 'N/A')
            row['First_Accuracy'] = f"{result.get('first_accuracy', 0)*100:.2f}%"
            row['Improved'] = result.get('improved', 0)
            row['Degraded'] = result.get('degraded', 0)
        else:
            row['First_Correct'] = result['correct']
            row['First_Accuracy'] = f"{result['accuracy']*100:.2f}%"
            row['Improved'] = 0
            row['Degraded'] = 0
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv(results_dir / 'summary.csv', index=False)
    
    with open(results_dir / 'detailed_results.json', 'w') as f:
        # Convert results to serializable format
        serializable_results = {}
        for k, v in all_results.items():
            serializable_results[k] = {
                'agent': v['agent'],
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': v['accuracy'],
                'first_correct': v.get('first_correct'),
                'first_accuracy': v.get('first_accuracy'),
                'improved': v.get('improved'),
                'degraded': v.get('degraded'),
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Key metrics comparison
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    baseline_acc = all_results['baseline']['accuracy']
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
    print()
    
    for agent_name in agents_to_test:
        if agent_name in all_results:
            result = all_results[agent_name]
            first_acc = result.get('first_accuracy', 0)
            final_acc = result['accuracy']
            improved = result.get('improved', 0)
            degraded = result.get('degraded', 0)
            
            # Check if first answer matches baseline
            first_match = abs(first_acc - baseline_acc) < 0.02  # 2% tolerance
            
            print(f"{agent_name}:")
            print(f"  First-try Accuracy: {first_acc*100:.2f}% {'(matches baseline)' if first_match else '(DIFFERS from baseline!)'}")
            print(f"  Final Accuracy: {final_acc*100:.2f}%")
            print(f"  Improvement: +{improved} | Degradation: -{degraded}")
            print(f"  Net Change: {(final_acc - first_acc)*100:+.2f}%")
            print()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()

