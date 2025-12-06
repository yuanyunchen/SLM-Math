"""
Collect Agent Trajectories for Fine-tuning

This script runs the interactive code agent on datasets and collects trajectories.
Trajectories are categorized for creating a fine-tuning dataset.

Categories:
1. ONE_SHOT_SUCCESS: Correct answer on first code execution
2. SELF_CORRECTED: Wrong at first, then corrected
3. MULTI_STEP_SUCCESS: Multiple code blocks, all correct
4. FAILED: Could not get correct answer (for analysis, not training)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm


def categorize_trajectory(result: Dict) -> str:
    """Categorize a trajectory based on execution pattern."""
    
    if not result['final_correct']:
        return "FAILED"
    
    iterations = result.get('iterations', [])
    if not iterations:
        return "UNKNOWN"
    
    first_iter = iterations[0]
    exec_results = first_iter.get('exec_results', [])
    
    if len(exec_results) == 0:
        # No code execution, direct answer
        return "NO_CODE_SUCCESS"
    
    elif len(exec_results) == 1:
        # Single code execution
        if exec_results[0]['success']:
            return "ONE_SHOT_SUCCESS"
        else:
            return "SELF_CORRECTED"  # Had error but final answer correct
    
    else:
        # Multiple code executions
        has_error = any(not r['success'] for r in exec_results)
        if has_error:
            return "SELF_CORRECTED"
        else:
            return "MULTI_STEP_SUCCESS"


def format_trajectory_for_sft(result: Dict, dataset_name: str = "") -> Dict:
    """
    Format a trajectory into SFT format.
    
    Returns dict with 'query' and 'response' fields.
    """
    question = result['question']
    
    # Build the response from iterations
    iterations = result.get('iterations', [])
    if not iterations:
        return None
    
    response = iterations[0].get('response', '')
    
    # Clean up response
    response = response.strip()
    
    return {
        'query': question,
        'response': response,
        'ground_truth': result['ground_truth'],
        'predicted_answer': result['predicted_answer'],
        'category': categorize_trajectory(result),
        'code_executions': result.get('total_code_executions', 0),
        'dataset': dataset_name
    }


def collect_trajectories(
    model_name: str,
    dataset_name: str,
    count: int = 100,
    output_dir: str = "data_collection/trajectories",
    detailed: bool = False
):
    """Run agent and collect trajectories."""
    
    from models.model_loader import load_model
    from evaluation.eval_pipeline import load_dataset_by_name
    from agent.solver_with_interactive_code import run_solver_with_interactive_code
    
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset_by_name(dataset_name)
    
    if count > 0:
        dataset = dataset[:count]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    trajectories = {
        "ONE_SHOT_SUCCESS": [],
        "SELF_CORRECTED": [],
        "MULTI_STEP_SUCCESS": [],
        "NO_CODE_SUCCESS": [],
        "FAILED": [],
        "UNKNOWN": []
    }
    
    print(f"\nCollecting trajectories from {len(dataset)} problems...")
    
    for i, item in enumerate(tqdm(dataset)):
        question = item['question']
        ground_truth = item['answer']
        
        try:
            result = run_solver_with_interactive_code(
                question=question,
                ground_truth=ground_truth,
                model=model,
                tokenizer=tokenizer,
                max_iterations=1,
                max_code_executions=5,
                detailed=detailed,
                dataset_name=dataset_name,
                share_variables=True
            )
            
            # Format and categorize
            formatted = format_trajectory_for_sft(result, dataset_name)
            if formatted:
                category = formatted['category']
                trajectories[category].append(formatted)
                
        except Exception as e:
            print(f"Error on problem {i}: {e}")
            continue
    
    # Save trajectories by category
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    stats = {}
    for category, items in trajectories.items():
        if items:
            filename = f"{model_name}_{dataset_name}_{category}_{timestamp}.json"
            filepath = output_path / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            stats[category] = len(items)
            print(f"Saved {len(items)} {category} trajectories to {filepath}")
    
    # Save summary
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "total_problems": len(dataset),
        "statistics": stats,
        "timestamp": timestamp
    }
    summary_path = output_path / f"summary_{model_name}_{dataset_name}_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nCollection Summary:")
    print(f"  Total: {len(dataset)}")
    for cat, count in stats.items():
        pct = count / len(dataset) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    return trajectories, stats


def create_sft_dataset(
    trajectory_dir: str,
    output_file: str,
    success_ratio: float = 0.7,
    correction_ratio: float = 0.3
):
    """
    Create balanced SFT dataset from collected trajectories.
    
    Args:
        trajectory_dir: Directory with trajectory JSON files
        output_file: Output file for combined dataset
        success_ratio: Ratio of one-shot success examples
        correction_ratio: Ratio of self-corrected examples
    """
    import random
    
    traj_path = Path(trajectory_dir)
    
    all_success = []
    all_corrected = []
    all_multistep = []
    
    for f in traj_path.glob("*.json"):
        if "summary" in f.name:
            continue
        
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        if "ONE_SHOT_SUCCESS" in f.name:
            all_success.extend(data)
        elif "SELF_CORRECTED" in f.name:
            all_corrected.extend(data)
        elif "MULTI_STEP_SUCCESS" in f.name:
            all_multistep.extend(data)
    
    print(f"Loaded trajectories:")
    print(f"  One-shot success: {len(all_success)}")
    print(f"  Self-corrected: {len(all_corrected)}")
    print(f"  Multi-step: {len(all_multistep)}")
    
    # Combine multi-step with success
    all_success.extend(all_multistep)
    
    # Balance dataset
    total_target = len(all_success) + len(all_corrected)
    
    if len(all_corrected) > 0:
        # Calculate how many of each to include
        n_success = int(total_target * success_ratio)
        n_corrected = int(total_target * correction_ratio)
        
        # Sample
        random.shuffle(all_success)
        random.shuffle(all_corrected)
        
        final_data = all_success[:n_success] + all_corrected[:n_corrected]
    else:
        final_data = all_success
    
    random.shuffle(final_data)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nCreated SFT dataset: {output_path}")
    print(f"  Total samples: {len(final_data)}")
    
    return final_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect agent trajectories for fine-tuning")
    parser.add_argument("--model", type=str, default="Qwen2.5-Math-1.5B", help="Model name")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--count", type=int, default=100, help="Number of problems (0 for all)")
    parser.add_argument("--output-dir", type=str, default="data_collection/trajectories")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    collect_trajectories(
        model_name=args.model,
        dataset_name=args.dataset,
        count=args.count,
        output_dir=args.output_dir,
        detailed=args.detailed
    )







