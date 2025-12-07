#!/usr/bin/env python3
"""
Agent Evaluation Script for SLM-Math
评估solver-checker多智能体工作流
"""

import sys
import json
import time
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import Counter, defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.prompt_utils import extract_question_and_answer
from dataset.dataloader import load_dataset_for_eval
from models.inference import load_model, load_inference_engine_wrapper
from agent.milestone_agents.solver_checker_stateless import run_solver_checker_workflow
from agent.milestone_agents.solver_checker_chat import run_solver_checker_chat_workflow  # Optimized chat
from agent.milestone_agents.solver_checker_trivial_chat import run_solver_checker_chat_workflow as run_solver_checker_trivial_chat_workflow
from agent.milestone_agents.solver_checker_with_tools import run_solver_checker_with_tools_workflow
from agent.solver_checker_with_tools_v2 import run_solver_checker_with_tools_workflow_v2
from agent.milestone_agents.solver_checker_summarizer import run_solver_checker_summarizer_workflow
from agent.milestone_agents.solver_checker_summarizer_chat import run_solver_checker_summarizer_chat_workflow
from agent.agent_with_python_tools import run_agent_with_python_tools
from agent.agent_with_code_feedback import run_agent_with_code_feedback
from agent.agent_default_prompt_with_code import run_agent_default_prompt_with_code
from agent.agent_code_as_answer import run_agent_code_as_answer
from agent.milestone_agents.majority_vote import run_majority_vote_workflow
from agent.milestone_agents.plan_and_reflection import run_plan_and_reflection_workflow
from agent.solver_verifier import run_solver_verifier_workflow
from agent.solver_verifier_check import run_solver_verifier_check_workflow
from agent.solver_coder import run_solver_coder_workflow
from agent.solver_with_interactive_code import run_solver_with_interactive_code
from agent.solver_step_by_step_code import run_solver_step_by_step_code


def parse_args():
    parser = argparse.ArgumentParser(description='Agent evaluation script')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (LoRA adapter or fine-tuned model). Can be relative to project root or absolute path.')
    parser.add_argument('--checker_model', type=str, default=None, help='Checker model name (for solver_checker only, default: same as solver)')
    parser.add_argument('--checker_checkpoint', type=str, default=None, help='Path to checker checkpoint (optional, for solver_checker only)')
    parser.add_argument('--agent', type=str, required=True, choices=['solver_checker', 'solver_checker_chat', 'solver_checker_trivial_chat', 'solver_checker_with_tools', 'solver_checker_with_tools_v2', 'solver_checker_summarizer', 'solver_checker_summarizer_chat', 'solver_verifier', 'solver_verifier_check', 'solver_coder', 'solver_interactive_code', 'solver_step_by_step_code', 'agent_with_python_tools', 'agent_with_code_feedback', 'agent_default_prompt_with_code', 'agent_code_as_answer', 'majority_vote', 'plan_and_reflection'], help='Agent method')
    parser.add_argument('--round', type=str, required=True, help='Test round name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--split', type=str, default=None, help='Dataset split')
    parser.add_argument('--count', type=int, required=True, help='Number of test cases (0 = all)')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--max_iterations', type=int, default=5, help='Max iterations per problem (for solver_checker and plan_and_reflection)')
    parser.add_argument('--max_subproblems', type=int, default=5, help='Max sub-problems (for plan_and_reflection)')
    parser.add_argument('--enable_solver_tools', type=str, default='true', choices=['true', 'false'], help='Enable code execution for solver (solver_checker_with_tools)')
    parser.add_argument('--enable_checker_tools', type=str, default='true', choices=['true', 'false'], help='Enable code execution for checker (solver_checker_with_tools)')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs (for majority_vote)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for majority_vote')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p for majority_vote')
    parser.add_argument('--enable_code_checker', type=str, default='true', choices=['true', 'false'], help='Enable checker validation for solver_coder')
    parser.add_argument('--code_timeout', type=int, default=10, help='Code execution timeout in seconds (for solver_coder)')
    parser.add_argument('--detailed', type=str, default='false', choices=['true', 'false'], help='Detailed output')
    parser.add_argument('--resume', type=str, default=None, help='Resume from existing results')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for parallel sample processing (default: 1)')
    parser.add_argument('--inference_backend', type=str, default='transformers', choices=['transformers', 'vllm'], help='Inference backend (default: transformers)')
    parser.add_argument('--apply_chat_template', type=str, default='false', choices=['true', 'false'], help='Apply chat template to prompts (default: false). Enable for chat-tuned models.')
    return parser.parse_args()


def run_evaluation(args):
    """Main evaluation function"""
    detailed = args.detailed.lower() == 'true'
    use_chat_template = args.apply_chat_template.lower() == 'true'
    
    print(f"\n{'='*80}")
    print(f"Agent Evaluation - Solver-Checker Workflow")
    print(f"{'='*80}")
    
    # Parse dataset name
    dataset_name = args.dataset
    split_name = args.split
    if '-' in dataset_name:
        parts = dataset_name.split('-', 1)
        dataset_name, suffix = parts[0], parts[1]
        if split_name is None:
            split_name = suffix
    if split_name is None:
        split_name = 'test'
    
    print(f"Model: {args.model}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Agent Method: {args.agent}")
    if args.agent in ['solver_checker', 'solver_checker_chat', 'solver_checker_trivial_chat', 'solver_checker_with_tools', 'solver_checker_with_tools_v2', 'solver_checker_summarizer', 'solver_checker_summarizer_chat']:
        if args.agent == 'solver_checker' and args.checker_model:
            print(f"Checker Model: {args.checker_model}")
            if args.checker_checkpoint:
                print(f"Checker Checkpoint: {args.checker_checkpoint}")
        elif args.agent == 'solver_checker':
            print(f"Checker Model: {args.model} (same as solver)")
        elif args.agent == 'solver_checker_chat':
            print(f"Using shared model for both solver and checker (optimized chat mode)")
        elif args.agent == 'solver_checker_trivial_chat':
            print(f"Using shared model for both solver and checker (trivial chat mode)")
        elif args.agent in ['solver_checker_with_tools', 'solver_checker_with_tools_v2']:
            print(f"Checker Model: {args.checker_model if args.checker_model else args.model}")
            if args.checker_checkpoint:
                print(f"Checker Checkpoint: {args.checker_checkpoint}")
            print(f"Solver Tools: {args.enable_solver_tools}")
            print(f"Checker Tools: {args.enable_checker_tools}")
            if args.agent == 'solver_checker_with_tools_v2':
                print(f"Version: V2 (improved)")
        elif args.agent == 'solver_checker_summarizer':
            print(f"Checker Model: {args.checker_model if args.checker_model else args.model}")
            if args.checker_checkpoint:
                print(f"Checker Checkpoint: {args.checker_checkpoint}")
            print(f"Summarizer: Enabled (Stateless mode)")
        elif args.agent == 'solver_checker_summarizer_chat':
            print(f"Using shared model for solver, checker, and summarizer (Chat mode)")
        print(f"Max Iterations: {args.max_iterations}")
    elif args.agent == 'plan_and_reflection':
        print(f"Plan-and-Reflection: Multi-phase agent workflow")
        print(f"Max Iterations: {args.max_iterations}")
        print(f"Max Sub-problems: {args.max_subproblems}")
    elif args.agent == 'solver_verifier':
        print(f"Solver-Verifier: Code self-verification + forward verification")
        print(f"Max Iterations: {args.max_iterations}")
    elif args.agent == 'solver_verifier_check':
        print(f"Solver-Verifier (check): Code self-verification + boolean check verifier")
        print(f"Max Iterations: {args.max_iterations}")
    elif args.agent == 'solver_coder':
        print(f"Solver-Coder: Pure code-based math solving with debug iterations")
        print(f"Max Iterations: {args.max_iterations}")
        print(f"Enable Checker: {args.enable_code_checker}")
        print(f"Code Timeout: {args.code_timeout}s")
    elif args.agent == 'solver_interactive_code':
        print(f"Solver with Interactive Code: Multi-round code execution within one generation")
        print(f"Max Code Executions: 5")
        print(f"Share Variables: True")
    elif args.agent == 'solver_step_by_step_code':
        print(f"Solver with Step-by-Step Code: Break problem into steps, execute code at each step")
        print(f"Max Steps: 8")
    elif args.agent == 'majority_vote':
        print(f"Num Runs: {args.num_runs}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
    print(f"Round: {args.round}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Split: {split_name}")
    print(f"Count: {args.count}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Inference Backend: {args.inference_backend}")
    print(f"Apply Chat Template: {use_chat_template}")
    print(f"Detailed Output: {detailed}")
    print(f"{'='*80}\n")
    
    base_path = Path(__file__).parent.parent
    
    # Load models/inference engines
    print("Loading models...")
    checker_model = None
    checker_tokenizer = None
    solver_engine = None
    checker_engine = None
    use_batch_inference = args.batch_size > 1 or args.inference_backend == 'vllm'
    
    try:
        if use_batch_inference:
            # Use inference engine for batch processing
            (model, tokenizer), solver_engine = load_inference_engine_wrapper(
                args.model, base_path, backend=args.inference_backend, checkpoint_path=args.checkpoint
            )
            
            # Adjust use_batch_inference if engine is None (checkpoint fallback)
            if solver_engine is None and args.batch_size == 1:
                use_batch_inference = False
            
            if args.agent in ['solver_checker', 'solver_checker_with_tools', 'solver_checker_with_tools_v2', 'solver_checker_summarizer']:
                if args.checker_model and args.checker_model != args.model:
                    (checker_model, checker_tokenizer), checker_engine = load_inference_engine_wrapper(
                        args.checker_model, base_path, backend=args.inference_backend, checkpoint_path=args.checker_checkpoint
                    )
                else:
                    checker_model, checker_tokenizer = model, tokenizer
                    checker_engine = solver_engine
            elif args.agent in ['solver_checker_chat', 'solver_checker_trivial_chat', 'solver_checker_summarizer_chat']:
                checker_model, checker_tokenizer = model, tokenizer
                checker_engine = solver_engine
        else:
            # Use original load_model for backward compatibility
            model, tokenizer = load_model(args.model, base_path, checkpoint_path=args.checkpoint)
            if args.agent in ['solver_checker', 'solver_checker_with_tools', 'solver_checker_with_tools_v2', 'solver_checker_summarizer']:
                if args.checker_model and args.checker_model != args.model:
                    checker_model, checker_tokenizer = load_model(args.checker_model, base_path, checkpoint_path=args.checker_checkpoint)
                else:
                    checker_model, checker_tokenizer = model, tokenizer
            elif args.agent in ['solver_checker_chat', 'solver_checker_trivial_chat', 'solver_checker_summarizer_chat']:
                checker_model, checker_tokenizer = model, tokenizer
        print("Models loaded successfully.\n")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None
    
    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    try:
        test_data = load_dataset_for_eval(dataset_name, str(base_path), split_name=split_name)
        total_samples = len(test_data)
        print(f"Dataset loaded: {total_samples} total samples\n")
        
        start_index = max(0, args.start)
        if start_index >= total_samples:
            print(f"ERROR: start index {start_index} >= dataset size {total_samples}")
            return None
        
        remaining = total_samples - start_index
        if args.count == 0:
            num_samples = remaining
        else:
            num_samples = min(args.count, remaining)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None
    
    # Setup results directory
    resume_mode = args.resume is not None
    processed_ids = set()
    
    if resume_mode:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = base_path / args.resume
        if not resume_path.exists():
            print(f"ERROR: Resume directory not found: {resume_path}")
            return None
        results_dir = resume_path
        log_dir = results_dir / 'log'
        answers_dir = results_dir / 'answers'
        
        answer_file = answers_dir / f"{args.model}_{args.dataset}_{args.agent}_answers.json"
        if answer_file.exists():
            with open(answer_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_ids = {pred['question_id'] for pred in results.get('predictions', [])}
            print(f"Resuming: {len(processed_ids)} already processed")
            start_time = time.time() - results.get('total_time', 0)
        else:
            results = create_empty_results(args, num_samples, start_index)
            start_time = time.time()
    else:
        now = datetime.now()
        month_date = now.strftime("%m%d")
        minute_second = now.strftime("%H%M")
        base_dir_name = f"{args.round}_{args.model}_{args.dataset}_{num_samples}_{month_date}"
        results_dir = base_path / 'results' / base_dir_name
        if results_dir.exists():
            base_dir_name = f"{args.round}_{args.model}_{args.dataset}_{num_samples}_{month_date}_{minute_second}"
            results_dir = base_path / 'results' / base_dir_name
        
        log_dir = results_dir / 'log'
        answers_dir = results_dir / 'answers'
        log_dir.mkdir(parents=True, exist_ok=True)
        answers_dir.mkdir(parents=True, exist_ok=True)
        
        results = create_empty_results(args, num_samples, start_index)
        start_time = time.time()
    
    print(f"Results directory: {results_dir}\n")
    
    # Setup logging
    log_file = log_dir / f"{args.model}_{args.dataset}_{args.agent}.log"
    
    def log_and_print(message, to_console=True):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        if to_console:
            print(message)
    
    log_and_print("="*80)
    log_and_print(f"Starting agent evaluation - {num_samples} samples")
    log_and_print("="*80 + "\n")
    
    # Calculate remaining samples
    remaining_samples = [idx for idx in range(start_index, start_index + num_samples) if idx not in processed_ids]
    remaining_count = len(remaining_samples)
    
    if remaining_count == 0:
        print("All samples already processed!")
        return results
    
    # Initialize progress bar
    progress_bar = None
    if not detailed:
        progress_bar = tqdm(total=num_samples, initial=len(processed_ids), desc="Progress", unit="sample")
    
    # Statistics tracking
    stats = {
        'total': 0,
        'correct': 0,
        'first_try_correct': 0,
        'improved_cases': 0,
        'degraded_cases': 0,
        'failed_cases': 0,
        'other_cases': 0,
        'total_iterations': 0,
        'checker_verdicts': Counter(),
        'verdict_by_iteration': defaultdict(Counter),
        'false_positives': 0,  # Checker says CORRECT but actually wrong
        'false_negatives': 0,  # Checker says INCORRECT but actually correct
    }
    
    # Helper function to save results
    answer_file = answers_dir / f"{args.model}_{args.dataset}_{args.agent}_answers.json"
    
    def save_results():
        results["total_time"] = time.time() - start_time
        results["avg_time_per_sample"] = results["total_time"] / stats['total'] if stats['total'] > 0 else 0
        results["accuracy"] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        with open(answer_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Evaluation loop
    processed_count = 0
    for dataset_idx in remaining_samples:
        processed_count += 1
        example = test_data[dataset_idx]
        question, ground_truth = extract_question_and_answer(example, dataset_name)
        
        log_and_print("\n" + "="*80, to_console=detailed)
        log_and_print(f"[Sample {processed_count}/{remaining_count} | Index: {dataset_idx}]", to_console=detailed)
        log_and_print(f"Question: {question}", to_console=detailed)
        log_and_print(f"Ground Truth: {ground_truth}", to_console=detailed)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Full Question: {question}\n")
        
        try:
            sample_start_time = time.time()
            
            # Run agent workflow based on method
            if args.agent == 'solver_checker':
                workflow_result = run_solver_checker_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    solver_model=model,
                    solver_tokenizer=tokenizer,
                    checker_model=checker_model,
                    checker_tokenizer=checker_tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_checker_chat':
                # Use solver_engine if batch inference is enabled
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_solver_checker_chat_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_checker_trivial_chat':
                # Use solver_engine if batch inference is enabled
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_solver_checker_trivial_chat_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_checker_with_tools':
                enable_solver_tools = args.enable_solver_tools.lower() == 'true'
                enable_checker_tools = args.enable_checker_tools.lower() == 'true'
                workflow_result = run_solver_checker_with_tools_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    solver_model=model,
                    solver_tokenizer=tokenizer,
                    checker_model=checker_model,
                    checker_tokenizer=checker_tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    enable_solver_tools=enable_solver_tools,
                    enable_checker_tools=enable_checker_tools,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_checker_with_tools_v2':
                enable_solver_tools = args.enable_solver_tools.lower() == 'true'
                enable_checker_tools = args.enable_checker_tools.lower() == 'true'
                workflow_result = run_solver_checker_with_tools_workflow_v2(
                    question=question,
                    ground_truth=ground_truth,
                    solver_model=model,
                    solver_tokenizer=tokenizer,
                    checker_model=checker_model,
                    checker_tokenizer=checker_tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    enable_solver_tools=enable_solver_tools,
                    enable_checker_tools=enable_checker_tools,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_checker_summarizer':
                workflow_result = run_solver_checker_summarizer_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    solver_model=model,
                    solver_tokenizer=tokenizer,
                    checker_model=checker_model,
                    checker_tokenizer=checker_tokenizer,
                    summarizer_model=model,  # Use solver model for summarization
                    summarizer_tokenizer=tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_checker_summarizer_chat':
                # Use solver_engine if batch inference is enabled
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_solver_checker_summarizer_chat_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'plan_and_reflection':
                # Use solver_engine if batch inference is enabled
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_plan_and_reflection_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    max_iterations=args.max_iterations,
                    max_subproblems=args.max_subproblems,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_verifier':
                # Solver-Verifier with code self-verification
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_solver_verifier_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    solver_model=model_to_use,
                    solver_tokenizer=tokenizer,
                    verifier_model=model_to_use,
                    verifier_tokenizer=tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    enable_solver_tools=True,
                    consistency_threshold=2
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_verifier_check':
                # Solver-Verifier with check(candidate) boolean verification
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_solver_verifier_check_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    solver_model=model_to_use,
                    solver_tokenizer=tokenizer,
                    verifier_model=model_to_use,
                    verifier_tokenizer=tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    enable_solver_tools=True,
                    consistency_threshold=2
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_coder':
                # Solver-Coder: Pure code-based math solving
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                enable_checker = args.enable_code_checker.lower() == 'true'
                workflow_result = run_solver_coder_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    solver_model=model_to_use,
                    solver_tokenizer=tokenizer,
                    checker_model=model_to_use,
                    checker_tokenizer=tokenizer,
                    max_iterations=args.max_iterations,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    enable_checker=enable_checker,
                    code_timeout=args.code_timeout
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                case_type = workflow_result['case_type']
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_interactive_code':
                # Solver with Interactive Code: model can see code output and continue
                workflow_result = run_solver_with_interactive_code(
                    question=question,
                    ground_truth=ground_truth,
                    model=model,
                    tokenizer=tokenizer,
                    max_iterations=1,
                    max_code_executions=5,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    share_variables=True,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                first_try_correct = workflow_result.get('first_try_correct', final_correct)
                case_type = workflow_result.get('case_type', 'TP' if final_correct else 'FN')
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'solver_step_by_step_code':
                # Solver with Step-by-Step Code: break problem into steps
                workflow_result = run_solver_step_by_step_code(
                    question=question,
                    ground_truth=ground_truth,
                    model=model,
                    tokenizer=tokenizer,
                    max_iterations=1,
                    max_steps=8,
                    detailed=detailed,
                    dataset_name=dataset_name,
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = workflow_result['total_iterations']
                first_try_correct = workflow_result['first_try_correct']
                case_type = 'TP' if final_correct else 'FN'  # Simplified case type
                iterations = workflow_result.get('iterations', [])
                runs = None
                num_runs = None
            elif args.agent == 'agent_with_python_tools':
                # Use solver_engine if batch inference is enabled
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_agent_with_python_tools(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    enable_tools=True,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = 1  # Single-shot
                first_try_correct = final_correct
                case_type = workflow_result['case_type']
                iterations = None
                runs = None
                num_runs = None
            elif args.agent == 'agent_with_code_feedback':
                # Agent with code execution feedback
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_agent_with_code_feedback(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    enable_tools=True,
                    greedy=True,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = 1  # Single-shot with feedback
                first_try_correct = final_correct
                case_type = workflow_result['case_type']
                iterations = None
                runs = None
                num_runs = None
            elif args.agent == 'agent_default_prompt_with_code':
                # Agent with default prompt + code execution (no tool instruction)
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_agent_default_prompt_with_code(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    greedy=True,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = 1  # Single-shot with feedback
                first_try_correct = final_correct
                case_type = workflow_result['case_type']
                iterations = None
                runs = None
                num_runs = None
            elif args.agent == 'agent_code_as_answer':
                # Agent that uses code execution result directly as answer
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_agent_code_as_answer(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    enable_tools=True,
                    greedy=True,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = 1  # Single-shot
                first_try_correct = final_correct
                case_type = workflow_result['case_type']
                iterations = None
                runs = None
                num_runs = None
            elif args.agent == 'majority_vote':
                # Use solver_engine if batch inference is enabled
                model_to_use = solver_engine if (use_batch_inference and solver_engine) else model
                workflow_result = run_majority_vote_workflow(
                    question=question,
                    ground_truth=ground_truth,
                    model=model_to_use,
                    tokenizer=tokenizer,
                    num_runs=args.num_runs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    detailed=detailed,
                    dataset_name=dataset_name,
                    apply_chat_template=use_chat_template
                )
                predicted_answer = workflow_result['predicted_answer']
                final_correct = workflow_result['final_correct']
                total_iterations = None
                first_try_correct = final_correct
                case_type = workflow_result['case_type']
                iterations = None
                runs = workflow_result.get('runs', [])
                num_runs = workflow_result.get('num_runs', 0)
            
            sample_time = time.time() - sample_start_time
            
            # Update statistics
            stats['total'] += 1
            
            if args.agent in ['solver_checker', 'solver_checker_chat', 'solver_checker_summarizer', 
                              'solver_checker_summarizer_chat', 'solver_checker_trivial_chat', 'solver_checker_with_tools', 'solver_checker_with_tools_v2']:
                stats['total_iterations'] += total_iterations
                
                if workflow_result['first_correct'] and total_iterations == 1:
                    stats['first_try_correct'] += 1
                
                # Analyze checker verdicts
                for iter_data in iterations:
                    verdict = iter_data.get('checker_verdict')
                    if verdict:
                        iter_num = iter_data['iteration']
                        stats['checker_verdicts'][verdict] += 1
                        stats['verdict_by_iteration'][iter_num][verdict] += 1
                        
                        # Check for false positives/negatives
                        is_actually_correct = iter_data.get('is_actually_correct', False)
                        if verdict == "CORRECT" and not is_actually_correct:
                            stats['false_positives'] += 1
                        elif verdict == "INCORRECT" and is_actually_correct:
                            stats['false_negatives'] += 1
            elif args.agent == 'plan_and_reflection':
                stats['total_iterations'] += total_iterations
                if workflow_result['first_correct'] and total_iterations == 1:
                    stats['first_try_correct'] += 1
            elif args.agent in ['solver_verifier', 'solver_verifier_check']:
                stats['total_iterations'] += total_iterations
                if workflow_result['first_correct']:
                    stats['first_try_correct'] += 1
            elif args.agent == 'solver_coder':
                stats['total_iterations'] += total_iterations
                if workflow_result['first_correct']:
                    stats['first_try_correct'] += 1
            elif args.agent == 'solver_interactive_code':
                stats['total_iterations'] += workflow_result.get('total_code_executions', 0)
                if workflow_result['first_correct']:
                    stats['first_try_correct'] += 1
            elif args.agent == 'agent_with_code_feedback':
                if workflow_result['first_correct']:
                    stats['first_try_correct'] += 1
            elif args.agent == 'agent_default_prompt_with_code':
                if workflow_result['first_correct']:
                    stats['first_try_correct'] += 1
            elif args.agent == 'agent_code_as_answer':
                if workflow_result['first_correct']:
                    stats['first_try_correct'] += 1
            elif args.agent == 'majority_vote':
                if workflow_result['first_correct']:
                    stats['first_try_correct'] += 1
            
            if final_correct:
                stats['correct'] += 1
            
            if case_type == "IMPROVED" or case_type == "MAJORITY_IMPROVED":
                stats['improved_cases'] += 1
            elif case_type == "DEGRADED" or case_type == "MAJORITY_DEGRADED":
                stats['degraded_cases'] += 1
            elif case_type == "FAILED" or case_type == "MAJORITY_FAILED":
                stats['failed_cases'] += 1
            else:
                stats['other_cases'] += 1
            
            # Create prediction entry
            prediction_entry = {
                "question_id": dataset_idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "final_correct": final_correct,
                "case_type": case_type,
                "first_answer": workflow_result.get('first_answer', predicted_answer),
                "first_correct": workflow_result.get('first_correct', final_correct),
                "final_verdict": workflow_result.get('final_verdict', 'correct' if final_correct else 'incorrect'),
                "sample_time": sample_time
            }
            
            if args.agent in ['solver_checker', 'solver_checker_chat', 'solver_checker_trivial_chat', 'solver_checker_with_tools', 'solver_checker_with_tools_v2', 'solver_checker_summarizer', 'solver_checker_summarizer_chat']:
                prediction_entry.update({
                    "total_iterations": total_iterations,
                    "iterations": iterations,
                    "solver_answers": workflow_result.get('solver_answers', []),
                    "checker_verdicts": workflow_result.get('checker_verdicts', [])
                })
                if args.agent == 'solver_checker_chat':
                    prediction_entry['conversation_history'] = workflow_result.get('conversation_history', [])
                elif args.agent in ['solver_checker_with_tools', 'solver_checker_with_tools_v2']:
                    prediction_entry['tools_config'] = workflow_result.get('tools_config', {})
            elif args.agent == 'plan_and_reflection':
                prediction_entry.update({
                    "total_iterations": total_iterations,
                    "iterations": iterations,
                })
            elif args.agent in ['solver_verifier', 'solver_verifier_check']:
                prediction_entry.update({
                    "total_iterations": total_iterations,
                    "iterations": iterations,
                    "solver_answers": workflow_result.get('solver_answers', []),
                    "verifier_verdicts": workflow_result.get('verifier_verdicts', []),
                    "config": workflow_result.get('config', {})
                })
            elif args.agent == 'solver_coder':
                prediction_entry.update({
                    "total_iterations": total_iterations,
                    "iterations": iterations,
                    "all_codes": workflow_result.get('all_codes', []),
                    "all_answers": workflow_result.get('all_answers', []),
                    "error_count": workflow_result.get('error_count', 0),
                    "success_count": workflow_result.get('success_count', 0),
                    "config": workflow_result.get('config', {})
                })
            elif args.agent == 'solver_interactive_code':
                prediction_entry.update({
                    "total_iterations": total_iterations,
                    "iterations": iterations,
                    "total_code_executions": workflow_result.get('total_code_executions', 0),
                    "config": workflow_result.get('config', {})
                })
            elif args.agent == 'solver_step_by_step_code':
                prediction_entry.update({
                    "total_iterations": total_iterations,
                    "iterations": iterations,
                    "total_code_executions": workflow_result.get('total_code_executions', 0),
                    "execution_results": workflow_result.get('execution_results', [])
                })
            elif args.agent == 'agent_with_python_tools':
                prediction_entry.update({
                    "response": workflow_result.get('response', ''),
                    "code_executed": workflow_result.get('code_executed', False),
                    "exec_results": workflow_result.get('exec_results', []),
                    "num_code_blocks": workflow_result.get('num_code_blocks', 0),
                    "tools_config": workflow_result.get('tools_config', {})
                })
            elif args.agent == 'agent_with_code_feedback':
                prediction_entry.update({
                    "response": workflow_result.get('response', ''),
                    "code_executed": workflow_result.get('code_executed', False),
                    "exec_results": workflow_result.get('exec_results', []),
                    "num_code_blocks": workflow_result.get('num_code_blocks', 0),
                    "used_feedback": workflow_result.get('used_feedback', False),
                    "first_answer": workflow_result.get('first_answer'),
                    "first_correct": workflow_result.get('first_correct', False),
                    "tools_config": workflow_result.get('tools_config', {})
                })
            elif args.agent == 'agent_default_prompt_with_code':
                prediction_entry.update({
                    "response": workflow_result.get('response', ''),
                    "code_executed": workflow_result.get('code_executed', False),
                    "exec_results": workflow_result.get('exec_results', []),
                    "num_code_blocks": workflow_result.get('num_code_blocks', 0),
                    "used_feedback": workflow_result.get('used_feedback', False),
                    "first_answer": workflow_result.get('first_answer'),
                    "first_correct": workflow_result.get('first_correct', False),
                    "config": workflow_result.get('config', {})
                })
            elif args.agent == 'agent_code_as_answer':
                prediction_entry.update({
                    "response": workflow_result.get('response', ''),
                    "code_executed": workflow_result.get('code_executed', False),
                    "exec_results": workflow_result.get('exec_results', []),
                    "num_code_blocks": workflow_result.get('num_code_blocks', 0),
                    "code_answer": workflow_result.get('code_answer'),
                    "model_answer": workflow_result.get('model_answer'),
                    "answer_source": workflow_result.get('answer_source'),
                    "tools_config": workflow_result.get('tools_config', {})
                })
            elif args.agent == 'majority_vote':
                prediction_entry.update({
                    "num_runs": num_runs,
                    "runs": runs,
                    "answers": workflow_result.get('answers', []),
                    "answer_counts": workflow_result.get('answer_counts', {})
                })
            
            results["predictions"].append(prediction_entry)
            
            # Log detailed info
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'─'*40}\n")
                if args.agent in ['solver_checker', 'solver_checker_chat', 'solver_checker_trivial_chat', 'solver_checker_with_tools', 'solver_checker_with_tools_v2', 'solver_checker_summarizer', 'solver_checker_summarizer_chat']:
                    f.write(f"Total Iterations: {total_iterations}\n")
                    for iter_data in iterations:
                        f.write(f"\n--- Iteration {iter_data['iteration']} ---\n")
                        f.write(f"Solver Answer: {iter_data.get('solver_answer', 'N/A')}\n")
                        f.write(f"Actually Correct: {iter_data.get('is_actually_correct', False)}\n")
                        f.write(f"Checker Verdict: {iter_data.get('checker_verdict', 'N/A')}\n")
                        if args.agent in ['solver_checker_chat', 'solver_checker_trivial_chat']:
                            f.write(f"Checker Feedback: {iter_data.get('checker_feedback', '')}\n")
                            f.write(f"Conversation Length: {iter_data.get('conversation_length', 0)}\n")
                        elif args.agent in ['solver_checker_with_tools', 'solver_checker_with_tools_v2']:
                            f.write(f"Solver Tools Used: {iter_data.get('solver_tools_used', False)}\n")
                            f.write(f"Checker Tools Used: {iter_data.get('checker_tools_used', False)}\n")
                            f.write(f"Checker Feedback: {iter_data.get('checker_feedback', '')}\n")
                        f.write(f"Solver Response:\n{iter_data.get('solver_response', '')}\n")
                        f.write(f"Checker Response:\n{iter_data.get('checker_response', '')}\n")
                elif args.agent == 'plan_and_reflection':
                    f.write(f"Total Iterations: {total_iterations}\n")
                    for iter_data in iterations:
                        f.write(f"\n--- Iteration {iter_data['iteration']} ---\n")
                        plan_info = iter_data.get('plan', {})
                        f.write(f"Plan: {plan_info.get('num_subproblems', 0)} sub-problems\n")
                        execute_info = iter_data.get('execute', {})
                        f.write(f"Execute: {execute_info.get('num_solved', 0)} solved\n")
                        reflect_info = iter_data.get('reflect', {})
                        f.write(f"Reflect Verdict: {reflect_info.get('verdict', 'N/A')}\n")
                        f.write(f"Final Answer: {iter_data.get('answer', 'N/A')}\n")
                        f.write(f"Correct: {iter_data.get('is_correct', False)}\n")
                elif args.agent == 'solver_verifier':
                    f.write(f"Total Iterations: {total_iterations}\n")
                    for iter_data in iterations:
                        f.write(f"\n--- Iteration {iter_data['iteration']} ---\n")
                        f.write(f"Solver Answer: {iter_data.get('solver_answer', 'N/A')}\n")
                        f.write(f"Code Result: {iter_data.get('code_result', 'N/A')}\n")
                        f.write(f"Boxed Answer: {iter_data.get('boxed_answer', 'N/A')}\n")
                        f.write(f"Verifier Verdict: {iter_data.get('verifier_verdict', 'N/A')}\n")
                        f.write(f"Actually Correct: {iter_data.get('is_actually_correct', False)}\n")
                        f.write(f"Solver Response:\n{iter_data.get('solver_response', '')}\n")
                elif args.agent == 'solver_coder':
                    f.write(f"Total Iterations: {total_iterations}\n")
                    f.write(f"Error Count: {workflow_result.get('error_count', 0)}\n")
                    f.write(f"Success Count: {workflow_result.get('success_count', 0)}\n")
                    for iter_data in iterations:
                        f.write(f"\n--- Iteration {iter_data['iteration']} ---\n")
                        f.write(f"Exec Success: {iter_data.get('exec_success', False)}\n")
                        f.write(f"Extracted Answer: {iter_data.get('extracted_answer', 'N/A')}\n")
                        f.write(f"Checker Verdict: {iter_data.get('checker_verdict', 'N/A')}\n")
                        if iter_data.get('exec_error'):
                            f.write(f"Error: {iter_data.get('exec_error', '')}\n")
                        if iter_data.get('code'):
                            f.write(f"Code:\n{iter_data.get('code', '')}\n")
                        if iter_data.get('exec_output'):
                            f.write(f"Output: {iter_data.get('exec_output', '')}\n")
                elif args.agent == 'majority_vote':
                    f.write(f"Total Runs: {num_runs}\n")
                    for run_data in runs:
                        f.write(f"\n--- Run {run_data['run']} (seed={run_data['seed']}) ---\n")
                        f.write(f"Answer: {run_data['answer']}\n")
                        f.write(f"Correct: {run_data.get('is_correct', False)}\n")
                        f.write(f"Response:\n{run_data['response']}\n")
                    f.write(f"\nAnswer Counts: {workflow_result.get('answer_counts', {})}\n")
                elif args.agent == 'agent_with_python_tools':
                    # Detailed logging for Python tools agent
                    f.write(f"Code Executed: {workflow_result.get('code_executed', False)}\n")
                    exec_results = workflow_result.get('exec_results', [])
                    f.write(f"Number of Code Blocks: {len(exec_results)}\n")
                    
                    if exec_results:
                        f.write(f"\nCode Execution Results:\n")
                        for i, result in enumerate(exec_results, 1):
                            if result.get('success'):
                                f.write(f"  Block {i}: Success\n")
                                f.write(f"    Output: {result.get('output', '')}\n")
                            else:
                                f.write(f"  Block {i}: Error\n")
                                f.write(f"    Error: {result.get('error', '')}\n")
                    
                    response = workflow_result.get('response', '')
                    f.write(f"\nModel Response:\n{response}\n")
                elif args.agent == 'solver_interactive_code':
                    # Detailed logging for interactive code agent
                    total_execs = workflow_result.get('total_code_executions', 0)
                    f.write(f"Code Executed: {total_execs > 0}\n")
                    f.write(f"Number of Code Executions: {total_execs}\n")
                    
                    # Get exec_results from iterations
                    if iterations:
                        all_exec_results = []
                        for it in iterations:
                            all_exec_results.extend(it.get('exec_results', []))
                        
                        if all_exec_results:
                            f.write(f"\nCode Execution Results:\n")
                            for i, result in enumerate(all_exec_results, 1):
                                if result.get('success'):
                                    f.write(f"  Block {i}: Success\n")
                                    f.write(f"    Output: {result.get('output', '')}\n")
                                else:
                                    f.write(f"  Block {i}: Error\n")
                                    f.write(f"    Error: {result.get('error', '')}\n")
                        
                        # Use last iteration's response
                        response = iterations[-1].get('response', '')
                        f.write(f"\nModel Response:\n{response}\n")
                f.write(f"\n{'─'*40}\n")
                f.write(f"Final Answer: {predicted_answer}\n")
                f.write(f"Final Correct: {final_correct}\n")
                f.write(f"Case Type: {case_type}\n")
                f.write(f"Time: {sample_time:.2f}s\n")
                f.write(f"{'='*80}\n")
            
            # Print progress
            current_accuracy = stats['correct'] / stats['total']
            first_try_accuracy = stats['first_try_correct'] / stats['total'] if stats['total'] > 0 else 0
            
            status_symbol = "✓" if final_correct else "✗"
            
            if detailed:
                print(f"\n{status_symbol} Final Answer: {predicted_answer} (Expected: {ground_truth})")
                if args.agent in ['solver_checker', 'solver_checker_summarizer', 'solver_checker_summarizer_chat']:
                    print(f"Case Type: {case_type} | Iterations: {total_iterations}")
                elif args.agent == 'majority_vote':
                    print(f"Case Type: {case_type} | Runs: {num_runs}")
                print(f"Overall Accuracy: {current_accuracy*100:.1f}% | First Try: {first_try_accuracy*100:.1f}%")
                print(f"Improved: {stats['improved_cases']} | Degraded: {stats['degraded_cases']}")
                print("="*80)
            else:
                progress_bar.set_postfix({
                    'Acc': f'{current_accuracy*100:.1f}%',
                    'First': f'{first_try_accuracy*100:.1f}%',
                    'Imp': stats['improved_cases'],
                    'Deg': stats['degraded_cases']
                })
                progress_bar.update(1)
            
            # Save periodically
            if processed_count % args.save_interval == 0:
                save_results()
                if detailed:
                    print(f"💾 Saved intermediate results")
        
        except Exception as e:
            error_msg = f"ERROR processing sample {dataset_idx}: {e}"
            log_and_print(f"\n✗ {error_msg}")
            
            stats['total'] += 1
            results["predictions"].append({
                "question_id": dataset_idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": None,
                "final_correct": False,
                "error": str(e)
            })
            
            if not detailed:
                progress_bar.update(1)
    
    if progress_bar:
        progress_bar.close()
    
    # Final statistics
    end_time = time.time()
    results["total_time"] = end_time - start_time
    results["avg_time_per_sample"] = results["total_time"] / stats['total'] if stats['total'] > 0 else 0
    results["accuracy"] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    results["correct"] = stats['correct']
    results["total"] = stats['total']
    results["stats"] = {
        "total": stats['total'],
        "correct": stats['correct'],
        "accuracy": results["accuracy"],
        "first_try_correct": stats['first_try_correct'],
        "first_try_accuracy": stats['first_try_correct'] / stats['total'] if stats['total'] > 0 else 0,
        "improved_cases": stats['improved_cases'],
        "degraded_cases": stats['degraded_cases'],
        "failed_cases": stats['failed_cases'],
        "other_cases": stats['other_cases'],
        "avg_iterations": stats['total_iterations'] / stats['total'] if stats['total'] > 0 and args.agent == 'solver_checker' else 0,
        "checker_verdicts": dict(stats['checker_verdicts']),
        "false_positives": stats['false_positives'],
        "false_negatives": stats['false_negatives']
    }
    
    # Save final results
    save_results()
    
    # Save unified metrics.csv
    save_metrics_csv(results, stats, results_dir, args)
    
    # Generate detailed analysis report
    generate_analysis_report(results, stats, results_dir, args)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Evaluation Summary:")
    print(f"  Total Samples: {stats['total']}")
    print(f"  Overall Accuracy: {results['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    print(f"  First Try Accuracy: {results['stats']['first_try_accuracy']*100:.2f}% ({stats['first_try_correct']}/{stats['total']})")
    print(f"  Improved Cases: {stats['improved_cases']}")
    print(f"  Degraded Cases: {stats['degraded_cases']}")
    print(f"  Avg Iterations: {results['stats']['avg_iterations']:.2f}")
    print(f"  False Positives: {stats['false_positives']}")
    print(f"  False Negatives: {stats['false_negatives']}")
    print(f"{'='*80}\n")
    
    print(f"Results saved to: {answer_file}")
    print(f"Analysis report: {results_dir / 'analysis_report.txt'}\n")
    
    # Cleanup
    del model
    if args.agent in ['solver_checker', 'solver_checker_summarizer'] and args.checker_model and args.checker_model != args.model:
        del checker_model
    torch.cuda.empty_cache()
    
    return results


def save_metrics_csv(results, stats, results_dir, args):
    """Save unified metrics to CSV file"""
    import pandas as pd
    
    metrics_csv = results_dir / "metrics.csv"
    
    # Build unified metrics (common across all agent types)
    metrics_data = {
        'model': [args.model],
        'agent': [args.agent],
        'dataset': [args.dataset],
        'total_samples': [stats['total']],
        'correct': [stats['correct']],
        'accuracy': [stats['correct'] / stats['total'] if stats['total'] > 0 else 0],
        'first_try_correct': [stats.get('first_try_correct', 0)],
        'first_try_accuracy': [stats['first_try_correct'] / stats['total'] if stats['total'] > 0 else 0],
        'improved_cases': [stats.get('improved_cases', 0)],
        'degraded_cases': [stats.get('degraded_cases', 0)],
        'failed_cases': [stats.get('failed_cases', 0)],
        'avg_time_per_sample': [results['avg_time_per_sample']],
        'total_time': [results['total_time']],
        'timestamp': [results['timestamp']]
    }
    
    # Add agent-specific metrics
    if args.agent in ['solver_checker', 'solver_checker_chat', 'solver_checker_trivial_chat', 
                      'solver_checker_with_tools', 'solver_checker_with_tools_v2', 'solver_checker_summarizer', 'solver_checker_summarizer_chat']:
        metrics_data['avg_iterations'] = [stats['total_iterations'] / stats['total'] if stats['total'] > 0 else 0]
        metrics_data['max_iterations'] = [args.max_iterations]
        
        # Checker model info
        if args.agent in ['solver_checker', 'solver_checker_summarizer', 'solver_checker_with_tools', 'solver_checker_with_tools_v2']:
            metrics_data['checker_model'] = [args.checker_model if args.checker_model else args.model]
        else:
            metrics_data['checker_model'] = [args.model]  # Shared model
        
        metrics_data['false_positives'] = [stats.get('false_positives', 0)]
        metrics_data['false_negatives'] = [stats.get('false_negatives', 0)]
        
        # Tool-specific
        if args.agent in ['solver_checker_with_tools', 'solver_checker_with_tools_v2']:
            metrics_data['solver_tools_enabled'] = [args.enable_solver_tools]
            metrics_data['checker_tools_enabled'] = [args.enable_checker_tools]
        
        # Summarizer-specific
        if 'summarizer' in args.agent:
            metrics_data['use_summarizer'] = [True]
    
    elif args.agent == 'majority_vote':
        metrics_data['avg_iterations'] = [1.0]  # Each run is 1 iteration
        metrics_data['num_runs'] = [args.num_runs]
        metrics_data['temperature'] = [args.temperature]
        metrics_data['top_p'] = [args.top_p]
    
    elif args.agent == 'plan_and_reflection':
        metrics_data['avg_iterations'] = [stats['total_iterations'] / stats['total'] if stats['total'] > 0 else 0]
        metrics_data['max_iterations'] = [args.max_iterations]
        metrics_data['max_subproblems'] = [args.max_subproblems]
    
    elif args.agent in ['solver_verifier', 'solver_verifier_check']:
        metrics_data['avg_iterations'] = [stats['total_iterations'] / stats['total'] if stats['total'] > 0 else 0]
        metrics_data['max_iterations'] = [args.max_iterations]
    
    elif args.agent == 'solver_coder':
        metrics_data['avg_iterations'] = [stats['total_iterations'] / stats['total'] if stats['total'] > 0 else 0]
        metrics_data['max_iterations'] = [args.max_iterations]
        metrics_data['code_timeout'] = [args.code_timeout]
    
    # Save to CSV
    df = pd.DataFrame(metrics_data)
    df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics CSV: {metrics_csv}")


def create_empty_results(args, num_samples, start_index):
    """Create empty results structure"""
    result = {
        "model": args.model,
        "agent": args.agent,
        "dataset": args.dataset,
        "mode": "agent",
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
    
    if args.agent in ['solver_checker', 'solver_checker_summarizer']:
        result["checker_model"] = args.checker_model or args.model
        result["max_iterations"] = args.max_iterations
        if 'summarizer' in args.agent:
            result["use_summarizer"] = True
    elif args.agent in ['solver_checker_chat', 'solver_checker_trivial_chat', 'solver_checker_summarizer_chat']:
        result["max_iterations"] = args.max_iterations
        if 'summarizer' in args.agent:
            result["use_summarizer"] = True
    elif args.agent == 'majority_vote':
        result["num_runs"] = args.num_runs
        result["temperature"] = args.temperature
        result["top_p"] = args.top_p
    
    return result


def generate_analysis_report(results, stats, results_dir, args):
    """Generate detailed analysis report"""
    report_file = results_dir / 'analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AGENT EVALUATION - DETAILED ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Basic info
        f.write("Configuration:\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Agent Method: {args.agent}\n")
        if args.agent in ['solver_checker', 'solver_checker_summarizer']:
            f.write(f"Checker Model: {args.checker_model or args.model}\n")
            f.write(f"Max Iterations: {args.max_iterations}\n")
            if 'summarizer' in args.agent:
                f.write(f"Use Summarizer: Yes\n")
        elif args.agent in ['solver_checker_chat', 'solver_checker_trivial_chat', 'solver_checker_summarizer_chat']:
            f.write(f"Max Iterations: {args.max_iterations}\n")
            if 'summarizer' in args.agent:
                f.write(f"Use Summarizer: Yes (Chat mode)\n")
        elif args.agent == 'majority_vote':
            f.write(f"Num Runs: {args.num_runs}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Top-p: {args.top_p}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Samples: {stats['total']}\n")
        f.write(f"Correct Answers: {stats['correct']}\n")
        f.write(f"Overall Accuracy: {results['accuracy']*100:.2f}%\n")
        f.write(f"First Try Correct: {stats['first_try_correct']}\n")
        f.write(f"First Try Accuracy: {results['stats']['first_try_accuracy']*100:.2f}%\n")
        f.write(f"Average Iterations: {results['stats']['avg_iterations']:.2f}\n")
        f.write(f"Total Iterations: {stats['total_iterations']}\n\n")
        
        # Case type distribution
        f.write("Case Type Distribution:\n")
        f.write("-"*80 + "\n")
        f.write(f"Improved Cases (Wrong -> Correct): {stats['improved_cases']}\n")
        f.write(f"Degraded Cases (Correct -> Wrong): {stats['degraded_cases']}\n")
        f.write(f"Failed Cases (Wrong -> Wrong): {stats['failed_cases']}\n")
        f.write(f"First Try Success: {stats['first_try_correct']}\n")
        f.write(f"Other Cases: {stats['other_cases']}\n\n")
        
        # Checker performance (for solver_checker variants)
        if args.agent in ['solver_checker', 'solver_checker_summarizer', 'solver_checker_summarizer_chat']:
            f.write("Checker Performance:\n")
            f.write("-"*80 + "\n")
            total_verdicts = sum(stats['checker_verdicts'].values())
            for verdict, count in sorted(stats['checker_verdicts'].items()):
                pct = count / total_verdicts * 100 if total_verdicts > 0 else 0
                f.write(f"  {verdict}: {count} ({pct:.1f}%)\n")
            f.write(f"\nFalse Positives (Checker says CORRECT but wrong): {stats['false_positives']}\n")
            f.write(f"False Negatives (Checker says INCORRECT but correct): {stats['false_negatives']}\n")
            
            if total_verdicts > 0:
                fp_rate = stats['false_positives'] / stats['checker_verdicts'].get('CORRECT', 1) * 100
                fn_rate = stats['false_negatives'] / stats['checker_verdicts'].get('INCORRECT', 1) * 100
                f.write(f"False Positive Rate: {fp_rate:.1f}% (of CORRECT verdicts)\n")
                f.write(f"False Negative Rate: {fn_rate:.1f}% (of INCORRECT verdicts)\n")
            f.write("\n")
            
            # Verdict by iteration
            f.write("Checker Verdict Distribution by Iteration:\n")
            f.write("-"*80 + "\n")
            for iter_num in sorted(stats['verdict_by_iteration'].keys()):
                iter_verdicts = stats['verdict_by_iteration'][iter_num]
                f.write(f"Iteration {iter_num}:\n")
                for verdict, count in sorted(iter_verdicts.items()):
                    f.write(f"  {verdict}: {count}\n")
            f.write("\n")
        
        # Answer change analysis (for solver_checker variants)
        if args.agent in ['solver_checker', 'solver_checker_summarizer', 'solver_checker_summarizer_chat']:
            f.write("Answer Change Analysis:\n")
            f.write("-"*80 + "\n")
            answer_changes = 0
            verdict_changes = 0
            for pred in results['predictions']:
                if 'iterations' not in pred or len(pred['iterations']) < 2:
                    continue
                
                iterations = pred['iterations']
                answers = [iter_data['solver_answer'] for iter_data in iterations if iter_data['solver_answer']]
                verdicts = [iter_data['checker_verdict'] for iter_data in iterations]
                
                if len(set(answers)) > 1:
                    answer_changes += 1
                if len(set(verdicts)) > 1:
                    verdict_changes += 1
            
            f.write(f"Cases with solver answer changes: {answer_changes}\n")
            f.write(f"Cases with checker verdict changes: {verdict_changes}\n\n")
            
            # Iteration distribution
            f.write("Iteration Distribution:\n")
            f.write("-"*80 + "\n")
            iter_dist = Counter()
            for pred in results['predictions']:
                if 'total_iterations' in pred:
                    iter_dist[pred['total_iterations']] += 1
            for iter_count in sorted(iter_dist.keys()):
                f.write(f"  {iter_count} iteration(s): {iter_dist[iter_count]} cases\n")
            f.write("\n")
        elif args.agent == 'majority_vote':
            # Answer distribution analysis
            f.write("Answer Distribution Analysis:\n")
            f.write("-"*80 + "\n")
            all_answer_counts = Counter()
            for pred in results['predictions']:
                if 'answer_counts' in pred:
                    all_answer_counts.update(pred['answer_counts'])
            
            f.write("Overall answer frequency:\n")
            for answer, count in all_answer_counts.most_common(10):
                f.write(f"  {answer}: {count} times\n")
            f.write("\n")
        
        # Detailed case examples
        f.write("Case Examples:\n")
        f.write("-"*80 + "\n")
        
        # Improved cases
        improved = [p for p in results['predictions'] if p.get('case_type') == 'IMPROVED']
        if improved:
            f.write(f"\nImproved Cases (all {len(improved)}):\n")
            for i, case in enumerate(improved, 1):
                f.write(f"\n  Example {i}:\n")
                f.write(f"    Question ID: {case['question_id']}\n")
                f.write(f"    First Answer: {case['first_answer']} (WRONG)\n")
                f.write(f"    Final Answer: {case['predicted_answer']} (CORRECT)\n")
                if 'total_iterations' in case:
                    f.write(f"    Iterations: {case['total_iterations']}\n")
        
        # Degraded cases
        degraded = [p for p in results['predictions'] if p.get('case_type') == 'DEGRADED']
        if degraded:
            f.write(f"\nDegraded Cases (all {len(degraded)}):\n")
            for i, case in enumerate(degraded, 1):
                f.write(f"\n  Example {i}:\n")
                f.write(f"    Question ID: {case['question_id']}\n")
                f.write(f"    First Answer: {case['first_answer']} (CORRECT)\n")
                f.write(f"    Final Answer: {case['predicted_answer']} (WRONG)\n")
                if 'total_iterations' in case:
                    f.write(f"    Iterations: {case['total_iterations']}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Analysis report saved to: {report_file}")


if __name__ == "__main__":
    args = parse_args()
    result = run_evaluation(args)
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)

