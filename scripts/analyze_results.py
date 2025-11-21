#!/usr/bin/env python3
"""
Analytics script for multi-agent evaluation results.

This script analyzes log files from the evaluation pipeline and generates:
- Detailed CSV reports with per-question analysis
- Summary statistics
- Visualizations of checker behavior and iteration patterns

Usage:
    python scripts/analyze_results.py [--log-dir PATH]
    
If no log-dir is specified, automatically uses the latest results directory.
"""

import argparse
import os
import re
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import csv


def find_latest_results_dir(results_base="results"):
    """Find the most recent results directory."""
    results_path = Path(results_base)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_base}")
    
    # Get all subdirectories with timestamps
    subdirs = [d for d in results_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No result directories found in {results_base}")
    
    # Sort by modification time (most recent first)
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest


def parse_log_file(log_file_path):
    """Parse the evaluation log file and extract structured data."""
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata from header
    metadata = {}
    header_match = re.search(r'Model: (.+?)\n.*?Round: (.+?)\n.*?Dataset: (.+?)\n.*?Count: (\d+)', content, re.DOTALL)
    if header_match:
        metadata['model'] = header_match.group(1).strip()
        metadata['round'] = header_match.group(2).strip()
        metadata['dataset'] = header_match.group(3).strip()
        metadata['count'] = int(header_match.group(4))
    
    # Also try to extract from filename (more reliable)
    # Format: Qwen2.5-Math-1.5B_gsm8k_multi_agent.log
    filename = Path(log_file_path).name
    filename_match = re.match(r'(.+?)_([^_]+)_multi_agent\.log', filename)
    if filename_match:
        if 'model' not in metadata or metadata['model'] == 'N/A':
            metadata['model'] = filename_match.group(1)
        if 'dataset' not in metadata or metadata['dataset'] == 'N/A':
            metadata['dataset'] = filename_match.group(2)
    
    # Split by sample
    sample_pattern = r'\[Sample (\d+)/(\d+) \| Dataset Index: (\d+)\]'
    samples = re.split(sample_pattern, content)
    
    results = []
    
    for i in range(1, len(samples), 4):
        if i + 3 > len(samples):
            break
            
        sample_num = int(samples[i])
        total_samples = int(samples[i + 1])
        dataset_idx = int(samples[i + 2])
        sample_content = samples[i + 3]
        
        # Extract question and ground truth
        question_match = re.search(r'Full Question: (.+?)(?=\n\n|\n=)', sample_content, re.DOTALL)
        gt_match = re.search(r'Ground Truth: (.+)', sample_content)
        
        question = question_match.group(1).strip() if question_match else ""
        ground_truth = gt_match.group(1).strip() if gt_match else ""
        
        # Extract iterations
        iteration_pattern = r'--- Iteration (\d+) ---\n(.+?)(?=--- Iteration|\n\n-{40}\nFinal|$)'
        iterations = re.findall(iteration_pattern, sample_content, re.DOTALL)
        
        iteration_data = []
        for iter_num, iter_content in iterations:
            # Extract solver response (full text before checker)
            solver_response_match = re.search(r'Solver Response:\n(.+?)(?=\n\nChecker Response:)', iter_content, re.DOTALL)
            solver_response = solver_response_match.group(1).strip() if solver_response_match else ""
            
            # Extract solver answer
            solver_answer_match = re.search(r'\\boxed\{([^}]+)\}', iter_content)
            solver_answer = solver_answer_match.group(1).strip() if solver_answer_match else None
            
            # Extract checker response and verdict
            checker_response_match = re.search(r'Checker Response:\n(.+?)\nChecker Verdict: (.+)', iter_content, re.DOTALL)
            if checker_response_match:
                checker_response = checker_response_match.group(1).strip()
                checker_verdict = checker_response_match.group(2).strip()
            else:
                checker_response = ""
                checker_verdict = "UNCLEAR"
            
            iteration_data.append({
                'iteration': int(iter_num),
                'solver_response': solver_response,
                'solver_answer': solver_answer,
                'checker_response': checker_response,
                'checker_verdict': checker_verdict
            })
        
        # Extract final result
        final_answer_match = re.search(r'Final Predicted Answer: (.+)', sample_content)
        predicted_answer = final_answer_match.group(1).strip() if final_answer_match else None
        
        result_match = re.search(r'Result: (CORRECT|WRONG)', sample_content)
        is_correct = result_match.group(1) == "CORRECT" if result_match else False
        
        total_iterations_match = re.search(r'Total Iterations: (\d+)', sample_content)
        total_iterations = int(total_iterations_match.group(1)) if total_iterations_match else len(iteration_data)
        
        results.append({
            'sample_num': sample_num,
            'dataset_idx': dataset_idx,
            'question': question,
            'ground_truth': ground_truth,
            'iterations': iteration_data,
            'total_iterations': total_iterations,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        })
    
    return metadata, results


def analyze_results(results):
    """Generate analytics from parsed results."""
    analytics = {
        'total_problems': len(results),
        'correct_answers': sum(1 for r in results if r['is_correct']),
        'accuracy': 0.0,
        
        # Overall statistics
        'total_iterations': 0,
        'avg_iterations': 0.0,
        'single_iter_correct': 0,
        'single_iter_wrong': 0,
        'multi_iter_correct': 0,
        'multi_iter_wrong': 0,
        
        # Verdict distribution
        'verdict_counts': {'CORRECT': 0, 'INCORRECT': 0, 'UNCLEAR': 0},
        
        # Four key groups
        'improved_cases': [],  # Type 1: First answer wrong -> Later correct (improvement)
        'degraded_cases': [],  # Type 2: First answer correct -> Later wrong (degradation)
        'first_try_success': [],  # Type 3: First iteration correct (efficient)
        'unnecessary_iterations': [],  # Type 4: Got correct answer but checker didn't stop
    }
    
    for result in results:
        iterations = result['iterations']
        if not iterations:
            continue
        
        ground_truth = str(result['ground_truth']).strip()
        total_iters = result['total_iterations']
        
        # Update overall statistics
        analytics['total_iterations'] += total_iters
        
        if total_iters == 1:
            if result['is_correct']:
                analytics['single_iter_correct'] += 1
            else:
                analytics['single_iter_wrong'] += 1
        else:
            if result['is_correct']:
                analytics['multi_iter_correct'] += 1
            else:
                analytics['multi_iter_wrong'] += 1
        
        # Count verdicts
        for iter_data in iterations:
            verdict = iter_data['checker_verdict']
            if verdict in analytics['verdict_counts']:
                analytics['verdict_counts'][verdict] += 1
        
        # Check each iteration to see if answer matches ground truth
        first_answer = str(iterations[0]['solver_answer']).strip() if iterations[0]['solver_answer'] else None
        first_answer_correct = (first_answer == ground_truth) if first_answer else False
        first_verdict = iterations[0]['checker_verdict']
        
        # Find which iteration got the correct answer (if any)
        correct_iteration = None
        wrong_iteration = None
        
        for idx, iter_data in enumerate(iterations, 1):
            iter_answer = str(iter_data['solver_answer']).strip() if iter_data['solver_answer'] else None
            if iter_answer == ground_truth and correct_iteration is None:
                correct_iteration = idx
            elif iter_answer != ground_truth and iter_answer is not None:
                if wrong_iteration is None:
                    wrong_iteration = idx
        
        # TYPE 1: Improvement Case
        # First iteration answer was WRONG, but later iteration got it CORRECT
        if not first_answer_correct and result['is_correct'] and result['total_iterations'] > 1:
            analytics['improved_cases'].append({
                'question_num': result['sample_num'],
                'dataset_idx': result['dataset_idx'],
                'question': result['question'],
                'ground_truth': ground_truth,
                'first_answer': first_answer,
                'first_correct': False,
                'final_answer': result['predicted_answer'],
                'final_correct': True,
                'wrong_at_iteration': 1,
                'correct_at_iteration': correct_iteration,
                'total_iterations': result['total_iterations'],
                'all_iterations': [{
                    'iteration': i,
                    'solver_answer': iter_data['solver_answer'],
                    'checker_verdict': iter_data['checker_verdict'],
                    'solver_response': iter_data['solver_response'],
                    'checker_response': iter_data['checker_response']
                } for i, iter_data in enumerate(iterations, 1)]
            })
        
        # TYPE 2: Degradation Case
        # First iteration answer was CORRECT, but later iterations made it WRONG
        if first_answer_correct and not result['is_correct'] and total_iters > 1:
            analytics['degraded_cases'].append({
                'question_num': result['sample_num'],
                'dataset_idx': result['dataset_idx'],
                'question': result['question'],
                'ground_truth': ground_truth,
                'first_answer': first_answer,
                'first_correct': True,
                'final_answer': result['predicted_answer'],
                'final_correct': False,
                'correct_at_iteration': 1,
                'wrong_at_iteration': wrong_iteration if wrong_iteration and wrong_iteration > 1 else total_iters,
                'total_iterations': total_iters,
                'all_iterations': [{
                    'iteration': i,
                    'solver_answer': iter_data['solver_answer'],
                    'checker_verdict': iter_data['checker_verdict'],
                    'solver_response': iter_data['solver_response'],
                    'checker_response': iter_data['checker_response']
                } for i, iter_data in enumerate(iterations, 1)]
            })
        
        # TYPE 3: First Try Success
        # First iteration got the correct answer (efficient)
        if first_answer_correct and result['is_correct'] and total_iters == 1:
            analytics['first_try_success'].append({
                'question_num': result['sample_num'],
                'dataset_idx': result['dataset_idx'],
                'question': result['question'],
                'ground_truth': ground_truth,
                'answer': first_answer,
                'checker_verdict': first_verdict
            })
        
        # TYPE 4: Unnecessary Iterations
        # Solver got correct answer, but checker didn't recognize it, leading to extra iterations
        if correct_iteration and correct_iteration < total_iters:
            # Check if checker said CORRECT at the correct iteration
            correct_iter_verdict = iterations[correct_iteration - 1]['checker_verdict']
            if correct_iter_verdict != "CORRECT":
                # Checker missed it - there were unnecessary iterations after the correct answer
                analytics['unnecessary_iterations'].append({
                    'question_num': result['sample_num'],
                    'dataset_idx': result['dataset_idx'],
                    'question': result['question'],
                    'ground_truth': ground_truth,
                    'correct_answer_at': correct_iteration,
                    'checker_verdict_at_correct': correct_iter_verdict,
                    'total_iterations': total_iters,
                    'unnecessary_iterations_count': total_iters - correct_iteration,
                    'unnecessary_range': f"{correct_iteration + 1} to {total_iters}",
                    'final_answer': result['predicted_answer'],
                    'final_correct': result['is_correct'],
                    'all_iterations': [{
                        'iteration': i,
                        'solver_answer': iter_data['solver_answer'],
                        'checker_verdict': iter_data['checker_verdict'],
                        'solver_response': iter_data['solver_response'],
                        'checker_response': iter_data['checker_response']
                    } for i, iter_data in enumerate(iterations, 1)]
                })
    
    if analytics['total_problems'] > 0:
        analytics['accuracy'] = analytics['correct_answers'] / analytics['total_problems']
        analytics['avg_iterations'] = analytics['total_iterations'] / analytics['total_problems']
    
    return analytics


def save_analysis_csv(analytics, output_path, metadata):
    """Save single CSV with improved and degraded cases."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header section
        writer.writerow(['=== MULTI-AGENT EVALUATION ANALYSIS ==='])
        writer.writerow(['Model', metadata.get('model', 'N/A')])
        writer.writerow(['Dataset', metadata.get('dataset', 'N/A')])
        writer.writerow([])
        writer.writerow(['=== OVERALL STATISTICS ==='])
        writer.writerow(['Total Problems', analytics['total_problems']])
        writer.writerow(['Correct Answers', analytics['correct_answers']])
        writer.writerow(['Accuracy', f"{analytics['accuracy']:.2%}"])
        writer.writerow(['Total Iterations', analytics['total_iterations']])
        writer.writerow(['Average Iterations per Problem', f"{analytics['avg_iterations']:.2f}"])
        writer.writerow([])
        writer.writerow(['Single Iteration - Correct', analytics['single_iter_correct']])
        writer.writerow(['Single Iteration - Wrong', analytics['single_iter_wrong']])
        writer.writerow(['Multi Iteration - Correct', analytics['multi_iter_correct']])
        writer.writerow(['Multi Iteration - Wrong', analytics['multi_iter_wrong']])
        writer.writerow([])
        total_verdicts = sum(analytics['verdict_counts'].values())
        writer.writerow(['Verdict Distribution'])
        for verdict, count in sorted(analytics['verdict_counts'].items()):
            pct = count / total_verdicts * 100 if total_verdicts > 0 else 0
            writer.writerow([f'  {verdict}', count, f'{pct:.1f}%'])
        writer.writerow([])
        
        # TYPE 1: Improved Cases
        writer.writerow(['=== TYPE 1: IMPROVED CASES (Initially Wrong -> Later Correct) ==='])
        writer.writerow(['Count', len(analytics['improved_cases'])])
        writer.writerow([])
        
        if analytics['improved_cases']:
            writer.writerow([
                'Question#', 'Dataset_Idx', 'Question', 'Ground_Truth',
                'First_Answer', 'Final_Answer', 'Wrong_at_Iteration',
                'Correct_at_Iteration', 'Total_Iterations',
                'Iteration#', 'Iter_Answer', 'Checker_Verdict',
                'Solver_Response_Preview', 'Checker_Response'
            ])
            
            for case in analytics['improved_cases']:
                for i, iter_data in enumerate(case['all_iterations']):
                    if i == 0:  # First row includes question info
                        writer.writerow([
                            case['question_num'],
                            case['dataset_idx'],
                            case['question'],
                            case['ground_truth'],
                            case['first_answer'],
                            case['final_answer'],
                            case['wrong_at_iteration'],
                            case['correct_at_iteration'],
                            case['total_iterations'],
                            iter_data['iteration'],
                            iter_data['solver_answer'],
                            iter_data['checker_verdict'],
                            (iter_data['solver_response'][:150] + '...') if len(iter_data['solver_response']) > 150 else iter_data['solver_response'],
                            iter_data['checker_response']
                        ])
                    else:  # Subsequent rows only have iteration data
                        writer.writerow([
                            '', '', '', '', '', '', '', '', '',  # Empty question cols
                            iter_data['iteration'],
                            iter_data['solver_answer'],
                            iter_data['checker_verdict'],
                            (iter_data['solver_response'][:150] + '...') if len(iter_data['solver_response']) > 150 else iter_data['solver_response'],
                            iter_data['checker_response']
                        ])
                writer.writerow([])  # Blank row between questions
        
        writer.writerow([])
        
        # TYPE 2: Degraded Cases
        writer.writerow(['=== TYPE 2: DEGRADED CASES (Initially Correct -> Later Wrong) ==='])
        writer.writerow(['Count', len(analytics['degraded_cases'])])
        writer.writerow([])
        
        if analytics['degraded_cases']:
            writer.writerow([
                'Question#', 'Dataset_Idx', 'Question', 'Ground_Truth',
                'First_Answer', 'Final_Answer', 'Correct_at_Iteration',
                'Wrong_at_Iteration', 'Total_Iterations',
                'Iteration#', 'Iter_Answer', 'Checker_Verdict',
                'Solver_Response_Preview', 'Checker_Response'
            ])
            
            for case in analytics['degraded_cases']:
                for i, iter_data in enumerate(case['all_iterations']):
                    if i == 0:  # First row includes question info
                        writer.writerow([
                            case['question_num'],
                            case['dataset_idx'],
                            case['question'],
                            case['ground_truth'],
                            case['first_answer'],
                            case['final_answer'],
                            case['correct_at_iteration'],
                            case['wrong_at_iteration'],
                            case['total_iterations'],
                            iter_data['iteration'],
                            iter_data['solver_answer'],
                            iter_data['checker_verdict'],
                            (iter_data['solver_response'][:150] + '...') if len(iter_data['solver_response']) > 150 else iter_data['solver_response'],
                            iter_data['checker_response']
                        ])
                    else:  # Subsequent rows only have iteration data
                        writer.writerow([
                            '', '', '', '', '', '', '', '', '',  # Empty question cols
                            iter_data['iteration'],
                            iter_data['solver_answer'],
                            iter_data['checker_verdict'],
                            (iter_data['solver_response'][:150] + '...') if len(iter_data['solver_response']) > 150 else iter_data['solver_response'],
                            iter_data['checker_response']
                        ])
                writer.writerow([])  # Blank row between questions
        
        writer.writerow([])
        
        # TYPE 3: First Try Success
        writer.writerow(['=== TYPE 3: FIRST TRY SUCCESS (Correct on First Iteration) ==='])
        writer.writerow(['Count', len(analytics['first_try_success'])])
        writer.writerow([])
        
        if analytics['first_try_success']:
            writer.writerow([
                'Question#', 'Dataset_Idx', 'Question', 'Ground_Truth',
                'First_Answer', 'Checker_Verdict'
            ])
            
            for case in analytics['first_try_success']:
                writer.writerow([
                    case['question_num'],
                    case['dataset_idx'],
                    case['question'],
                    case['ground_truth'],
                    case['answer'],
                    case['checker_verdict']
                ])
        
        writer.writerow([])
        
        # TYPE 4: Unnecessary Iterations
        writer.writerow(['=== TYPE 4: UNNECESSARY ITERATIONS (Correct Answer but Checker Missed It) ==='])
        writer.writerow(['Count', len(analytics['unnecessary_iterations'])])
        writer.writerow([])
        
        if analytics['unnecessary_iterations']:
            writer.writerow([
                'Question#', 'Dataset_Idx', 'Question', 'Ground_Truth',
                'Correct_Answer_At_Iteration', 'Checker_Verdict_At_Correct',
                'Total_Iterations', 'Unnecessary_Iterations_Count', 'Unnecessary_Range',
                'Final_Answer', 'Final_Correct',
                'Iteration#', 'Iter_Answer', 'Checker_Verdict',
                'Solver_Response_Preview', 'Checker_Response'
            ])
            
            for case in analytics['unnecessary_iterations']:
                for i, iter_data in enumerate(case['all_iterations']):
                    if i == 0:  # First row includes question info
                        writer.writerow([
                            case['question_num'],
                            case['dataset_idx'],
                            case['question'],
                            case['ground_truth'],
                            case['correct_answer_at'],
                            case['checker_verdict_at_correct'],
                            case['total_iterations'],
                            case['unnecessary_iterations_count'],
                            case['unnecessary_range'],
                            case['final_answer'],
                            'Yes' if case['final_correct'] else 'No',
                            iter_data['iteration'],
                            iter_data['solver_answer'],
                            iter_data['checker_verdict'],
                            (iter_data['solver_response'][:150] + '...') if len(iter_data['solver_response']) > 150 else iter_data['solver_response'],
                            iter_data['checker_response']
                        ])
                    else:  # Subsequent rows only have iteration data
                        writer.writerow([
                            '', '', '', '', '', '', '', '', '', '', '',  # Empty question cols
                            iter_data['iteration'],
                            iter_data['solver_answer'],
                            iter_data['checker_verdict'],
                            (iter_data['solver_response'][:150] + '...') if len(iter_data['solver_response']) > 150 else iter_data['solver_response'],
                            iter_data['checker_response']
                        ])
                writer.writerow([])  # Blank row between questions
    
    print(f"[OK] Analysis CSV saved to: {output_path}")


def save_detailed_csv_DEPRECATED(results, analytics, output_path, metadata):
    """Save detailed per-question CSV report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'sample_num', 'dataset_idx', 'question_preview',
            'ground_truth', 'predicted_answer', 'is_correct',
            'total_iterations',
            'first_verdict', 'last_verdict',
            'first_answer', 'last_answer',
            'verdict_changed', 'answer_changed',
            'classification'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            iterations = result['iterations']
            if not iterations:
                continue
            
            first_verdict = iterations[0]['checker_verdict']
            last_verdict = iterations[-1]['checker_verdict']
            first_answer = iterations[0]['solver_answer']
            last_answer = iterations[-1]['solver_answer']
            
            verdict_changed = first_verdict != last_verdict
            answer_changed = first_answer != last_answer
            
            # Classify the result
            classification = ""
            if result['total_iterations'] == 1:
                if result['is_correct']:
                    classification = "Single iteration - Correct"
                else:
                    classification = "Single iteration - Wrong (False Positive)" if first_verdict == "CORRECT" else "Single iteration - Wrong"
            else:
                if first_verdict != "CORRECT" and result['is_correct']:
                    classification = "Improved - Final Correct"
                elif first_verdict == "CORRECT" and not result['is_correct']:
                    classification = "Degraded - Final Wrong (False Positive)"
                elif result['is_correct']:
                    classification = "Multi-iteration - Correct"
                else:
                    classification = "Multi-iteration - Wrong"
            
            writer.writerow({
                'sample_num': result['sample_num'],
                'dataset_idx': result['dataset_idx'],
                'question_preview': result['question'][:100] + "..." if len(result['question']) > 100 else result['question'],
                'ground_truth': result['ground_truth'],
                'predicted_answer': result['predicted_answer'],
                'is_correct': 'Yes' if result['is_correct'] else 'No',
                'total_iterations': result['total_iterations'],
                'first_verdict': first_verdict,
                'last_verdict': last_verdict,
                'first_answer': first_answer,
                'last_answer': last_answer,
                'verdict_changed': 'Yes' if verdict_changed else 'No',
                'answer_changed': 'Yes' if answer_changed else 'No',
                'classification': classification
            })
    
    print(f"[OK] Detailed CSV saved to: {output_path}")


def save_improved_questions_csv(analytics, output_path):
    """Save detailed CSV for improved questions (initially wrong -> final correct)."""
    if not analytics['improved_details']:
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'sample_num', 'dataset_idx', 'question', 'ground_truth', 
            'predicted_answer', 'total_iterations',
            'iteration', 'solver_answer', 'checker_verdict', 
            'solver_response_preview', 'checker_response'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for detail in analytics['improved_details']:
            for iteration in detail['iterations']:
                writer.writerow({
                    'sample_num': detail['sample_num'],
                    'dataset_idx': detail['dataset_idx'],
                    'question': detail['question'],
                    'ground_truth': detail['ground_truth'],
                    'predicted_answer': detail['predicted_answer'],
                    'total_iterations': detail['total_iterations'],
                    'iteration': iteration['iteration'],
                    'solver_answer': iteration['solver_answer'],
                    'checker_verdict': iteration['checker_verdict'],
                    'solver_response_preview': iteration['solver_response'][:200] + "..." if len(iteration['solver_response']) > 200 else iteration['solver_response'],
                    'checker_response': iteration['checker_response']
                })
    
    print(f"[OK] Improved questions CSV saved to: {output_path}")


def save_degraded_questions_csv(analytics, output_path):
    """Save detailed CSV for degraded questions (initially correct -> final wrong)."""
    if not analytics['degraded_details']:
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'sample_num', 'dataset_idx', 'question', 'ground_truth', 
            'predicted_answer', 'total_iterations',
            'iteration', 'solver_answer', 'checker_verdict', 
            'solver_response_preview', 'checker_response'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for detail in analytics['degraded_details']:
            for iteration in detail['iterations']:
                writer.writerow({
                    'sample_num': detail['sample_num'],
                    'dataset_idx': detail['dataset_idx'],
                    'question': detail['question'],
                    'ground_truth': detail['ground_truth'],
                    'predicted_answer': detail['predicted_answer'],
                    'total_iterations': detail['total_iterations'],
                    'iteration': iteration['iteration'],
                    'solver_answer': iteration['solver_answer'],
                    'checker_verdict': iteration['checker_verdict'],
                    'solver_response_preview': iteration['solver_response'][:200] + "..." if len(iteration['solver_response']) > 200 else iteration['solver_response'],
                    'checker_response': iteration['checker_response']
                })
    
    print(f"[OK] Degraded questions CSV saved to: {output_path}")


def save_summary_csv(analytics, metadata, output_path):
    """Save summary statistics CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['metric', 'value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Metadata
        writer.writerow({'metric': 'timestamp', 'value': timestamp})
        writer.writerow({'metric': 'model', 'value': metadata.get('model', 'N/A')})
        writer.writerow({'metric': 'dataset', 'value': metadata.get('dataset', 'N/A')})
        writer.writerow({'metric': 'round', 'value': metadata.get('round', 'N/A')})
        
        # Overall statistics
        writer.writerow({'metric': 'total_problems', 'value': analytics['total_problems']})
        writer.writerow({'metric': 'correct_answers', 'value': analytics['correct_answers']})
        writer.writerow({'metric': 'accuracy', 'value': f"{analytics['accuracy']:.2%}"})
        
        # Iteration statistics
        writer.writerow({'metric': 'single_iteration_correct', 'value': analytics['single_iteration_correct']})
        writer.writerow({'metric': 'single_iteration_wrong', 'value': analytics['single_iteration_wrong']})
        writer.writerow({'metric': 'multi_iteration_correct', 'value': analytics['multi_iteration_correct']})
        writer.writerow({'metric': 'multi_iteration_wrong', 'value': analytics['multi_iteration_wrong']})
        
        # Checker performance
        writer.writerow({'metric': 'first_iter_correct_final_correct', 'value': analytics['first_iter_correct_final_correct']})
        writer.writerow({'metric': 'first_iter_correct_final_wrong', 'value': analytics['first_iter_correct_final_wrong']})
        writer.writerow({'metric': 'first_iter_wrong_final_correct', 'value': analytics['first_iter_wrong_final_correct']})
        writer.writerow({'metric': 'first_iter_wrong_final_wrong', 'value': analytics['first_iter_wrong_final_wrong']})
        
        # Improvement statistics
        writer.writerow({'metric': 'improved_questions', 'value': len(analytics['improved_questions'])})
        writer.writerow({'metric': 'degraded_questions', 'value': len(analytics['degraded_questions'])})
        writer.writerow({'metric': 'false_positives', 'value': len(analytics['false_positives'])})
        writer.writerow({'metric': 'false_negatives', 'value': len(analytics['false_negatives'])})
    
    print(f"[OK] Summary CSV saved to: {output_path}")


def print_report(analytics, metadata):
    """Print a formatted report to console."""
    print("\n" + "=" * 80)
    print("MULTI-AGENT EVALUATION ANALYTICS")
    print("=" * 80)
    
    print(f"\nMetadata:")
    print(f"  Model: {metadata.get('model', 'N/A')}")
    print(f"  Dataset: {metadata.get('dataset', 'N/A')}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"  Total Problems: {analytics['total_problems']}")
    print(f"  Correct Answers: {analytics['correct_answers']}")
    print(f"  Accuracy: {analytics['accuracy']:.2%}")
    print(f"  Total Iterations: {analytics['total_iterations']}")
    print(f"  Average Iterations per Problem: {analytics['avg_iterations']:.2f}")
    
    print(f"\n  Single Iteration - Correct: {analytics['single_iter_correct']}")
    print(f"  Single Iteration - Wrong: {analytics['single_iter_wrong']}")
    print(f"  Multi Iteration - Correct: {analytics['multi_iter_correct']}")
    print(f"  Multi Iteration - Wrong: {analytics['multi_iter_wrong']}")
    
    print(f"\n  Verdict Distribution:")
    total_verdicts = sum(analytics['verdict_counts'].values())
    for verdict, count in sorted(analytics['verdict_counts'].items()):
        pct = count / total_verdicts * 100 if total_verdicts > 0 else 0
        print(f"    {verdict}: {count} ({pct:.1f}%)")
    
    print(f"\n=== KEY FINDINGS ===")
    
    print(f"\n[TYPE 1] IMPROVED CASES (Initially Wrong -> Later Correct): {len(analytics['improved_cases'])}")
    if analytics['improved_cases']:
        print(f"  These questions were answered incorrectly at first, but checker feedback")
        print(f"  helped the solver improve and get the correct answer in later iterations.")
        for case in analytics['improved_cases'][:3]:  # Show first 3
            print(f"\n  Question #{case['question_num']}:")
            print(f"    First answer: {case['first_answer']} (WRONG)")
            print(f"    Final answer: {case['final_answer']} (CORRECT, matches GT: {case['ground_truth']})")
            print(f"    Improved at iteration {case['correct_at_iteration']} of {case['total_iterations']}")
    
    print(f"\n[TYPE 2] DEGRADED CASES (Initially Correct -> Later Wrong): {len(analytics['degraded_cases'])}")
    if analytics['degraded_cases']:
        print(f"  These questions were answered correctly at first, but checker")
        print(f"  misclassified them, leading to wrong answers in later iterations.")
        for case in analytics['degraded_cases'][:3]:  # Show first 3
            print(f"\n  Question #{case['question_num']}:")
            print(f"    First answer: {case['first_answer']} (CORRECT, matched GT: {case['ground_truth']})")
            print(f"    Final answer: {case['final_answer']} (WRONG)")
            print(f"    Degraded at iteration {case['wrong_at_iteration']} of {case['total_iterations']}")
    
    print(f"\n[TYPE 3] FIRST TRY SUCCESS (Correct on First Iteration): {len(analytics['first_try_success'])}")
    if analytics['first_try_success']:
        print(f"  These questions were solved correctly and efficiently on the first try.")
        print(f"  Questions: {', '.join(str(c['question_num']) for c in analytics['first_try_success'][:10])}")
        if len(analytics['first_try_success']) > 10:
            print(f"  ... and {len(analytics['first_try_success']) - 10} more")
    
    print(f"\n[TYPE 4] UNNECESSARY ITERATIONS (Correct Answer but Checker Missed It): {len(analytics['unnecessary_iterations'])}")
    if analytics['unnecessary_iterations']:
        print(f"  These questions had the correct answer, but checker didn't recognize it,")
        print(f"  leading to extra unnecessary iterations.")
        for case in analytics['unnecessary_iterations'][:3]:  # Show first 3
            print(f"\n  Question #{case['question_num']}:")
            print(f"    Got correct answer at iteration {case['correct_answer_at']}")
            print(f"    Checker verdict: {case['checker_verdict_at_correct']} (should have been CORRECT)")
            print(f"    Wasted {case['unnecessary_iterations_count']} extra iteration(s) (iterations {case['unnecessary_range']})")
            print(f"    Final: {case['final_answer']} ({'CORRECT' if case['final_correct'] else 'WRONG'})")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-agent evaluation results and generate reports"
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        help='Path to results directory (default: latest in results/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='summary',
        help='Output directory for CSV reports (default: summary/)'
    )
    
    args = parser.parse_args()
    
    # Find log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = find_latest_results_dir()
    
    print(f"Analyzing results from: {log_dir}")
    
    # Find log file
    log_file = log_dir / "log"
    log_files = list(log_file.glob("*.log"))
    if not log_files:
        print(f"Error: No log files found in {log_file}")
        return
    
    log_file_path = log_files[0]
    print(f"Reading log file: {log_file_path}")
    
    # Parse log file
    metadata, results = parse_log_file(log_file_path)
    print(f"Parsed {len(results)} samples")
    
    # Analyze results
    analytics = analyze_results(results)
    
    # Print report
    print_report(analytics, metadata)
    
    # Generate output filenames with metadata
    dataset = metadata.get('dataset', 'unknown').replace(' ', '_')
    model = metadata.get('model', 'unknown').replace(' ', '_').replace('/', '_')
    count = metadata.get('count', len(results))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_filename = f"{dataset}_{model}_{count}problems_{timestamp}"
    
    # Save single comprehensive CSV
    output_dir = Path(args.output_dir)
    analysis_csv = output_dir / f"{base_filename}_analysis.csv"
    
    save_analysis_csv(analytics, analysis_csv, metadata)
    
    print(f"\n[SUCCESS] Analysis complete! Report saved to: {analysis_csv}")
    print(f"  -> Type 1: {len(analytics['improved_cases'])} improved cases (initially wrong -> later correct)")
    print(f"  -> Type 2: {len(analytics['degraded_cases'])} degraded cases (initially correct -> later wrong)")
    print(f"  -> Type 3: {len(analytics['first_try_success'])} first try success (efficient)")
    print(f"  -> Type 4: {len(analytics['unnecessary_iterations'])} unnecessary iterations (checker missed correct answer)")


if __name__ == "__main__":
    main()

