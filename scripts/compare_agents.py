#!/usr/bin/env python3
"""
Agent Comparison and Analysis Script

This script compares results from multiple agent evaluations and identifies
improved/degraded cases across different methods.
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Compare agent evaluation results')
    parser.add_argument('--result_dirs', nargs='+', required=True, help='Result directories to compare')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for comparison')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    return parser.parse_args()


def load_agent_results(result_dir):
    """Load results from an agent evaluation directory"""
    result_dir = Path(result_dir)
    
    # Load metrics CSV
    metrics_csv = result_dir / "metrics.csv"
    if not metrics_csv.exists():
        print(f"Warning: No metrics.csv found in {result_dir}")
        return None
    
    metrics_df = pd.read_csv(metrics_csv)
    
    # Load answer JSON files
    answers_dir = result_dir / "answers"
    if not answers_dir.exists():
        # Try log directory as fallback
        answers_dir = result_dir / "log"
    
    answer_files = list(answers_dir.glob("*_answers.json"))
    if not answer_files:
        print(f"Warning: No answer JSON files found in {result_dir}")
        return None
    
    # Load the first answer file
    answer_file = answer_files[0]
    with open(answer_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return {
        'dir': result_dir,
        'agent': results.get('agent', 'unknown'),
        'metrics': metrics_df.iloc[0].to_dict() if len(metrics_df) > 0 else {},
        'results': results,
        'predictions': results.get('predictions', [])
    }


def compare_agents(agent_results_list, output_dir):
    """Compare multiple agent results and generate analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nComparing {len(agent_results_list)} agent evaluations...")
    print(f"Output directory: {output_dir}\n")
    
    # Create comparison dataframe
    comparison_data = []
    
    # Collect per-question results for detailed analysis
    question_results = defaultdict(dict)
    
    for agent_data in agent_results_list:
        agent_name = agent_data['agent']
        metrics = agent_data['metrics']
        
        # Add to comparison table
        comparison_data.append({
            'agent': agent_name,
            'accuracy': metrics.get('accuracy', 0) * 100,
            'first_try_accuracy': metrics.get('first_try_accuracy', 0) * 100,
            'improved_cases': metrics.get('improved_cases', 0),
            'degraded_cases': metrics.get('degraded_cases', 0),
            'failed_cases': metrics.get('failed_cases', 0),
            'avg_iterations': metrics.get('avg_iterations', 0),
            'avg_time_per_sample': metrics.get('avg_time_per_sample', 0),
            'total_samples': metrics.get('total_samples', 0)
        })
        
        # Collect per-question results
        for pred in agent_data['predictions']:
            q_id = pred['question_id']
            question_results[q_id][agent_name] = {
                'correct': pred.get('final_correct', False),
                'answer': pred.get('predicted_answer', ''),
                'first_correct': pred.get('first_correct', False),
                'case_type': pred.get('case_type', ''),
                'iterations': pred.get('total_iterations', 1)
            }
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    # Save comparison CSV
    comparison_csv = output_dir / "agent_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Saved comparison table: {comparison_csv}")
    
    # Generate detailed per-question comparison
    question_comparison = []
    for q_id, agent_answers in sorted(question_results.items()):
        row = {'question_id': q_id}
        
        # Check if different agents got different results
        correctness = [data['correct'] for data in agent_answers.values()]
        has_disagreement = len(set(correctness)) > 1
        
        row['has_disagreement'] = has_disagreement
        row['num_correct'] = sum(correctness)
        row['num_agents'] = len(agent_answers)
        
        # Add per-agent results
        for agent_name, data in agent_answers.items():
            row[f'{agent_name}_correct'] = data['correct']
            row[f'{agent_name}_answer'] = data['answer']
            row[f'{agent_name}_case_type'] = data['case_type']
        
        question_comparison.append(row)
    
    question_df = pd.DataFrame(question_comparison)
    question_csv = output_dir / "question_level_comparison.csv"
    question_df.to_csv(question_csv, index=False)
    print(f"Saved question-level comparison: {question_csv}")
    
    # Find questions where agents disagree
    disagreements = question_df[question_df['has_disagreement'] == True]
    print(f"\nFound {len(disagreements)} questions with disagreements across agents")
    
    # Generate text report
    generate_comparison_report(
        comparison_df, 
        question_df, 
        disagreements,
        agent_results_list,
        output_dir
    )
    
    # Generate visual comparison
    generate_visual_comparison(comparison_df, output_dir)
    
    return comparison_df, question_df


def generate_comparison_report(comparison_df, question_df, disagreements, agent_results_list, output_dir):
    """Generate detailed text report comparing agents"""
    report_file = output_dir / "comparison_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-AGENT COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Agents: {len(agent_results_list)}\n")
        f.write(f"Total Questions: {len(question_df)}\n\n")
        
        # Overall Rankings
        f.write("="*80 + "\n")
        f.write("AGENT PERFORMANCE RANKINGS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Ranked by Overall Accuracy:\n")
        f.write("-"*80 + "\n")
        for idx, row in comparison_df.iterrows():
            f.write(f"{idx+1}. {row['agent']:<40} {row['accuracy']:>6.2f}%\n")
        f.write("\n")
        
        # Detailed metrics table
        f.write("Detailed Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Agent':<30} {'Accuracy':<12} {'First Try':<12} {'Improved':<10} {'Degraded':<10} {'Avg Iter':<10}\n")
        f.write("-"*80 + "\n")
        
        for idx, row in comparison_df.iterrows():
            f.write(f"{row['agent']:<30} "
                   f"{row['accuracy']:>6.2f}%     "
                   f"{row['first_try_accuracy']:>6.2f}%     "
                   f"{int(row['improved_cases']):>4}       "
                   f"{int(row['degraded_cases']):>4}       "
                   f"{row['avg_iterations']:>6.2f}\n")
        f.write("\n")
        
        # Analysis of improvement patterns
        f.write("="*80 + "\n")
        f.write("IMPROVEMENT ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Questions where all agents succeeded
        all_correct = question_df[question_df['num_correct'] == question_df['num_agents']]
        f.write(f"Questions solved by ALL agents: {len(all_correct)} ({len(all_correct)/len(question_df)*100:.1f}%)\n")
        
        # Questions where all agents failed
        all_wrong = question_df[question_df['num_correct'] == 0]
        f.write(f"Questions failed by ALL agents: {len(all_wrong)} ({len(all_wrong)/len(question_df)*100:.1f}%)\n")
        
        # Questions with disagreement
        f.write(f"Questions with disagreement: {len(disagreements)} ({len(disagreements)/len(question_df)*100:.1f}%)\n\n")
        
        # Per-agent unique successes (questions only this agent got right)
        f.write("Unique Successes per Agent:\n")
        f.write("-"*80 + "\n")
        
        for agent_data in agent_results_list:
            agent_name = agent_data['agent']
            correct_col = f'{agent_name}_correct'
            if correct_col not in question_df.columns:
                continue
            
            # Find questions this agent got right but others got wrong
            unique_successes = question_df[
                (question_df[correct_col] == True) & 
                (question_df['num_correct'] < question_df['num_agents'])
            ]
            
            # Count how many are truly unique (only this agent)
            truly_unique = []
            for idx, row in unique_successes.iterrows():
                if row['num_correct'] == 1:  # Only this agent got it right
                    truly_unique.append(row)
            
            f.write(f"  {agent_name}: {len(truly_unique)} unique successes\n")
        f.write("\n")
        
        # Best and worst performing questions
        f.write("="*80 + "\n")
        f.write("QUESTION DIFFICULTY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Easiest questions (most agents solved)
        easiest = question_df.nlargest(5, 'num_correct')
        f.write("Easiest Questions (most agents solved):\n")
        f.write("-"*80 + "\n")
        for idx, row in easiest.iterrows():
            f.write(f"  Q{row['question_id']}: {row['num_correct']}/{row['num_agents']} agents correct\n")
        f.write("\n")
        
        # Hardest questions (fewest agents solved)
        hardest = question_df.nsmallest(5, 'num_correct')
        f.write("Hardest Questions (fewest agents solved):\n")
        f.write("-"*80 + "\n")
        for idx, row in hardest.iterrows():
            f.write(f"  Q{row['question_id']}: {row['num_correct']}/{row['num_agents']} agents correct\n")
        f.write("\n")
        
        # Disagreement examples
        if len(disagreements) > 0:
            f.write("="*80 + "\n")
            f.write("EXAMPLES OF AGENT DISAGREEMENTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Showing first 5 of {len(disagreements)} disagreements:\n\n")
            
            for idx, (_, row) in enumerate(disagreements.head(5).iterrows(), 1):
                f.write(f"Example {idx}: Question {row['question_id']}\n")
                f.write("-"*80 + "\n")
                f.write(f"  Agents correct: {row['num_correct']}/{row['num_agents']}\n")
                
                # Show which agents got it right/wrong
                for agent_data in agent_results_list:
                    agent_name = agent_data['agent']
                    correct_col = f'{agent_name}_correct'
                    answer_col = f'{agent_name}_answer'
                    case_col = f'{agent_name}_case_type'
                    
                    if correct_col in row and answer_col in row:
                        status = "CORRECT" if row[correct_col] else "WRONG"
                        case_type = row.get(case_col, 'N/A')
                        f.write(f"  {agent_name}: {status} (answer: {row[answer_col]}, type: {case_type})\n")
                f.write("\n")
        
        # Statistical summary
        f.write("="*80 + "\n")
        f.write("STATISTICAL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Accuracy Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Mean Accuracy: {comparison_df['accuracy'].mean():.2f}%\n")
        f.write(f"  Median Accuracy: {comparison_df['accuracy'].median():.2f}%\n")
        f.write(f"  Std Dev: {comparison_df['accuracy'].std():.2f}%\n")
        f.write(f"  Best: {comparison_df['accuracy'].max():.2f}% ({comparison_df.loc[comparison_df['accuracy'].idxmax(), 'agent']})\n")
        f.write(f"  Worst: {comparison_df['accuracy'].min():.2f}% ({comparison_df.loc[comparison_df['accuracy'].idxmin(), 'agent']})\n")
        f.write(f"  Range: {comparison_df['accuracy'].max() - comparison_df['accuracy'].min():.2f}%\n\n")
        
        f.write("Iteration Statistics (for iterative agents):\n")
        f.write("-"*80 + "\n")
        iterative = comparison_df[comparison_df['avg_iterations'] > 1.0]
        if len(iterative) > 0:
            f.write(f"  Mean Avg Iterations: {iterative['avg_iterations'].mean():.2f}\n")
            f.write(f"  Max Avg Iterations: {iterative['avg_iterations'].max():.2f} ({iterative.loc[iterative['avg_iterations'].idxmax(), 'agent']})\n")
            f.write(f"  Min Avg Iterations: {iterative['avg_iterations'].min():.2f} ({iterative.loc[iterative['avg_iterations'].idxmin(), 'agent']})\n")
        else:
            f.write("  No iterative agents found\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Saved comparison report: {report_file}")


def generate_visual_comparison(comparison_df, output_dir):
    """Generate visual comparison (simple text-based bar chart)"""
    viz_file = output_dir / "visual_comparison.txt"
    
    with open(viz_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VISUAL COMPARISON - ACCURACY BAR CHART\n")
        f.write("="*80 + "\n\n")
        
        max_agent_len = comparison_df['agent'].str.len().max()
        
        for idx, row in comparison_df.iterrows():
            agent_name = row['agent']
            accuracy = row['accuracy']
            
            # Create bar
            bar_len = int(accuracy / 2)  # Scale to fit in terminal (50 chars = 100%)
            bar = '█' * bar_len
            
            f.write(f"{agent_name:<{max_agent_len}} | {bar} {accuracy:.2f}%\n")
        
        f.write("\n")
        f.write("="*80 + "\n\n")
        
        # Improvement comparison
        f.write("IMPROVEMENT vs DEGRADATION\n")
        f.write("="*80 + "\n\n")
        
        for idx, row in comparison_df.iterrows():
            agent_name = row['agent']
            improved = int(row['improved_cases'])
            degraded = int(row['degraded_cases'])
            
            imp_bar = '▲' * min(improved, 40)
            deg_bar = '▼' * min(degraded, 40)
            
            f.write(f"{agent_name}\n")
            f.write(f"  Improved:  {imp_bar} {improved}\n")
            f.write(f"  Degraded:  {deg_bar} {degraded}\n")
            f.write(f"  Net:       {improved - degraded:+d}\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Saved visual comparison: {viz_file}")


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("Agent Comparison Analysis")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Result directories: {len(args.result_dirs)}")
    print("="*80 + "\n")
    
    # Load all agent results
    agent_results_list = []
    for result_dir in args.result_dirs:
        print(f"Loading results from: {result_dir}")
        agent_data = load_agent_results(result_dir)
        if agent_data:
            agent_results_list.append(agent_data)
            print(f"  Agent: {agent_data['agent']}")
            print(f"  Accuracy: {agent_data['metrics'].get('accuracy', 0)*100:.2f}%")
        else:
            print(f"  Failed to load results")
        print()
    
    if len(agent_results_list) == 0:
        print("ERROR: No valid agent results found!")
        return 1
    
    # Compare agents
    comparison_df, question_df = compare_agents(agent_results_list, args.output_dir)
    
    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print()
    
    # Print quick summary
    print("Quick Summary:")
    print("-"*80)
    print(f"{'Agent':<40} {'Accuracy':<12}")
    print("-"*80)
    for idx, row in comparison_df.iterrows():
        print(f"{row['agent']:<40} {row['accuracy']:>6.2f}%")
    print("="*80)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


