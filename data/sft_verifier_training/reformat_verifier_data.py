"""
Reformat verifier training data for SFT training.
Input: Question + Solution (CoT + Code + Answer)
Output: Verdict (CORRECT/INCORRECT/UNCLEAR)
"""

import json
import csv

def reformat_for_training(input_jsonl, output_jsonl, output_csv):
    """
    Reformat verifier data into training format:
    - input: question + solution (reasoning + code + answer)
    - output: verdict only (CORRECT/INCORRECT/UNCLEAR)
    """
    
    training_examples = []
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            
            # Construct the input: question + solution
            input_text = f"""Question: {example['question']}

Solution:
{example['solution']}"""
            
            # Output is just the verdict
            output_text = example['verdict']
            
            training_example = {
                'index': example['index'],
                'input': input_text,
                'output': output_text,
                'question': example['question'],
                'ground_truth': example['ground_truth'],
                'verdict': example['verdict'],
                'category': example['category']
            }
            
            training_examples.append(training_example)
    
    # Save to JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Save to CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['index', 'input', 'output', 'question', 'ground_truth', 'verdict', 'category']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for example in training_examples:
            writer.writerow(example)
    
    return training_examples

def main():
    input_file = 'data/sft_verifier_training/sft_verifier_training_data.jsonl'
    output_jsonl = 'data/sft_verifier_training/sft_verifier_training_formatted.jsonl'
    output_csv = 'data/sft_verifier_training/sft_verifier_training_formatted.csv'
    
    print("Reformatting verifier training data...")
    examples = reformat_for_training(input_file, output_jsonl, output_csv)
    
    print(f"\n[SUCCESS] Reformatted {len(examples)} examples")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Output CSV: {output_csv}")
    
    # Show sample
    print("\n=== Sample Training Example ===")
    sample = examples[0]
    print(f"\nINPUT:")
    print(sample['input'][:400] + "..." if len(sample['input']) > 400 else sample['input'])
    print(f"\nOUTPUT:")
    print(sample['output'])
    print(f"\nCategory: {sample['category']}")
    
    # Show statistics
    print("\n=== Statistics ===")
    verdict_counts = {}
    for ex in examples:
        verdict_counts[ex['verdict']] = verdict_counts.get(ex['verdict'], 0) + 1
    
    for verdict, count in sorted(verdict_counts.items()):
        print(f"{verdict}: {count} ({count/len(examples)*100:.1f}%)")

if __name__ == '__main__':
    main()

