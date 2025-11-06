import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm

def preprocess_gsm8k_to_csv():
    """Convert GSM8K dataset to human-readable CSV format."""
    print("Processing GSM8K dataset...")
    
    dataset_path = Path("../data/gsm8k")
    output_dir = Path("../data/csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_from_disk(str(dataset_path))
    
    for split_name in ['train', 'test']:
        split_data = dataset[split_name]
        
        rows = []
        for idx, example in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            question_id = f"GSM8K_{split_name}_{idx+1}"
            question = example['question']
            full_answer = example['answer']
            
            if '####' in full_answer:
                solution, final_answer = full_answer.split('####')
                solution = solution.strip()
                final_answer = final_answer.strip()
            else:
                solution = full_answer
                final_answer = ""
            
            rows.append({
                'question_id': question_id,
                'dataset': 'GSM8K',
                'split': split_name,
                'question': question,
                'solution': solution,
                'final_answer': final_answer,
                'difficulty': 'elementary',
                'topic': 'arithmetic'
            })
        
        df = pd.DataFrame(rows)
        output_file = output_dir / f"gsm8k_{split_name}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ Saved {len(df)} {split_name} samples to {output_file}")
    
    return True

def preprocess_math_to_csv():
    """Convert MATH dataset to human-readable CSV format."""
    print("\nProcessing MATH dataset...")
    
    # Try Hugging Face format first
    math_path_hf = Path("../data/math")
    if math_path_hf.exists():
        print(f"Loading MATH dataset from Hugging Face format: {math_path_hf}")
        try:
            dataset = load_from_disk(str(math_path_hf))
            output_dir = Path("../data/csv")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process train split
            if 'train' in dataset:
                split_data = dataset['train']
                rows = []
                
                for idx, example in enumerate(tqdm(split_data, desc="Processing MATH train")):
                    question_id = f"MATH_train_{idx+1}"
                    problem = example.get('problem', '')
                    solution = example.get('solution', '')
                    answer = example.get('answer', '')
                    level = example.get('level', '')
                    topic = example.get('type', '')
                    
                    rows.append({
                        'question_id': question_id,
                        'dataset': 'MATH',
                        'split': 'train',
                        'topic': topic,
                        'difficulty': level,
                        'question': problem,
                        'solution': solution,
                        'final_answer': answer
                    })
                
                df = pd.DataFrame(rows)
                output_file = output_dir / "math_train.csv"
                df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"✓ Saved {len(df)} train samples to {output_file}")
                return True
            else:
                print("✗ No 'train' split found in MATH dataset")
                return False
                
        except Exception as e:
            print(f"✗ Failed to load MATH dataset from {math_path_hf}: {e}")
            return False
    
    # Try GitHub format as fallback
    math_path = Path("../data/MATH")
    if not math_path.exists():
        print(f"✗ MATH dataset not found at {math_path_hf} or {math_path}")
        print("  Please download from: https://github.com/hendrycks/math")
        return False
    
    output_dir = Path("../data/csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': math_path / 'train',
        'test': math_path / 'test'
    }
    
    for split_name, split_path in splits.items():
        if not split_path.exists():
            print(f"✗ Split {split_name} not found at {split_path}")
            continue
        
        rows = []
        idx = 0
        
        for topic_dir in sorted(split_path.iterdir()):
            if not topic_dir.is_dir():
                continue
            
            topic = topic_dir.name
            
            for json_file in sorted(topic_dir.glob('*.json')):
                idx += 1
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                question_id = f"MATH_{split_name}_{idx}"
                problem = data.get('problem', '')
                solution = data.get('solution', '')
                answer = data.get('answer', '')
                level = data.get('level', '')
                
                problem_clean = problem.replace('\\n', '\n').replace('\\t', '\t')
                solution_clean = solution.replace('\\n', '\n').replace('\\t', '\t')
                answer_clean = answer.replace('\\n', ' ').replace('\\t', ' ')
                
                rows.append({
                    'question_id': question_id,
                    'dataset': 'MATH',
                    'split': split_name,
                    'topic': topic,
                    'difficulty': level,
                    'question': problem_clean,
                    'solution': solution_clean,
                    'final_answer': answer_clean
                })
        
        if rows:
            df = pd.DataFrame(rows)
            output_file = output_dir / f"math_{split_name}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✓ Saved {len(df)} {split_name} samples to {output_file}")
    
    return True

def create_combined_csv():
    """Create combined dataset CSV for easy searching."""
    print("\nCreating combined dataset...")
    
    csv_dir = Path("../data/csv")
    
    all_dfs = []
    for csv_file in csv_dir.glob("*.csv"):
        if 'combined' not in csv_file.name:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        output_file = csv_dir / "combined_dataset.csv"
        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ Created combined dataset with {len(combined_df)} total samples")
        print(f"  File: {output_file}")
        
        print("\nDataset Statistics:")
        print(combined_df.groupby(['dataset', 'split']).size())
    
    return True

def main():
    print("="*60)
    print("Dataset Preprocessing to CSV")
    print("="*60)
    
    gsm8k_success = preprocess_gsm8k_to_csv()
    math_success = preprocess_math_to_csv()
    
    if gsm8k_success or math_success:
        create_combined_csv()
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("CSV files saved in: ../data/csv/")
    print("="*60)

if __name__ == "__main__":
    main()

