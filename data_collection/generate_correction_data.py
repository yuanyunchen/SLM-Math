"""
Generate Self-Correction Training Data using API

This script:
1. Loads math problems from GSM8K train set
2. Calls API (Grok/OpenAI) to generate self-correction trajectories
3. Saves the generated data for fine-tuning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import argparse
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import time

from data_collection.self_correction_prompts import (
    SYSTEM_PROMPT, TEMPLATES, get_prompt
)


def load_gsm8k_train():
    """Load GSM8K training set."""
    from datasets import load_from_disk
    ds = load_from_disk("data/gsm8k")
    return ds['train']


def extract_answer_from_gsm8k(solution: str) -> str:
    """Extract numerical answer from GSM8K solution."""
    import re
    # GSM8K answers are after ####
    match = re.search(r'####\s*(.+)$', solution, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def generate_single(
    client,
    model: str,
    question: str,
    answer: str,
    error_type: str
) -> Dict:
    """Generate a single correction trajectory."""
    
    prompt = get_prompt(error_type, question, answer)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        generated = response.choices[0].message.content
        
        return {
            'success': True,
            'query': question,
            'response': generated,
            'ground_truth': answer,
            'error_type': error_type
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'query': question,
            'ground_truth': answer,
            'error_type': error_type
        }


def generate_batch(
    api_key: str,
    base_url: str,
    model: str,
    n_samples: int = 1000,
    output_file: str = "data/correction_trajectories.json",
    error_distribution: Dict[str, float] = None
):
    """
    Generate batch of correction trajectories.
    
    Args:
        api_key: API key
        base_url: API base URL (e.g., https://api.x.ai/v1)
        model: Model name (e.g., grok-3-fast)
        n_samples: Number of samples to generate
        output_file: Output file path
        error_distribution: Distribution of error types (default: weighted by real errors)
    """
    from openai import OpenAI
    
    # Default distribution based on real error analysis
    if error_distribution is None:
        error_distribution = {
            'logic_error': 0.50,      # Most common
            'name_error': 0.20,       # Common in MATH500
            'wrong_output': 0.15,     # Variation of logic error
            'syntax_error': 0.05,
            'type_error': 0.05,
            'multi_step': 0.05,       # Advanced: multiple corrections
        }
    
    print(f"Loading GSM8K training data...")
    gsm8k = load_gsm8k_train()
    print(f"Loaded {len(gsm8k)} problems")
    
    # Sample problems
    indices = random.sample(range(len(gsm8k)), min(n_samples, len(gsm8k)))
    
    print(f"\nInitializing API client...")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Determine error types for each sample
    error_types = []
    for _ in range(n_samples):
        r = random.random()
        cumulative = 0
        for error_type, prob in error_distribution.items():
            cumulative += prob
            if r < cumulative:
                error_types.append(error_type)
                break
    
    print(f"\nError type distribution:")
    from collections import Counter
    for et, count in Counter(error_types).most_common():
        print(f"  {et}: {count}")
    
    # Generate
    results = []
    success_count = 0
    
    print(f"\nGenerating {n_samples} correction trajectories...")
    
    for i, idx in enumerate(tqdm(indices)):
        item = gsm8k[idx]
        question = item['question']
        answer = extract_answer_from_gsm8k(item['answer'])
        error_type = error_types[i]
        
        result = generate_single(client, model, question, answer, error_type)
        results.append(result)
        
        if result['success']:
            success_count += 1
        
        # Rate limiting
        time.sleep(0.5)
        
        # Save checkpoint every 100 samples
        if (i + 1) % 100 == 0:
            checkpoint_file = output_file.replace('.json', f'_checkpoint_{i+1}.json')
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nCheckpoint saved: {checkpoint_file}")
            print(f"Progress: {success_count}/{i+1} successful")
    
    # Save final results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Generation Complete ===")
    print(f"Total: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Saved to: {output_file}")
    
    return results


def convert_to_sft_format(
    input_file: str,
    output_file: str,
    format_type: str = "chatml"
):
    """Convert generated data to SFT training format."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter successful generations
    successful = [d for d in data if d.get('success', False)]
    print(f"Successful generations: {len(successful)}/{len(data)}")
    
    formatted = []
    for item in successful:
        if format_type == "chatml":
            formatted.append({
                "messages": [
                    {"role": "user", "content": item['query']},
                    {"role": "assistant", "content": item['response']}
                ]
            })
        else:
            formatted.append({
                "query": item['query'],
                "response": item['response']
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(formatted)} samples to: {output_file}")
    
    return formatted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate self-correction training data")
    parser.add_argument("--api-key", type=str, required=True, help="API key")
    parser.add_argument("--base-url", type=str, default="https://api.x.ai/v1", help="API base URL")
    parser.add_argument("--model", type=str, default="grok-3-fast", help="Model name")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/agent_sft/correction_trajectories.json")
    
    args = parser.parse_args()
    
    generate_batch(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        n_samples=args.n_samples,
        output_file=args.output
    )



