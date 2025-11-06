import os
import sys
import time
import json
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils import extract_answer, check_answer, save_results

MODELS = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B"
]

DATASETS = {
    "gsm8k": "../data/gsm8k",
    "math": "../data/math"
}

MAX_SAMPLES = 50  # Evaluate 50 samples per dataset

def format_prompt(question: str, dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        prompt = f"""Solve the following math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution:"""
    else:
        prompt = f"""Solve the following math problem. Provide a detailed solution and your final answer.

Problem: {question}

Solution:"""
    
    return prompt

def evaluate_model_on_dataset(model_name: str, dataset_name: str, dataset_path: str):
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on {dataset_name.upper()}")
    print(f"{'='*80}")
    
    model_dir = Path(f"../models/{model_name}")
    if not model_dir.exists():
        print(f"Model directory {model_dir} not found. Skipping...")
        return None
    
    model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not model_files:
        print(f"Model files not found in {model_dir} (download may be incomplete). Skipping...")
        return None
    
    print(f"Loading model from {model_dir}...")
    device = torch.device("cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            dtype=torch.float32,
            device_map="cpu"
        )
        model.eval()
        print(f"Model loaded on device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
        
        # Try 'test' split first, fall back to 'train' if not available
        if 'test' in dataset:
            test_data = dataset['test']
            print(f"Using 'test' split: {len(test_data)} samples")
        elif 'train' in dataset:
            test_data = dataset['train']
            print(f"Using 'train' split (no 'test' available): {len(test_data)} samples")
        else:
            print(f"Error: No 'test' or 'train' split found in dataset")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    num_samples = min(MAX_SAMPLES, len(test_data))
    print(f"Testing on {num_samples} samples...")
    
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
        "total_time": 0.0,
        "avg_time_per_sample": 0.0,
        "predictions": []
    }
    
    start_time = time.time()
    
    for idx, example in enumerate(tqdm(test_data.select(range(num_samples)), desc="Evaluating")):
        # Handle different dataset schemas
        if dataset_name == "gsm8k":
            question = example['question']
            ground_truth = example['answer'].split('####')[-1].strip()
        elif dataset_name == "math":
            # MATH dataset has 'problem' and 'solution' fields
            question = example['problem']
            # Extract the boxed answer from solution, or use the whole solution
            solution = example['solution']
            # Try to extract \boxed{...} answer from solution
            import re
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
            if boxed_match:
                ground_truth = boxed_match.group(1)
            else:
                # Fall back to last number in solution
                numbers = re.findall(r'[+-]?\d+\.?\d*', solution)
                ground_truth = numbers[-1] if numbers else solution
        else:
            question = example.get('question', example.get('problem', ''))
            ground_truth = example.get('answer', example.get('solution', ''))
        
        prompt = format_prompt(question, dataset_name)
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            predicted_answer = extract_answer(response)
            is_correct = check_answer(predicted_answer, ground_truth)
            
            if is_correct:
                results["correct"] += 1
            
            results["total"] += 1
            
            results["predictions"].append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "response": response[:200],
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            results["total"] += 1
            results["predictions"].append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": None,
                "response": f"Error: {e}",
                "correct": False
            })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    results["total_time"] = total_time
    results["avg_time_per_sample"] = total_time / results["total"] if results["total"] > 0 else 0
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    
    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"  Total Time: {results['total_time']:.2f} seconds")
    print(f"  Avg Time/Sample: {results['avg_time_per_sample']:.2f} seconds")
    
    output_dir = Path(f"../results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_{dataset_name}_results.json"
    save_results(results, str(output_file))
    
    del model
    torch.cuda.empty_cache()
    
    return results

def generate_summary_report(all_results):
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    
    summary_file = Path("../results/summary_report.txt")
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SLM-Math Evaluation Report\n")
        f.write("="*80 + "\n\n")
        
        for result in all_results:
            if result is None:
                continue
            
            report = f"""
Model: {result['model']}
Dataset: {result['dataset'].upper()}
{'='*80}
Samples Evaluated: {result['total']}
Correct: {result['correct']}
Accuracy: {result['accuracy']*100:.2f}%
Total Time: {result['total_time']:.2f} seconds
Avg Time per Sample: {result['avg_time_per_sample']:.2f} seconds
{'='*80}

"""
            print(report)
            f.write(report)
        
        print(f"\nDetailed results saved in ../results/metrics/")
        print(f"Summary report saved to {summary_file}")

def main():
    print("Starting SLM-Math Evaluation")
    print("="*80)
    
    print("\nChecking available models...")
    available_models = []
    for model_name in MODELS:
        model_dir = Path(f"../models/{model_name}")
        model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if model_files:
            available_models.append(model_name)
            print(f"  ✓ {model_name}")
        else:
            print(f"  ✗ {model_name} (not yet downloaded)")
    
    if not available_models:
        print("\n⚠ No models available for evaluation. Please download models first.")
        return
    
    print(f"\nEvaluating {len(available_models)} model(s)...")
    
    all_results = []
    
    for model_name in available_models:
        for dataset_name, dataset_path in DATASETS.items():
            result = evaluate_model_on_dataset(model_name, dataset_name, dataset_path)
            if result:
                all_results.append(result)
    
    if all_results:
        generate_summary_report(all_results)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()

