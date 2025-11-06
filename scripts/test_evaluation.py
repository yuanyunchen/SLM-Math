import os
import sys
import time
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import extract_answer, check_answer

def test_single_model():
    print("Testing Evaluation Setup")
    print("=" * 60)
    
    model_name = "Qwen2.5-0.5B"
    model_dir = Path(f"../models/{model_name}")
    
    if not model_dir.exists():
        print(f"Model directory {model_dir} not found")
        return False
    
    model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not model_files:
        print(f"Model files not found in {model_dir}")
        return False
    
    print(f"\nLoading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    dataset_path = Path("../data/gsm8k")
    if not dataset_path.exists():
        print(f"✗ Dataset not found at {dataset_path}")
        return False
    
    print(f"\nLoading dataset: GSM8K")
    try:
        dataset = load_from_disk(str(dataset_path))
        test_data = dataset['test']
        print(f"✓ Dataset loaded: {len(test_data)} test samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False
    
    print(f"\nTesting inference on 3 samples...")
    test_samples = test_data.select(range(3))
    
    for idx, example in enumerate(test_samples):
        print(f"\n--- Sample {idx+1} ---")
        question = example['question']
        ground_truth = example['answer'].split('####')[-1].strip()
        
        print(f"Question: {question[:100]}...")
        print(f"Ground Truth: {ground_truth}")
        
        prompt = f"""Solve the following math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution:"""
        
        try:
            start_time = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            elapsed = time.time() - start_time
            
            predicted_answer = extract_answer(response)
            is_correct = check_answer(predicted_answer, ground_truth)
            
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {'✓' if is_correct else '✗'}")
            print(f"Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'=' * 60}")
    print("✓ Test evaluation completed successfully!")
    print("System is ready for full evaluation.")
    return True

if __name__ == "__main__":
    success = test_single_model()
    sys.exit(0 if success else 1)

