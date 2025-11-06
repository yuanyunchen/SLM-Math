#!/usr/bin/env python3
"""
Test loading Qwen3 models directly from Hugging Face Hub.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_TOKEN'] = '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'

def test_model(model_id):
    print(f"\n{'='*80}")
    print(f"Testing {model_id}")
    print(f"{'='*80}")
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN']
        )
        print(f"✓ Tokenizer loaded from Hub")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    
    print("Testing tokenization...")
    try:
        test_prompt = "What is 2+2?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        print(f"✓ Tokenization works")
        print(f"  Input shape: {inputs['input_ids'].shape}")
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        return False
    
    return True

def main():
    models = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-4B-Thinking-2507"
    ]
    
    print("Testing Qwen3 Models from Hugging Face Hub")
    print("=" * 80)
    
    results = {}
    for model_id in models:
        results[model_id] = test_model(model_id)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for model_id, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_id}")

if __name__ == "__main__":
    main()

