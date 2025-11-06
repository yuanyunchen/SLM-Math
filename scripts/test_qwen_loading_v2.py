#!/usr/bin/env python3
"""
Test Qwen3 with trust_remote_code and direct model class loading.
"""

import os
import torch

os.environ['HF_TOKEN'] = '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'

def test_with_auto_class():
    print("=" * 80)
    print("Test 1: Using AutoModel with trust_remote_code=True")
    print("=" * 80)
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model_id = "Qwen/Qwen3-0.6B"
        print(f"Loading {model_id}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN']
        )
        print("✓ Tokenizer loaded")
        
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN'],
            torch_dtype=torch.float32
        )
        print("✓ Model loaded")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_qwen2_5_math():
    print("\n" + "=" * 80)
    print("Test 2: Using Qwen2.5-Math-1.5B (designed for math)")
    print("=" * 80)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_id = "Qwen/Qwen2.5-Math-1.5B"
        print(f"Loading {model_id}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN']
        )
        print("✓ Tokenizer loaded")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=os.environ['HF_TOKEN'],
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        print("✓ Model loaded successfully!")
        
        # Quick test
        test_input = "What is 2+2? Answer: "
        inputs = tokenizer(test_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Test inference successful: {response}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result1 = test_with_auto_class()
    result2 = test_qwen2_5_math()
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Qwen3: {'✓ WORKS' if result1 else '✗ FAILED'}")
    print(f"  Qwen2.5-Math: {'✓ WORKS' if result2 else '✗ FAILED'}")
    print("=" * 80)

