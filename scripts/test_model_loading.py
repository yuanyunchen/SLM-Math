#!/usr/bin/env python3
"""
Test script to check if Qwen3 models can be loaded and used for inference.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading(model_name):
    print(f"\n{'='*80}")
    print(f"Testing {model_name}")
    print(f"{'='*80}")
    
    model_dir = Path(f"../models/{model_name}")
    
    if not model_dir.exists():
        print(f"✗ Model directory {model_dir} not found")
        return False
    
    print(f"✓ Model directory found: {model_dir}")
    
    # Check for model files
    model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not model_files:
        print(f"✗ No model files found in {model_dir}")
        return False
    
    print(f"✓ Found model files: {[f.name for f in model_files]}")
    
    # Try loading tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    
    # Try loading model
    print("\nLoading model...")
    device = torch.device("cpu")
    
    try:
        # Try with torch_dtype=torch.float32 for CPU
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully on {device}")
        print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try a simple inference test
    print("\nTesting inference...")
    try:
        test_prompt = "What is 2+2? Answer: "
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=1,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Inference successful")
        print(f"  Prompt: {test_prompt}")
        print(f"  Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    models = [
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-4B-Thinking-2507"
    ]
    
    print("=" * 80)
    print("Testing Qwen3 Model Loading and Inference")
    print("=" * 80)
    
    results = {}
    for model_name in models:
        results[model_name] = test_model_loading(model_name)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_name}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

