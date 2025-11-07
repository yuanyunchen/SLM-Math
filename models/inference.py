"""
Model utilities for loading models and generating responses.
"""

import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MAX_TOKEN = 2048



def load_model(model_name: str, base_path: Path):
    """Load model and tokenizer from disk"""
    model_dir = base_path / 'pretrained_models' / model_name
    
    # Check model exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} not found!")
    
    model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}!")
    
    # Load model
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True
    )
    
    # Ensure pad_token is set (use eos_token if pad_token is None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    print("Model loaded successfully on CPU\n")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, mode: str, detailed: bool = False):
    """Generate response from model given a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    prompt_length = inputs['input_ids'].shape[1]
        
    # Create streamer for real-time token-by-token output (only if detailed)
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKEN,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            # pad_token_id=tokenizer.pad_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            # stopping_criteria=stopping_criteria,
            streamer=streamer
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response

