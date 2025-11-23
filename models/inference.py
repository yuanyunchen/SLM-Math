"""
Model utilities for loading models and generating responses.
Supports both single and batch inference with transformers or vLLM backends.
"""

import re
import torch
from pathlib import Path
from typing import Union, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

MAX_TOKEN = 4096


class StopOnBoxedAnswer(StoppingCriteria):
    """Halts generation once a \\boxed{} answer is produced."""

    def __init__(self, tokenizer, prompt_token_len: int):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len

    def _has_boxed_answer(self, text: str) -> bool:
        """Check if text contains a complete \boxed{} with balanced braces."""
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return False
        
        # Count opening and closing braces after \boxed{
        after_boxed = text[idx + 7:]  # Skip "\boxed{"
        brace_count = 1  # Start with 1 for the opening brace of \boxed{
        
        for char in after_boxed:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    return True
        
        return False  # No matching closing brace found

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Decode the running hypothesis beyond the prompt so we react to newly created text only.
        generated_ids = input_ids[0][self.prompt_token_len :]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if self._has_boxed_answer(text):
            return True
        return False
    


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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()
    print(f"Model loaded successfully on {device.upper()}\n")
    
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

    # Stop decoding immediately after the boxed answer instead of wasting the full budget.
    stopping_criteria = StoppingCriteriaList([StopOnBoxedAnswer(tokenizer, prompt_length)])
    
    with torch.no_grad():
        outputs = model.generate(
            # **inputs,
            # max_new_tokens=MAX_TOKEN,
            # temperature=0.2,
            # top_p=0.3,
            # top_k=10,
            # do_sample=True,
            # repetition_penalty=1.1,
            # streamer=streamer,
            # eos_token_id=tokenizer.eos_token_id,
            # stopping_criteria=stopping_criteria,
            
            **inputs,
            max_new_tokens=MAX_TOKEN,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            # pad_token_id=tokenizer.pad_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            stopping_criteria=stopping_criteria,
            streamer=streamer
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response


def generate_response_batch(
    model, 
    tokenizer, 
    prompts: List[str], 
    mode: str, 
    detailed: bool = False
) -> List[str]:
    """
    Generate responses for a batch of prompts (transformers backend)
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompts: List of prompt strings
        mode: Generation mode (for compatibility)
        detailed: Whether to show detailed output
    
    Returns:
        List of response strings
    """
    if not prompts:
        return []
    
    # Tokenize all prompts with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    prompt_lengths = inputs['attention_mask'].sum(dim=1).tolist()
    
    # Note: Detailed mode (streaming) is not supported for batch inference
    if detailed:
        print("[Batch Mode] Detailed streaming output is disabled for batch processing")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKEN,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            # Stopping criteria for batch is complex, handled post-generation
        )
    
    # Decode responses
    responses = []
    for i, output_ids in enumerate(outputs):
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Try to remove prompt from response
        prompt = prompts[i]
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            # Fallback: extract by token length
            response_ids = output_ids[prompt_lengths[i]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        responses.append(response)
    
    return responses


def load_inference_engine_wrapper(
    model_name: str,
    base_path: Path,
    backend: str = "transformers",
    **kwargs
):
    """
    Load inference engine with specified backend
    
    Args:
        model_name: Name of the model
        base_path: Base path to project
        backend: 'transformers' or 'vllm'
        **kwargs: Additional parameters for engine
    
    Returns:
        Inference engine instance, or (model, tokenizer) for transformers
    """
    from models.inference_engine import load_inference_engine
    
    if backend.lower() == "vllm":
        # Check if vLLM is available
        from models.check_vllm import check_vllm_available
        if not check_vllm_available():
            print("WARNING: vLLM not available, falling back to transformers")
            backend = "transformers"
    
    engine = load_inference_engine(model_name, base_path, backend, **kwargs)
    
    # For transformers backend, return model and tokenizer for compatibility
    if backend.lower() == "transformers":
        return engine.get_model_and_tokenizer(), engine
    else:
        # For vLLM, return None for model and engine
        return (None, engine.tokenizer), engine
