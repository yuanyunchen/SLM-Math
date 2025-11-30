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
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY,
    MAX_TOKEN  # Legacy alias for backward compatibility
)


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
    


def load_model(model_name: str, base_path: Path, checkpoint_path: str = None):
    """
    Load model and tokenizer from disk
    
    Args:
        model_name: Name of the base model
        base_path: Base path to project
        checkpoint_path: Optional path to checkpoint (LoRA adapter or fine-tuned model)
                        Can be relative to base_path or absolute path
    
    Returns:
        model, tokenizer
    """
    # Determine model directory
    if checkpoint_path:
        # If checkpoint path is provided, use it
        if Path(checkpoint_path).is_absolute():
            model_dir = Path(checkpoint_path)
        else:
            model_dir = base_path / checkpoint_path
        
        print(f"Loading from checkpoint: {model_dir}")
        
        # Check if this is a LoRA adapter (has adapter_config.json)
        is_lora = (model_dir / "adapter_config.json").exists()
        
        if is_lora:
            print("Detected LoRA adapter checkpoint")
            # For LoRA, we need to load base model first, then adapter
            base_model_dir = base_path / 'pretrained_models' / model_name
            
            if not base_model_dir.exists():
                raise FileNotFoundError(f"Base model directory {base_model_dir} not found!")
            
            print(f"Loading base model from {base_model_dir}...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(base_model_dir),
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_dir),
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            
            print(f"Loading LoRA adapter from {model_dir}...")
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, str(model_dir))
                model.eval()
                print(f"LoRA adapter loaded successfully on {device.upper()}\n")
            except ImportError:
                raise ImportError("peft library is required to load LoRA adapters. Install with: pip install peft")
            
            return model, tokenizer
        else:
            print("Detected full fine-tuned model checkpoint")
            # Regular fine-tuned model
            if not model_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory {model_dir} not found!")
            
            model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
            if not model_files:
                raise FileNotFoundError(f"No model files found in {model_dir}!")
    else:
        # No checkpoint, load from pretrained_models
        model_dir = base_path / 'pretrained_models' / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} not found!")
        
        model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}!")
        
        print(f"Loading model from {model_dir}...")
    
    # Load tokenizer and model (for non-LoRA cases)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True
    )
    
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


def generate_response(model, tokenizer, prompt: str, mode: str, detailed: bool = False, greedy: bool = True):
    """Generate response from model given a prompt
    
    Args:
        model: The model or inference engine
        tokenizer: The tokenizer
        prompt: Input prompt string
        mode: Generation mode (for compatibility)
        detailed: Whether to show detailed output
        greedy: If True, use greedy decoding (do_sample=False), ignoring temperature/top_p/etc.
    """
    
    # Check if using inference engine
    if hasattr(model, 'generate_single'):
        # Using inference engine (vLLM or TransformersEngine)
        if greedy:
            return model.generate_single(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1.0,  # Ignored when do_sample=False
                do_sample=False,
                detailed=detailed
            )
        else:
            return model.generate_single(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                detailed=detailed
            )
    
    # Using standard PyTorch model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Get model device properly
    try:
        if hasattr(model, 'device'):
            model_device = model.device
        else:
            model_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    prompt_length = inputs['input_ids'].shape[1]
        
    # Create streamer for real-time token-by-token output (only if detailed)
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None

    # Stop decoding immediately after the boxed answer instead of wasting the full budget.
    stopping_criteria = StoppingCriteriaList([StopOnBoxedAnswer(tokenizer, prompt_length)])
    
    with torch.no_grad():
        if greedy:
            # Greedy decoding: ignore temperature/top_p/etc.
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                streamer=streamer
            )
        else:
            # Sampling with configured parameters
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=REPETITION_PENALTY,
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
    detailed: bool = False,
    greedy: bool = True
) -> List[str]:
    """
    Generate responses for a batch of prompts
    
    Args:
        model: The loaded model or inference engine
        tokenizer: The loaded tokenizer
        prompts: List of prompt strings
        mode: Generation mode (for compatibility)
        detailed: Whether to show detailed output
        greedy: If True, use greedy decoding (do_sample=False), ignoring temperature/top_p/etc.
    
    Returns:
        List of response strings
    """
    if not prompts:
        return []
    
    # Check if using inference engine
    if hasattr(model, 'generate_batch'):
        # Using inference engine (vLLM or TransformersEngine)
        if greedy:
            return model.generate_batch(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1.0,  # Ignored when do_sample=False
                do_sample=False,
                detailed=detailed
            )
        else:
            return model.generate_batch(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                detailed=detailed
            )
    
    # Using standard PyTorch model
    # Tokenize all prompts with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Get model device properly
    try:
        if hasattr(model, 'device'):
            model_device = model.device
        else:
            model_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    prompt_lengths = inputs['attention_mask'].sum(dim=1).tolist()
    
    # Note: Detailed mode (streaming) is not supported for batch inference
    if detailed:
        print("[Batch Mode] Detailed streaming output is disabled for batch processing")
    
    with torch.no_grad():
        if greedy:
            # Greedy decoding: ignore temperature/top_p/etc.
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            # Sampling with configured parameters
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=REPETITION_PENALTY,
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
    checkpoint_path: str = None,
    **kwargs
):
    """
    Load inference engine with specified backend
    
    Args:
        model_name: Name of the model
        base_path: Base path to project
        backend: 'transformers' or 'vllm'
        checkpoint_path: Optional path to checkpoint
        **kwargs: Additional parameters for engine
    
    Returns:
        Inference engine instance, or (model, tokenizer) for transformers
    """
    # If checkpoint is provided, force use of standard load_model (no inference engine)
    # This is because checkpoints (especially LoRA) are not yet supported by vLLM
    if checkpoint_path:
        if backend.lower() == "vllm":
            print("WARNING: Checkpoints are not supported with vLLM backend, falling back to transformers")
            print(f"Loading checkpoint: {checkpoint_path}")
        
        model, tokenizer = load_model(model_name, base_path, checkpoint_path)
        # Return None for engine to use standard transformers generation
        return (model, tokenizer), None
    
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
