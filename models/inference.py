"""
Model utilities for loading models and generating responses.
"""

import re
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

MAX_TOKEN = 2048  # Default max tokens for generation


class StopOnBoxedAnswer(StoppingCriteria):
    """Halts generation once a \\boxed{} answer is produced."""

    def __init__(self, tokenizer, prompt_token_len: int):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len

    def _has_boxed_answer(self, text: str) -> bool:
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return False
        return "}" in text[idx:]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Decode the running hypothesis beyond the prompt so we react to newly created text only.
        generated_ids = input_ids[0][self.prompt_token_len :]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if self._has_boxed_answer(text):
            return True
        return False


class StopAfterCheckerConclusion(StoppingCriteria):
    """
    Halts checker generation once a VERDICT line is produced.
    
    The checker should output: VERDICT: CORRECT / INCORRECT / UNCLEAR
    We stop as soon as we detect a complete VERDICT line with one of these outcomes.
    """

    def __init__(self, tokenizer, prompt_token_len: int):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_token_len :]
        if generated_ids.numel() == 0:
            return False

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        upper = text.upper()

        # Check for complete VERDICT pattern: VERDICT: CORRECT/INCORRECT/UNCLEAR
        # We look for the pattern and stop once we see it
        verdict_pattern = r'VERDICT\s*:\s*(CORRECT|INCORRECT|UNCLEAR)'
        match = re.search(verdict_pattern, upper)
        
        if match:
            # Found a complete verdict, stop generation immediately
            return True
        
        # Also stop if we see verdict words (with possible typos) as standalone
        if len(text.strip()) > 5:  # After some minimal generation
            # Look for these words (with possible typos) at the start or after newline/punctuation
            # Handle common typos like CORREECT
            if re.search(r'(^|\n|\s)(CORRE+CT|INCORREE+CT|UNCLEAR)(\s|$|\n)', upper):
                return True
            if re.search(r'(^|\n|\s)(COR+ECT|INCOR+ECT)(\s|$|\n)', upper):
                return True
        
        # Stop if response is getting too long (>150 chars) to prevent rambling
        if len(text) > 150:
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
    
    # Prefer CUDA when available. Use float16 on CUDA for memory savings.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Choose device_map for transformers. For single-GPU, using "cuda" works.
    device_map = "cuda" if device == "cuda" else "cpu"

    # Log CUDA info for user visibility
    if device == "cuda":
        try:
            cuda_count = torch.cuda.device_count()
            cuda_name = torch.cuda.get_device_name(0) if cuda_count > 0 else "<unknown>"
        except Exception:
            cuda_count = 0
            cuda_name = "<unknown>"
        print(f"CUDA available: True | GPU count: {cuda_count} | Using device: {cuda_name}")
    else:
        print("CUDA available: False | Using CPU for inference")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    print(f"Model loaded successfully (device_map={device_map}, dtype={torch_dtype})\n")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, mode: str, detailed: bool = False):
    """
    Generate response from model given a prompt.

    The `mode` argument is used to select decoding behavior and stopping criteria:
      - "standard", "thinking", "solver": stop when a \\boxed{} answer appears,
        with a larger max_new_tokens budget.
      - "checker": use a custom stopping rule that ends after a VERDICT line is found.
    """
    # Don't truncate prompts/responses - allow full length
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)

    # Determine the device where the model's parameters live. This is more reliable
    # than `model.device` when device_map places parameters on specific CUDA devices.
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        # Fallback if model has no parameters (unlikely)
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    prompt_length = inputs['input_ids'].shape[1]
        
    # Create streamer for real-time token-by-token output (only if detailed)
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None

    # Configure stopping criteria and decoding parameters based on mode
    stopping_criteria = StoppingCriteriaList()
    
    # Default generation kwargs
    gen_kwargs = {
        "temperature": 0.1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,
        "streamer": streamer,
    }

    # Get model's maximum context length for max_new_tokens calculation
    # Use MAX_TOKEN as default, but allow more if model supports it
    model_max_length = getattr(tokenizer, 'model_max_length', None)
    if model_max_length and model_max_length > 0:
        # Reserve some tokens for the prompt, allow generation of the rest
        # Use a large fraction of available context for generation
        available_for_generation = max(MAX_TOKEN, model_max_length - prompt_length - 100)
    else:
        # If no model_max_length, use MAX_TOKEN
        available_for_generation = MAX_TOKEN
    
    # Implementation for multi-agent: differentiate solver/checker behavior
    if mode in {"standard", "thinking", "solver"}:
        # Stop once a \\boxed{} answer appears to avoid wasting tokens
        stopping_criteria.append(StopOnBoxedAnswer(tokenizer, prompt_length))
        gen_kwargs["max_new_tokens"] = available_for_generation
        gen_kwargs["stopping_criteria"] = stopping_criteria
    elif mode == "checker":
        # Checker: stop immediately after VERDICT is found or after 200 chars to prevent rambling
        stopping_criteria.append(StopAfterCheckerConclusion(tokenizer, prompt_length))
        # Checker needs less tokens - keep it short and focused
        gen_kwargs["max_new_tokens"] = min(100, available_for_generation)
        gen_kwargs["stopping_criteria"] = stopping_criteria
        # Make checker thoughtful but not too random
        gen_kwargs["temperature"] = 0.3  # Low but not too low - allows some variance
        gen_kwargs["do_sample"] = True  # Enable sampling to allow variability
        gen_kwargs["top_p"] = 0.9  # Nucleus sampling
        gen_kwargs["repetition_penalty"] = 1.5  # Moderate penalty to discourage rambling
    else:
        # Fallback: generic generation
        gen_kwargs["max_new_tokens"] = available_for_generation

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response
