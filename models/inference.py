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

MAX_TOKEN = 2048


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
    Halts checker generation once a verdict and a short explanation have been produced.

    Heuristic: once we see a line starting with 'VERDICT:' and at least two subsequent
    non-empty lines, we treat the checker response as complete.
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

        # Once the checker has written a VERDICT line, we consider the response complete.
        # The prompt instructs the checker to end with a line like:
        #   VERDICT: CORRECT / INCORRECT / UNCLEAR
        return "VERDICT:" in upper


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
    """
    Generate response from model given a prompt.

    The `mode` argument is used to select decoding behavior and stopping criteria:
      - "standard", "thinking", "solver", "reflector": stop when a \\boxed{} answer appears,
        with a larger max_new_tokens budget.
      - "checker": use a custom stopping rule that ends after a VERDICT line plus a short explanation.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    prompt_length = inputs['input_ids'].shape[1]
        
    # Create streamer for real-time token-by-token output (only if detailed)
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None

    # Configure stopping criteria and decoding parameters based on mode
    stopping_criteria = StoppingCriteriaList()
    gen_kwargs = {
        "temperature": 0.1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,
        "streamer": streamer,
    }

    # Implementation for multi-agent: differentiate solver/checker/reflector behavior
    if mode in {"standard", "thinking", "solver", "reflector"}:
        # Stop once a \\boxed{} answer appears to avoid wasting tokens
        stopping_criteria.append(StopOnBoxedAnswer(tokenizer, prompt_length))
        gen_kwargs["max_new_tokens"] = MAX_TOKEN
        gen_kwargs["stopping_criteria"] = stopping_criteria
    elif mode == "checker":
        # Checker: stop after VERDICT and a couple of explanation lines
        stopping_criteria.append(StopAfterCheckerConclusion(tokenizer, prompt_length))
        gen_kwargs["max_new_tokens"] = min(512, MAX_TOKEN)
        gen_kwargs["stopping_criteria"] = stopping_criteria
    else:
        # Fallback: generic short generation
        gen_kwargs["max_new_tokens"] = min(256, MAX_TOKEN)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response
