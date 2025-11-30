"""
Unified Inference Engine for Batch and Accelerated Inference
Supports both transformers and vLLM backends
"""

import re
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY
)


class BaseInferenceEngine(ABC):
    """Abstract base class for inference engines"""
    
    def __init__(self, model_name: str, base_path: Path, **kwargs):
        self.model_name = model_name
        self.base_path = base_path
        self.model_dir = base_path / 'pretrained_models' / model_name
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory {self.model_dir} not found!")
        
        model_files = list(self.model_dir.glob("*.safetensors")) + list(self.model_dir.glob("*.bin"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.model_dir}!")
    
    @abstractmethod
    def load(self):
        """Load model and tokenizer"""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a batch of prompts"""
        pass
    
    def generate_single(self, prompt: str, **kwargs) -> str:
        """Generate response for a single prompt (convenience wrapper)"""
        return self.generate_batch([prompt], **kwargs)[0]


class TransformersEngine(BaseInferenceEngine):
    """Transformers-based inference with batch processing"""
    
    def __init__(self, model_name: str, base_path: Path, **kwargs):
        super().__init__(model_name, base_path, **kwargs)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load()
    
    def load(self):
        """Load model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"[TransformersEngine] Loading model from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        print(f"[TransformersEngine] Model loaded successfully on {self.device.upper()}\n")
    
    def _has_boxed_answer(self, text: str) -> bool:
        """Check if text contains complete \\boxed{} answer"""
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return False
        
        after_boxed = text[idx + 7:]
        brace_count = 1
        
        for char in after_boxed:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return True
        
        return False
    
    def _check_stopping_batch(self, generated_texts: List[str]) -> List[bool]:
        """Check stopping criteria for batch"""
        return [self._has_boxed_answer(text) for text in generated_texts]
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for batch of prompts with dynamic stopping"""
        if not prompts:
            return []
        
        # Extract parameters with standardized defaults
        max_new_tokens = kwargs.get('max_new_tokens', MAX_NEW_TOKENS)
        temperature = kwargs.get('temperature', TEMPERATURE)
        do_sample = kwargs.get('do_sample', DO_SAMPLE)
        top_p = kwargs.get('top_p', TOP_P)
        repetition_penalty = kwargs.get('repetition_penalty', REPETITION_PENALTY)
        detailed = kwargs.get('detailed', False)
        
        # Tokenize all prompts with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        prompt_lengths = inputs['attention_mask'].sum(dim=1).tolist()
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                # Note: Batch stopping criteria is complex, handled post-generation
            )
        
        # Decode responses
        responses = []
        for i, output_ids in enumerate(outputs):
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            # Remove prompt from response
            prompt = prompts[i]
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                # Fallback: extract by token length
                response_ids = output_ids[prompt_lengths[i]:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            responses.append(response)
        
        return responses
    
    def get_model_and_tokenizer(self):
        """Get underlying model and tokenizer (for compatibility)"""
        return self.model, self.tokenizer


class VLLMEngine(BaseInferenceEngine):
    """vLLM-based inference for high-throughput generation"""
    
    def __init__(self, model_name: str, base_path: Path, **kwargs):
        super().__init__(model_name, base_path, **kwargs)
        self.llm = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        self.load()
    
    def load(self):
        """Load vLLM model"""
        try:
            from vllm import LLM, SamplingParams
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm\n"
                "Note: vLLM requires CUDA and may have specific GPU requirements."
            )
        
        print(f"[vLLM] Loading model from {self.model_dir}...")
        self.llm = LLM(
            model=str(self.model_dir),
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        # Load tokenizer for compatibility
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[vLLM] Model loaded successfully\n")
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses using vLLM's efficient batching"""
        if not prompts:
            return []
        
        # Extract parameters with standardized defaults
        max_tokens = kwargs.get('max_new_tokens', MAX_NEW_TOKENS)
        temperature = kwargs.get('temperature', TEMPERATURE)
        do_sample = kwargs.get('do_sample', DO_SAMPLE)
        top_p = kwargs.get('top_p', TOP_P)
        repetition_penalty = kwargs.get('repetition_penalty', REPETITION_PENALTY)
        
        # For greedy decoding (do_sample=False), use temperature=0
        # This matches transformers' behavior
        if not do_sample and temperature > 0:
            temperature = 0.0
        
        # vLLM sampling parameters
        # Note: Removed stop=["\\boxed{"] because it prematurely truncates answers
        # The model should naturally complete the answer and reach EOS
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=True,
            # stop tokens removed to prevent premature truncation
        )
        
        # Generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract responses
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            responses.append(generated_text.strip())
        
        return responses
    
    def get_model_and_tokenizer(self):
        """Get underlying tokenizer (model is internal to vLLM)"""
        return None, self.tokenizer


def load_inference_engine(
    model_name: str,
    base_path: Path,
    backend: str = "transformers",
    **kwargs
) -> BaseInferenceEngine:
    """
    Factory function to load inference engine
    
    Args:
        model_name: Name of the model
        base_path: Base path to project
        backend: 'transformers' or 'vllm'
        **kwargs: Additional engine-specific parameters
    
    Returns:
        Inference engine instance
    """
    backend = backend.lower()
    
    if backend == "transformers":
        return TransformersEngine(model_name, base_path, **kwargs)
    elif backend == "vllm":
        return VLLMEngine(model_name, base_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'transformers' or 'vllm'")

