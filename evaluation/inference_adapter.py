"""
Inference Adapter for Agent Workflows
Provides a unified interface for agent workflows to use inference engines
"""

import torch
from typing import Optional, List


class InferenceAdapter:
    """
    Adapter to make inference engines compatible with existing agent workflows
    Provides both single and batch inference
    """
    
    def __init__(self, model, tokenizer, inference_engine=None):
        """
        Args:
            model: The loaded model (can be None for vLLM)
            tokenizer: The loaded tokenizer
            inference_engine: Optional inference engine for batch processing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.inference_engine = inference_engine
        self.device = model.device if model is not None else "cuda"
    
    def generate_single(self, prompt: str, **kwargs) -> str:
        """
        Generate response for a single prompt
        Uses inference engine if available, otherwise falls back to model.generate
        """
        if self.inference_engine:
            # Use inference engine
            return self.inference_engine.generate_single(prompt, **kwargs)
        else:
            # Fallback to direct model generation
            return self._direct_generate(prompt, **kwargs)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts
        """
        if self.inference_engine:
            # Use inference engine
            return self.inference_engine.generate_batch(prompts, **kwargs)
        else:
            # Fallback: generate one by one
            return [self._direct_generate(p, **kwargs) for p in prompts]
    
    def _direct_generate(self, prompt: str, **kwargs) -> str:
        """Direct generation using model (fallback)"""
        from models.inference import generate_response
        return generate_response(
            self.model, 
            self.tokenizer, 
            prompt, 
            mode=kwargs.get('mode', 'standard'),
            detailed=kwargs.get('detailed', False)
        )
    
    def get_model_and_tokenizer(self):
        """Get underlying model and tokenizer"""
        return self.model, self.tokenizer


def create_inference_adapters(
    model, 
    tokenizer, 
    solver_engine, 
    checker_model=None, 
    checker_tokenizer=None, 
    checker_engine=None
):
    """
    Create inference adapters for solver and checker
    
    Returns:
        solver_adapter, checker_adapter
    """
    solver_adapter = InferenceAdapter(model, tokenizer, solver_engine)
    
    if checker_model is not None or checker_tokenizer is not None:
        checker_adapter = InferenceAdapter(
            checker_model if checker_model is not None else model,
            checker_tokenizer if checker_tokenizer is not None else tokenizer,
            checker_engine
        )
    else:
        # Same as solver
        checker_adapter = solver_adapter
    
    return solver_adapter, checker_adapter


