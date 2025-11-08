# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, sys
import torch
from rstar_deepthink.llms.rm import *
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

def llm_init(config):
    llm = LLM(
        model=config.model_dir, 
        tensor_parallel_size=config.tp, 
        trust_remote_code=True,
        seed=config.seed if config.seed else 0,
        swap_space=config.swap_space,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.llm_gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend='ray' if config.tp > 1 else None,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        best_of=config.best_of,
        max_tokens=config.max_tokens, 
        n=config.n_generate_sample,
        stop=config.stop,
        skip_special_tokens=False,
        seed=config.seed if config.temperature == 0 else None, # vllm0.6.6.post1 
    )
    return llm, sampling_params

def llm_engine(config):
    llm, sampling_params = llm_init(config)
    return llm, sampling_params

def rm_engine(config):
    if config.need_value_func:
        prm_model = LLM(
            model=config.reward_model_dir, 
            task="reward",
            tensor_parallel_size=1, 
            trust_remote_code=True,
            max_model_len=config.max_model_len,
            enforce_eager=True,
            swap_space=0,
            gpu_memory_utilization=0.98 - config.llm_gpu_memory_utilization, # for qwen 7b, rm need 15G memory
        )
        
        v_head_state = torch.load(os.path.join(config.reward_model_dir, "value_head.bin"), weights_only=True)
        v_state = {}
        for name, param in v_head_state.items():
            v_state[name.replace("v_head.", "")] = param
        model_config = AutoConfig.from_pretrained(config.reward_model_dir, trust_remote_code=True, use_cache = False)
        v_head = ValueHead(model_config)
        v_head.load_state_dict(v_state)
        v_head.eval()
        tokenizer = AutoTokenizer.from_pretrained(config.reward_model_dir, trust_remote_code=True, use_cache = False, split_special_tokens=False,)
        return prm_model, v_head, tokenizer
    else:
        return None, None, None
