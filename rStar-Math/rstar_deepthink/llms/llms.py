# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from typing import List
from vllm import LLM, SamplingParams


class Reward():
    value_estimate: float = 0
    def __init__(self, value_estimate: float):
        self.value_estimate = value_estimate

def llm_generate(
    prompts: List[str],
    sampling_params: SamplingParams,
    engine: LLM,
):
    if not prompts: return []
    outputs = engine.generate(prompts, sampling_params=sampling_params)   
     
    # remove duplicate outputs
    for output in outputs:
        outs = output.outputs
        text_set = set()
        filterd_outs = []
        for out in outs:
            if out.text not in text_set:
                filterd_outs.append(out)
                text_set.add(out.text)
        output.outputs = filterd_outs
    
    # if touch the <end_of_answer>, only keep the first 
    for output in outputs:
        outs = output.outputs
        end_ans, other_ans = [], []
        for out in outs:
            if out.stop_reason != "<end_of_answer>":
                other_ans.append(out)
            else:
                end_ans.append(out)
        if len(end_ans) != 0:
            output.outputs = end_ans[:1]
        else:
            output.outputs = other_ans
    return outputs


def prevent_overlength(texts, tokenizer, max_model_len):
    """
    https://github.com/vllm-project/vllm/issues/10794 
    Tokenize the input string and crop it to ensure that the length of each string is strictly less than x_seq.
    """
    # tokenize
    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
    
    # truncate
    truncated_texts = [
        tokenizer.convert_tokens_to_string(tokens[:max_model_len - 1]) if len(tokens) >= max_model_len 
        else tokenizer.convert_tokens_to_string(tokens)
        for tokens in tokenized_texts
    ]
    
    return truncated_texts


def rm_generate(model: LLM, v_head, prompts, tokenizer, max_model_len):
    if not prompts:
        return []
    rewards = []
    batch_size = 2000
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            inputs = [prompt['prefix'] + prompt['text'] for prompt in prompts[i:i+batch_size]]
            batch_outputs = model.encode(prevent_overlength(inputs, tokenizer, max_model_len))
            for output in batch_outputs:
                last_hidden_states = output.outputs.data[-1]
                reward = v_head(last_hidden_states)
                rewards.append(reward.cpu().item())
    rewards = [reward / 5 for reward in rewards] # PPM use 5 as the scale
    rewards = torch.tanh(torch.tensor(rewards)).tolist()
    return [Reward(value_estimate=reward) for reward in rewards]
