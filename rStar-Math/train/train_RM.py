# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import warnings
import torch
import os
from rm import *
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from typing import Dict
from trl import ModelConfig, RewardConfig
import random
import numpy as np
IGNORE_INDEX = 0

tqdm.pandas()
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
seed_torch(0)

def rank0_print(rank, *args, **kwargs):
    """Print, but only on rank 0."""
    if rank == 0:
        print(*args, **kwargs)
        
if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    parser.add_argument('--pair_json_path', type=str, default="/home/v-xinyuguan/teamdrive/teamdrive/xy/xy/mcts/0924/qwen2SFT__ORM_filter.json")
    parser.add_argument('--test_pair_json_path', type=str, default=None)
    parser.add_argument('--metrics_path', type=str, default=None)
    parser.add_argument('--linear_tpye', type=str, default="single")
    parser.add_argument('--attn_impl', type=str, default="eager")
    config, model_config, remain_args = parser.parse_args_into_dataclasses()
    
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    config.pair_json_path = remain_args.pair_json_path

    from accelerate import Accelerator
    accelerator = Accelerator()
    rank = accelerator.process_index
    print(rank)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=True, 
        use_fast=True,
        padding_side="right",
        split_special_tokens=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if remain_args.attn_impl == "flash_attention_2" else torch.float32,
        attn_implementation=remain_args.attn_impl,
        use_cache=False,
    )
    model = RewardModelWithValueHead(pretrained_model=model, linear_tpye=remain_args.linear_tpye)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ['<code>', '<end_of_step>', '<end_of_code>', '<output>', '<end_of_output>', '<answer>', '<end_of_answer>', '<|user|>', '<|assistant|>', '<refine>', '<end_of_refine>', '\n<|assistant|>']
        },
        replace_additional_special_tokens=False,
    )
    model.pretrained_model.resize_token_embeddings(len(tokenizer))


    ################
    # Dataset
    ################
    if remain_args.pair_json_path is not None:
        raw_datasets = load_dataset('json', data_files=remain_args.pair_json_path, writer_batch_size=100)
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42)
        if remain_args.test_pair_json_path is not None:
            raw_datasets['test'] = load_dataset('json', data_files=remain_args.test_pair_json_path)['train']
        else:
            raw_datasets['train'], raw_datasets['test'] = raw_datasets['train'].train_test_split(test_size=0.05, seed=42).values()
            rank0_print(rank, 'trainset size:', len(raw_datasets['train']), 'testset size:', len(raw_datasets['test']))

    remove_columns = ['prompt', 'neg', 'pos', 'neg_count', 'pos_count']
    remove_columns = [col for col in remove_columns if col in raw_datasets['train'].column_names]
    partial_func = partial(preprocess_value_dataset, tokenizer=tokenizer, max_length=config.max_length)
    raw_datasets = raw_datasets.map(
        partial_func,
        batched=True,
        num_proc=16,
        remove_columns=remove_columns
    ) 
    rank0_print(rank, 'after preprocess', raw_datasets['train'].column_names)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    rank0_print(rank, 'After filtering, trainset size:', len(train_dataset), 'testset size:', len(eval_dataset))

    ################
    # Training
    ################
    rank0_print(rank, config)
    rank0_print(rank, train_dataset.column_names)
    
    trainer = RMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=config.max_length,
            padding='max_length'
            ),
        compute_metrics=ComputeAccuracy()
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    trainer.save_state()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    rank0_print(rank, metrics)
    import json
    import os
    if remain_args.metrics_path:
        os.makedirs(os.path.dirname(remain_args.metrics_path), exist_ok=True)
        with open(remain_args.metrics_path, 'w') as f:
            metrics['test_pair_json_path'] = remain_args.test_pair_json_path
            metrics['model_name_or_path'] = model_config.model_name_or_path
            json.dump(metrics, f)
