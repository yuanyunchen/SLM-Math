# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import os
import json
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from rstar_deepthink.agents import BS, MCTS
from rstar_deepthink.solver import Solver
from rstar_deepthink.config import BaseConfig

torch.set_num_threads(12)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_qaf(filename: str):
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
        if "example" in data:
            data = data["example"]
    elif filename.endswith(".jsonl"):
        data = []
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    else:
        raise ValueError(f"Unrecognized file format: {filename}")
    return data

def batch(iterable, n=-1):
    l = len(iterable)
    if n <= 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="config/sft_eval_mcts.yaml")
    args.add_argument("--qaf", type=str, default="", help="quesuion and answer file")
    args.add_argument('--model_dir', type=str, default="") 
    args.add_argument('--reward_model_dir', type=str, default="") 
    args.add_argument('--save_in_model', type=str, default="")
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.reward_model_dir:
        config.reward_model_dir = args.reward_model_dir
    print(config)

    llm_version = os.path.basename(config.model_dir.rstrip("/"))

    data = load_qaf(args.qaf)
    solver = Solver(config=config)

    # init agent
    if config.mode == "mcts":
        agent = MCTS
    elif config.mode == "bs":
        agent = BS
    else:
        raise NotImplementedError
    if args.reward_model_dir:
        llm_version += "." + args.reward_model_dir.split("/")[-1]
        
    saved_jsonl_file = f"{args.qaf}.{config.mode}.{llm_version}.{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl" 
    
    if args.save_in_model:
        saved_jsonl_file = args.save_in_model + '.jsonl'
        saved_jsonl_file_dir = os.path.dirname(saved_jsonl_file)
        os.makedirs(saved_jsonl_file_dir, exist_ok=True)
        
    with open(saved_jsonl_file, "a+", encoding='utf-8') as writer:
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            agents = [agent(config=config, question=d["question"], ground_truth=str(d["answer"])) 
                      for d in cur_data]
            jsonlines = solver.solve(agents, saved_jsonl_file, cur_data)
            for d in cur_data:
                question = d["question"]
                d["rstar"] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()
