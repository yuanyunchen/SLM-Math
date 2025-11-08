# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import random
import torch
import transformers
import sys
import json
from shutil import copy
from eval_output import eval_output_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--base_mode", type=str, default="")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--save_res", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.save_res:
        files_to_evaluate = {
            "gsm8k.jsonl": "gsm8k_res",
            "math.jsonl": "math_res",
            "aime2024.jsonl": "aime2024_res",
            "amc23.jsonl": "amc23_res",
            "collegemath.jsonl": "collegemath_res",
            "gaokao2023en.jsonl": "gaokao2023en_res",
            "olympiadbench.jsonl": "olympiadbench_res",
            "math500.jsonl": "math500_res",
            "omni-math.jsonl": "omni_res"
        }

        for dir in os.listdir(args.save_res):
            results = {"model_dir": dir}
            
            for file, res_key in files_to_evaluate.items():
                try:
                    res, _, _ = eval_output_file(os.path.join(args.save_res, dir, file))
                    results[res_key] = res
                except:
                    results[res_key] = 0
            
            if any(results[res_key] != 0 for res_key in files_to_evaluate.values()):
                with open(os.path.join(args.save_res, "result.json"), "a") as f:
                    f.write(json.dumps(results) + "\n")
        
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        model = args.model
        print("path: ", model)
        if args.task:
            task_to_file = {
                "gsm8k": "GSM8K_test.json",
                "math": "MATH_test.json",
                "aime2024": "aime2024_test.json",
                "amc23": "amc23_test.json",
                "collegemath": "collegemath_test.json",
                "gaokao2023en": "gaokao2023en_test.json",
                "olympiadbench": "olympiadbench_test.json",
                "math500": "math500_test.json",
                "omni-math": "omni_test.json"
            }

            if args.task in task_to_file:
                save_dir = os.path.join(args.model, args.task)
                command = (
                    f'python main.py --custom_cfg config/sft_eval_greedy.yaml '
                    f'--qaf ./eval_data/{task_to_file[args.task]} '
                    f'--save_in_model {save_dir} --model_dir {model}'
                )
                os.system(command)
            else:
                print(f"Error: Unknown task '{args.task}'")
            
