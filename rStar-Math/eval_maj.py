# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import sys, os
import argparse
from pebble import ProcessPool
from tqdm import tqdm
from functools import partial
import random
from rstar_deepthink.agents.utils import math_equiv


def eval(full_tree_dict):
    if math_equiv(full_tree_dict["gt"], full_tree_dict["pred"]):
        return 1
    return 0


def cal_final_results(inputs, task_size=1):
    solutions = []
    with ProcessPool(max_workers=os.cpu_count() - 8) as pool:
        executor = partial(eval)
        future = pool.map(executor, inputs, timeout=240)
        iterator = future.result()

        progress_bar = tqdm(total=len(inputs), desc="Execute")  

        while True:
            try:
                result = next(iterator)
                solutions.append(result)
            except StopIteration:
                break
            except Exception as error:
                solutions.append(0) 
                print(error)
            if progress_bar is not None:
                progress_bar.update(1) 
        
        if progress_bar is not None:
            progress_bar.close() 
    # print("All batches processed.",sum(solutions), len(solutions))
    # print(solutions)
    # print(inputs)
    return sum(solutions) / task_size


def major_vote_eval(input):
    pred = input['pred']
    for item in pred:
        item['count'] = 0
        item['sum_score'] = 0
    for item in pred:
        for item2 in pred:
            # TODO: For benchmarks with longer evaluation times, such as college-math and omni-math, it is advisable to use the `==` operator for equality checks.
            # if item['ans'] == item2['ans']:
            if math_equiv(item['ans'], item2['ans']):
                item['count'] += 1
                item['sum_score'] += item2['score']
    input['pred'] = pred
    return input


def cal_major_vote(inputs):
    ret_inputs = []
    with ProcessPool(max_workers=os.cpu_count() - 8) as pool:
        executor = partial(
            major_vote_eval
        )
        future = pool.map(executor, inputs, timeout=240)
        iterator = future.result()

        progress_bar = tqdm(total=len(inputs), desc="Execute")  

        while True:
            try:
                result = next(iterator)
                ret_inputs.append(result)
            except StopIteration:
                break
            except Exception as error:
                print("Failed to retrieve the major_vote answer: ", error)

            if progress_bar is not None:
                progress_bar.update(1) 
        
        if progress_bar is not None:
            progress_bar.close()
    return ret_inputs


def load_data(dirs, max_tree, max_node_per_tree):
    tree_count = 0
    question_results = {}

    for dir_path in dirs:
        try:
            with open(dir_path, 'r') as file:
                data = json.load(file)
                qr_data = data[1]
            if qr_data:
                tree_count += 1
        except:
            continue

        for item in qr_data:
            question = item['question'].strip()
            answers = item['answers']
            judgements = item['judgements']
            value_estimates = item['value_estimate']

            # Sort by value_estimate and select the top max_node_per_tree entries.
            sorted_items = sorted(
                zip(answers, judgements, value_estimates),
                key=lambda x: x[2], 
                reverse=True
            )[:max_node_per_tree]

            # Extract the sorted results
            top_items = {
                "gt": item['gt'],
                "answers": [item[0] for item in sorted_items],
                "judgements": [item[1] for item in sorted_items],
                "value_estimate": [item[2] for item in sorted_items],
            }

            # update question_results
            if question not in question_results:
                question_results[question] = top_items
            else:
                for key in ['answers', 'judgements', 'value_estimate']:
                    question_results[question][key].extend(top_items[key])

        if tree_count >= max_tree:
            break

    return question_results


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, default="", help="The directory for storing intermediate_metric")
    args.add_argument("--max_tree", type=int, default=256, help="Maximum number of MCTS files to select")
    args.add_argument("--max_node_per_tree", type=int, default=1, help="Number of high-score results selected per MCTS file")
    args.add_argument("--task_size", type=int, default=1, help="Total number of benchmark questions, for instance, math500 consists of 500 questions.")
    args.add_argument("--save_path", type=str, default="", help="The location where the result files are stored.")
    args.add_argument("--top_n", type=int, default=4, help="Select the top n highest-scoring options for a majority vote.")
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dir = args.path
    dirs = [os.path.join(dir, item) for item in os.listdir(dir) if item.startswith("intermediate_metric")]

    question_results = load_data(dirs, args.max_tree, args.max_node_per_tree)

    pass_count = 0

    inputs = []
    sorted_items = []
    for question, item in question_results.items():
        gt = item['gt']
        has_pass = any(item['judgements'])
        pass_count += 1 if has_pass else 0

        sorted_item = sorted(
            [
                {"ans": ans, "judge": judge, "value_estimate": value}
                for ans, judge, value in zip(item['answers'], item['judgements'], item['value_estimate'])
            ],
            key=lambda x: x['value_estimate'],
            reverse=True
        )
        sorted_items.append(sorted_item[:])
        if not sorted_item:
            continue

        # Retain the top n with the highest scores.
        input_data = {
            "gt": gt,
            "pred": [{"score": it['value_estimate'], "ans": it['ans']} for it in sorted_item[:args.top_n]]
        }
        inputs.append(input_data)

    pass1 = 0
    for item in sorted_items:
        if item and item[0]["judge"]:
            pass1 += 1

    maj_inputs = cal_major_vote(inputs)
    maj = []
    for item in maj_inputs:
        max_count = max([pred['count'] for pred in item['pred']])
        maj_ans = random.sample([pred['ans'] for pred in item['pred'] if pred['count'] == max_count], min(1, len(item['pred'])))
        maj.append({
            "gt": item['gt'],
            'pred': maj_ans[0] if maj_ans else ""
        })
    major_vote = cal_final_results(maj)

    weighted_maj = []
    for inp in maj_inputs:
        anss = sorted(inp['pred'], key=lambda x: x['sum_score'], reverse=True)
        weighted_maj.append({
            "gt": inp['gt'],
            "pred": anss[0]['ans']
        })
    weighted_major_vote = cal_final_results(weighted_maj)

    print("pass 1: ", pass1 / args.task_size, "pass n: ", pass_count / args.task_size)
    print("task_size: ", args.task_size, "top_n: ", args.top_n)
    print("top_n_major_vote: ", major_vote / args.task_size, "top_n_weighted_major_vote: ", weighted_major_vote / args.task_size)
    if args.save_path:
        with open(args.save_path, "a+") as f:
            f.write(f"{args.path}  ")
            f.write(f"pass 1: {pass1 / args.task_size}  ")
            f.write(f"pass n: {pass_count / args.task_size}  ")
            f.write(f"task_size: {args.task_size}  ")
            f.write(f"top_n: {args.top_n}  ")
            f.write(f"top_n_major_vote: {major_vote / args.task_size}  ")
            f.write(f"top_n_weighted_major_vote: {weighted_major_vote / args.task_size}  ")
            f.write("\n")