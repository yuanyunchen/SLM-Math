# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import argparse
import json
import numpy as np
import os
from pebble import ProcessPool
from functools import partial
from typing import Any, Dict, Type, Optional, List, Tuple
from pydantic import BaseModel
from omegaconf import OmegaConf
from tqdm import tqdm
from rstar_deepthink.constants import (
    NO_VALID_CHILD, 
    TOO_MANY_STEPS, 
    TOO_MANY_CODE_ERRORS, 
)
from rstar_deepthink.config import BaseConfig
from rstar_deepthink.agents.utils import math_equiv


class InferNode(BaseModel):

    tag: str = "0"

    text: str = ""
    extra_info: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str = ""

    c_puct: float = 2
    depth: int = 0

    value: float = 0
    q_value: float = 0
    visit_count: int = 0

    parent: Optional[Any] = None
    children: List[Any] = []

    prune: bool = False
    mark: str = "no_final"

    final_correct: int = 0
    final_wrong: int= 0
    
    def puct(self) -> float:
        q_value = self.q_value if self.visit_count > 0 else 0
        if self.parent.visit_count == 0 or self.visit_count == 0:
            u_value = 0
        else:
            u_value = self.c_puct * np.sqrt(np.log(self.parent.visit_count) / (self.visit_count))
        return q_value + u_value


def rebuild_tree(
    tree_dict: Dict[str, Any], 
    max_num_children: int,
    c_puct: float,
    root_tag: str = "0",
) -> Tuple[Type[InferNode], int]:
    root = InferNode(
        parent=None,
        tag=root_tag,
        c_puct=c_puct,
        **tree_dict[root_tag],
    )
    candidates = [root]
    max_depth = 0
    while candidates:
        node = candidates.pop(0)
        for idx in range(max_num_children):
            tag = f"{node.tag}.{idx}"
            depth = node.depth + 1
            if tag in tree_dict:
                child = InferNode(
                    parent=node,
                    tag=tag,
                    depth=depth,
                    c_puct=c_puct,
                    **tree_dict[tag],
                )
                max_depth = max(max_depth, depth)
                node.children.append(child)
                candidates.append(child)
    return root, max_depth


def is_valid_final_answer_node(node: Type[InferNode]) -> bool:
    if not node.children and node.final_answer and \
        node.final_answer not in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
        return True
    return False


def prune_node(node: Type[InferNode]) -> bool:
    if node.children:
        children_prune = []
        for child in node.children:
            children_prune.append(prune_node(child))
        if all(children_prune):
            node.prune = True
    else:
        # for leaf node
        if not is_valid_final_answer_node(node): 
            node.prune = True
    return node.prune


def select_non_prune(current_nodes: List[Type[InferNode]]) -> List[Type[InferNode]]:
        candidate_nodes = []
        for node in current_nodes:
            candidate_nodes.extend([child for child in node.children if not child.prune])
        return candidate_nodes


def sort_by_strategy(
    candidate_nodes: List[Type[InferNode]],
    strategy: str = "q_value",
) -> List[Type[InferNode]]:
    if strategy == "value":
        return sorted(candidate_nodes, key=lambda x: x.value, reverse=True)
    elif strategy == "q_value":
        return sorted(candidate_nodes, key=lambda x: x.q_value, reverse=True)
    elif strategy == "visit_count":
        return sorted(candidate_nodes, key=lambda x: x.visit_count, reverse=True)
    elif strategy == "puct":
        return sorted(candidate_nodes, key=lambda x: x.puct(), reverse=True)
    else:
        raise NotImplementedError(f"strategy {strategy} not implemented")


def extra_solution_dict( 
    full_tree_dict: Dict[str, Any], 
    prune: bool = False,
    max_num_children: int = 1000,
    strategy: str = "q_value",
    c_puct: float = 2,
) -> Optional[Dict[str, Any]]:
    """
    This function is used to extract solution from a built tree.
    It is mainly used for MCTS, but also works for saved tree from step_beam.
    """
    question = full_tree_dict["question"]
    ground_truth = full_tree_dict.get("answer", None)
    tree_dict = full_tree_dict["rstar"]

    # rebuild tree
    root, tree_depth = rebuild_tree(tree_dict, max_num_children=max_num_children, c_puct=c_puct)

    #pruning tree
    if prune:
        prune_node(root)
        if root.prune:
            # no valid leaf node for the entire tree
            return {}
    
    # search in tree
    final_answer_nodes = []
    current_nodes = [root] 
    max_dep = 0
    for _ in range(tree_depth):
        candidate_nodes = select_non_prune(current_nodes)
        candidate_nodes = sort_by_strategy(candidate_nodes, strategy)
        current_nodes = candidate_nodes[:]

        for current_node in current_nodes[:]:
            max_dep = max(max_dep, current_node.depth)
            if is_valid_final_answer_node(current_node):
                final_answer_nodes.append(current_node)
                current_nodes.remove(current_node)
            elif not current_node.children:
                current_nodes.remove(current_node)
    if not final_answer_nodes:
        return {"max_dep": max_dep}

    final_answer_nodes = sort_by_strategy(final_answer_nodes, strategy)
    
    top_final_answer_node = final_answer_nodes[0]
    final_answer_trace = ""
    tmp_node = top_final_answer_node
    while tmp_node:
        final_answer_trace = f"{tmp_node.text}" + final_answer_trace
        tmp_node = tmp_node.parent
    judge = False
    for node in final_answer_nodes:
        if math_equiv(ground_truth, node.final_answer):
            judge = True
            break
    return {
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": top_final_answer_node.final_answer,
        "tag": top_final_answer_node.tag,
        "judge": judge,
        "max_dep": max_dep,
        "final_answer_trace": final_answer_trace,
    }


def get_solution_dict(full_tree_dict):
    solution = extra_solution_dict(
                full_tree_dict,
                prune=False,
                max_num_children=3000,
                strategy="q_value", # value  q_value         
            )
    final_answer_trace = solution.get("final_answer_trace", "")
    if not final_answer_trace:
        print("no answer", full_tree_dict['index'])
    judge = solution.get("judge", False)
    # if solution.get("question", False):
    if math_equiv(solution["ground_truth"], solution["final_answer"]):
        return 1, final_answer_trace, judge
    return 0, final_answer_trace, judge


def eval_output_file(dir):
    full_tree_dicts = []
    with open(dir, "r") as f:
        for line in tqdm(f):
            full_tree_dict = json.loads(line)
            full_tree_dicts.append(full_tree_dict)
            
    solutions = []
    final_answer_traces = []
    judges = []
    with ProcessPool(max_workers=os.cpu_count() - 2) as pool:
    #with ProcessPool(max_workers=1) as pool:
        executor = partial(
            get_solution_dict
        )
        future = pool.map(executor, full_tree_dicts, timeout=120)
        iterator = future.result()

        progress_bar = tqdm(total=len(full_tree_dicts), desc="Execute")  

        while True:
            try:
                result, final_answer_trace, judge = next(iterator)
                solutions.append(result)
                final_answer_traces.append(final_answer_trace)
                judges.append(judge)
            except StopIteration:
                break
            except Exception as error:
                solutions.append(0) 
                final_answer_traces.append("")
                judges.append(False)
                print(error)
            if progress_bar is not None:
                progress_bar.update(1) 
        
        if progress_bar is not None:
            progress_bar.close() 

    total = len(solutions)
    cnt = sum(solutions)
    print(cnt, total, f"Accuracy: {cnt / total}")
    output_ct, output_ct_gt, error_ct = 0, 0, 0
    assert len(final_answer_traces) == len(solutions), "tags should be the same length as solutions"
    return cnt / total, output_ct / total, output_ct_gt / total


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, default="")
    args = args.parse_args()
    
    dirs = [args.file_path]
    cnt, total, total_cnt = 0, 0, 0
    
    inputs = []
    for dir in dirs:
        with open(dir, "r") as f:
            for line in tqdm(f):
                full_tree_dict = json.loads(line)
                # if "[asy]" not in full_tree_dict["question"]:
                #     continue
                inputs.append(full_tree_dict)
    
    
    test_math500 = False
    if test_math500:
        with open("./eval_data/math500_test.json", "r") as f:
            math500 = json.load(f)
            math500_ques = set()
            for item in math500:
                math500_ques.add(item['question'])
        inputs = inputs[:5000]
        inputs_new = []
        for item in inputs:
            if item['question'] in math500_ques:
                inputs_new.append(item)
        inputs = inputs_new
        print(len(inputs), "only math500")
    
    solutions = []
    final_answer_traces = []
    judges = []
    with ProcessPool(max_workers=os.cpu_count() - 2) as pool:
    #with ProcessPool(max_workers=1) as pool:
        executor = partial(
            get_solution_dict
        )
        future = pool.map(executor, inputs, timeout=120)
        iterator = future.result()

        progress_bar = tqdm(total=len(inputs), desc="Execute")  

        while True:
            try:
                result, final_answer_trace, judge = next(iterator)
                solutions.append(result)
                final_answer_traces.append(final_answer_trace)
                judges.append(judge)
            except StopIteration:
                break
            except Exception as error:
                solutions.append(0) 
                final_answer_traces.append("")
                judges.append(False)
                print("process error",error)
            if progress_bar is not None:
                progress_bar.update(1) 
        if progress_bar is not None:
            progress_bar.close() 

    total = len(solutions)
    cnt = sum(solutions)
    print(cnt, total, f"Pass 1 : {cnt / total}")

    jud = 0
    for item in judges:
        if item: jud += 1 
    print(f"Pass n : {jud / total}")