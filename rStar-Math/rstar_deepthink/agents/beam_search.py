# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from functools import partial
from vllm.outputs import RequestOutput
from rstar_deepthink.nodes.base_node import BaseNode
from rstar_deepthink.constants import (
    TOO_MANY_CODE_ERRORS, 
    TOO_MANY_STEPS, 
    NO_VALID_CHILD, 
    CODE_END,
    ANSWER,
    CODE_END,
    OUTPUT,
    OUTPUT_END,
)
from .tree import BaseTree, code_execution


class BS(BaseTree):
    """
    Step-level Beam Search
    """
    NODE_KEYS: List[str] = ["action", "action_input", "final_answer"]
    prompt_wrap: Optional[Callable[[...], str]] = None
    obs_wrap: Optional[Callable[str, str]] = None
    step_unwrap: Optional[Callable[[...], Dict[str, str]]] = None
    current_top_num: int = 1
    current_nodes: List[Type[BaseNode]] = []
    final_answer_nodes: List[Type[BaseNode]] = [] 
    candidate_nodes: List[Type[BaseNode]] = [] 
    rollout_idx: int = 0
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.config.prompt_wrap == "rstar":
            from .utils import rstar_prompt_wrap, rstar_obs_wrap, rstar_step_result_unwrap

            self.prompt_wrap = rstar_prompt_wrap
            self.obs_wrap = rstar_obs_wrap
            self.step_unwrap = rstar_step_result_unwrap

        self.candidate_nodes.append(self.current_node)
        self.current_top_num = self.config.step_beam_width

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.n_generate_sample >= 1:
            raise ValueError(f"Wrong value for config n_generate_sample, must be greater than 1")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg
    

    def is_ignored_node(self, node: Type[BaseNode]) -> bool:
        return node.is_terminal or node.depth > self.config.max_depth

    def should_generate_next(self) -> bool:
        need_generate = False
        for step_node in self.current_nodes:
            if not self.is_ignored_node(step_node):
                need_generate = True
                break
        return need_generate
    
    def has_expanded(self) -> bool:
        if not self.current_nodes:
            return False
        step_node = self.current_nodes[0]
        if step_node.has_children():
            return True
        return False

    def get_rewards(self):
        rewards = []
        for node in self.current_nodes:
            rewards.append(node.reward if node.reward is not None else 0) # default reward is 0
        return rewards

    def create_prompt(
        self,
        is_value_only: bool = False,
    ) -> str:
        """
        if is_value_only, the prompt is used to produce value estimate.
        """
        prompts = []
        current_nodes = self.candidate_nodes if is_value_only else self.current_nodes
        for current_node in current_nodes:
            if not is_value_only and self.is_ignored_node(current_node):
                continue
            partial_solution = self.collect_partial_solution(current_node)
            prompt = self.prompt_wrap(
                self.question, 
                partial_solution,
                self.config
            )
            if is_value_only:
                prompt = {
                    "prefix": "",
                    "text": prompt,
                }
            prompts.append(prompt)
        return prompts
    
    @staticmethod
    def is_valid_final_answer_node(node: Type[BaseNode]) -> bool:
        if node.is_terminal and node.state["final_answer"] and \
           node.state["final_answer"] not in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            return True
        return False
    
    def select_next_step(self, outputs=None, from_root=False) -> None:
        self.current_nodes = []
        if outputs is not None:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                candidate_node.value = output.value_estimate if output.value_estimate is not None else 0
            
        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)
        self.current_nodes = self.candidate_nodes[:]

        for current_node in self.current_nodes[:]: 
            if self.__class__.is_valid_final_answer_node(current_node):
                self.final_answer_nodes.append(current_node)
                self.current_nodes.remove(current_node)
                self.current_top_num -= 1
            elif current_node.is_terminal or current_node.depth > self.config.max_depth:
                self.current_nodes.remove(current_node)
        self.current_nodes = self.candidate_nodes[:self.current_top_num]
        
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            self.current_node = current_node
            for idx, output in enumerate(output.outputs):
                if not output.stop_reason: output.stop_reason = ""
                step_result, parser_result = self.step_unwrap(output.text + output.stop_reason)
                self.create_child(step_result, parser_result, current_node)
            self.candidate_nodes.extend(current_node.children)

    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        return BaseNode(
            parent=parent, 
            additional_state_keys=self.NODE_KEYS,
        )
    
    def create_child(
        self, 
        step_result: str, 
        parser_result: Dict[str, str], 
        node: Type[BaseNode],
    ) -> None:
        new_node = self.create_node(parent=node)
        parent_child_count = len(node.children)
        new_node.tag = f"{node.tag}.{parent_child_count + 1}"
        new_node.depth = node.depth + 1

        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
        elif parser_result["action"]:
            observation = code_execution(node, parser_result)
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["observation"] = observation
            if CODE_END in parser_result["action_input"]:
                observation = self.obs_wrap(observation)
                new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
                if "Error" in observation:
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
            else:
                new_node.state["text"] = step_result
                
            if "error" in observation.lower():
                observation = self.obs_wrap(observation)
                step_result = step_result + CODE_END if CODE_END not in step_result else step_result
                new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
                new_node.is_terminal = True
                new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS

        else:
            new_node.state["text"] = step_result

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS

        node.children.append(new_node)

    def get_steps(self):
        final_answer_states = []
        for cur_node in self.final_answer_nodes:
            states = {
                "question": self.question,
                "ground_truth": self.ground_truth,
                "value": cur_node.value,
                "final_answer": cur_node.state["final_answer"],
                "solution": self.collect_partial_solution(cur_node),
                "tag": cur_node.tag,
            }
            final_answer_states.append(states)

        solutions = sorted(final_answer_states, key=lambda x: x['value'], reverse=True)
        return solutions

    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            if node.has_children():
                candidates.extend(node.children)
        states["solutions"] = self.get_steps()
        return states
