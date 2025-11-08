# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import os
import json
import os.path as osp
from tqdm import tqdm
from termcolor import colored
from functools import partial
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from pebble import ProcessPool
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
from pydantic import BaseModel, ConfigDict, field_validator
from .agents.tree import BaseTree
from .agents.mcts import MCTS
from .llms.llms import llm_generate, rm_generate
from .llms.llm_engine import llm_engine, rm_engine
from .constants import TIMEOUT_SECONDS, ERROR_COLOR


class Solver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: Any
    stop: List[str] = None
    llm: Optional[Callable[[...], List[str]]] = None
    llm_engine: Optional[LLM] = None
    generate_sampling_params: Optional[SamplingParams] = None
    need_value_func: bool = False
    max_agent_steps: int = 1
    reward_model: Optional[Any] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.config.stop:
            self.stop = OmegaConf.to_object(self.config.stop)

        self.need_value_func = self.config.need_value_func
        if self.need_value_func:
            self.reward_model = self.create_rm()
        self.llm = self.create_llm()
        if self.config.mode == "sbs":
            self.max_agent_steps = 1
        elif self.config.mode == "mcts":
            self.max_agent_steps = self.config.iterations
            self.config.step_beam_width = 1
            

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            return cfg
        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")


    def create_rm(self):
        rm, v_head, tokenizer = rm_engine(self.config)
        return partial(
            rm_generate,
            model=rm,
            v_head=v_head,
            tokenizer=tokenizer,
            max_model_len=self.config.max_model_len,
        )


    def create_llm(self):
        engine, sampling_params = llm_engine(self.config)
        self.llm_engine = engine
        self.generate_sampling_params = sampling_params
        return partial(
            llm_generate,
            engine=self.llm_engine,
        )

        
    @staticmethod
    def processor(agent, output) -> BaseTree:
        agent.generate_next_step(output)
        return agent


    @staticmethod
    def selector(agent, output) -> BaseTree:
        agent.select_next_step(output)
        return agent


    def generate_preprocess(self, agents):
        prompts = []
        rewards = []
        prompts_span = [0]
        valid_agents = []
        invalid_agents = []
        expanded_agents = []

        for agent in agents:
            if agent.should_generate_next():
                if agent.has_expanded():
                    expanded_agents.append(agent)
                else:
                    agent_prompts = agent.create_prompt()
                    rewards.extend(agent.get_rewards())
                    prompts.extend(agent_prompts)
                    prompts_span.append(prompts_span[-1] + len(agent_prompts))
                    valid_agents.append(agent)
            else:
                invalid_agents.append(agent)
        return prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, rewards


    def generate_postprocess(
        self, 
        outputs: List[List[RequestOutput]], 
        valid_agents: List[BaseTree],
    ) -> List[BaseTree]:
        post_agents = []
        #with ProcessPool(max_workers=min(len(valid_agents), os.cpu_count())) as pool:
        with ProcessPool(max_workers=12) as pool:
            future = pool.map(self.__class__.processor, valid_agents, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()
        
        progress_bar = tqdm(total=len(valid_agents), desc="generate_postprocess")  
        while True:
            try:
                result = next(iterator)
                post_agents.append(result)
            except StopIteration:
                break
            except Exception as error:
                print(colored(f"{error}\n", ERROR_COLOR))
                post_agents.append(None)
            progress_bar.update(1) 
        progress_bar.close() 
            
        # update agents
        updated_agents = [
            post_agent if post_agent is not None else valid_agent
            for post_agent, valid_agent in zip(post_agents, valid_agents)
        ]
        return updated_agents
    

    def value_preprocess(self, agents: List[BaseTree]) -> Tuple[List[str], List[int]]:
        prompts = []
        prompts_span = [0]
        for agent in agents:
            agent_prompts = agent.create_prompt(is_value_only=True)
            prompts.extend(agent_prompts)
            prompts_span.append(prompts_span[-1] + len(agent_prompts))
        return prompts, prompts_span
    
    
    def value_postprocess(
        self, 
        outputs, 
        valid_agents,
    ) -> List[BaseTree]:
        for agent, output in zip(valid_agents, outputs):
            if agent is not None:
                self.selector(agent, output)
        return valid_agents
    

    def save_intermediate_metric(self, path: str, agents: List[MCTS], rollout) -> None:
        if self.config.is_sampling: return
        states = [s.intermediate_metric for s in agents]
        statics = []
        for i in range(rollout + 1):
            pass1, passn = 0, 0
            for idx, state in enumerate(states):
                max_value = -100
                max_value_result = False
                pass1_ans = False
                for idx, rollout_index in enumerate(state["rollout_indexs"]):
                    if rollout_index <= i:
                        if state["value_estimate"][idx] > max_value:
                            max_value = state["value_estimate"][idx]
                            max_value_result = state["judgements"][idx]
                        if state["judgements"][idx]:
                            pass1_ans = True
                if max_value_result:
                    pass1 += 1
                if pass1_ans:
                    passn += 1
            statics.append({
                "rollout": i,
                "pass1": pass1,
                "passn": passn,
                "len": len(states),
            })
        with open(path, "w", encoding='utf-8') as f:
            json.dump([statics,states], f, ensure_ascii=False, indent=4)

    
    def save_intermediate_rollouts(self, saved_jsonl_file, cur_data, agents, rollout_idx):
        if self.config.save_intermediate_rollouts and saved_jsonl_file and self.config.mode == "mcts":
            saved_json_dir = osp.dirname(saved_jsonl_file)
            saved_jsonl_file_name = osp.basename(saved_jsonl_file)
            saved_json_path = osp.join(saved_json_dir, f"rollout")
            if not os.path.exists(saved_json_path):
                os.mkdir(saved_json_path)
            self.save_intermediate_metric(osp.join(saved_json_path, f"intermediate_metric_{saved_jsonl_file_name}"), agents, rollout_idx)
            outs = self.output(agents)
            with open(osp.join(saved_json_path, f"rollout{rollout_idx:02}" + saved_jsonl_file_name), "a+", encoding='utf-8') as writer:
                for d in cur_data:
                    question = d["question"]
                    d["rstar"] = outs[question]
                    writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                    writer.flush()
    
    def output(self, agents: List[BaseTree]):
        jsonlines = {}
        for i, agent in enumerate(agents):         
            jsonlines[agent.question] = agent.return_states()
        
        return jsonlines
    
    def solve(self, agents: List[BaseTree], saved_jsonl_file: str, cur_data: List[Dict[str, Any]]):
        
        for rollout in tqdm(range(self.max_agent_steps), desc="Rollout Processing"):
            # Initialize the initial search starting point of agents, and the initial point of each rollout is root
            for agent in agents:
                agent.select_next_step(from_root=True)
                agent.rollout_idx = rollout

            for step in range(self.config.max_depth):
                print("-----------------Current Rollout: ", rollout, "-----------------")
                print("-----------------Current Step: ", step, "-----------------")
                prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, valid_rewards = self.generate_preprocess(agents)
                
                if len(valid_agents + expanded_agents) < 1:
                    break
                
                # step expansion
                outputs = self.llm(prompts, self.generate_sampling_params)
                
                for output, reward in zip(outputs, valid_rewards): # attach reward to prevent repeat rewarding
                    output.value_estimate = reward
                reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
                
                # process output and run python code
                valid_agents = self.generate_postprocess(reconstructed_outputs, valid_agents)

                # step evaluation
                prompts, prompts_span = self.value_preprocess(valid_agents)
                if self.need_value_func:
                    outputs = self.reward_model(prompts=prompts)
                    reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
                else:
                    reconstructed_outputs = [None] * (len(prompts_span) - 1)
                
                # selection
                valid_agents = self.value_postprocess(reconstructed_outputs, valid_agents)
                expanded_agents = self.value_postprocess([None] * len(expanded_agents), expanded_agents) # for expanded agents, just do selection step
                
                # keep all agents
                agents = valid_agents + invalid_agents + expanded_agents

            # Save agents internal rollouts
            self.save_intermediate_rollouts(saved_jsonl_file, cur_data, agents, rollout)
            
        return self.output(agents)
    
    
