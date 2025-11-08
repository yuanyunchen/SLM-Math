# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
import random
from typing import List, Optional, Literal
from enum import Enum, EnumMeta
from dataclasses import dataclass, field


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


_SEARCH_CHOICES = [
    "mcts",  # monte carlo tree search
    "bs",    # beam search
]

SEARCH_CHOICES = ChoiceEnum(_SEARCH_CHOICES)


_PROMPT_CHOICES = [
    "rstar", 
]

PROMPT_CHOICES = ChoiceEnum(_PROMPT_CHOICES)
@dataclass
class BaseConfig:

    mode: SEARCH_CHOICES = field(
        default="mcts", metadata={"help": "search mode for inference"}
    )
    model_dir: Optional[str] = field(
        default=None, metadata={"help": "llm model dir"}
    )
    reward_model_dir: Optional[str] = field(
        default=None, metadata={"help": "reward model dir"}
    )
    few_shot_path: Optional[str] = field(
        default=None, metadata={"help": "few shot data json"}
    )
    prompt_path: Optional[str] = field(
        default=None, metadata={"help": "prompt config json"}
    )
    num_few_shot: int = field(
        default=0, metadata={"help": "the number of few-shot examples"}
    )
    # prompt args
    prompt_wrap: PROMPT_CHOICES = field(
        default="rstar", metadata={"help": "prompt wrap type"}
    )
    result_unwrap: PROMPT_CHOICES = field(
        default="rstar", metadata={"help": "result unwrap"}
    )
    step_delim: str = field(
        default="\n", metadata={"help": "delimiter between two steps"}
    )
    # vllm args
    temperature: float = field(
        default=0.7, metadata={"help": "control diversity of llm generation"}
    )
    top_p: float = field(
        default=1.0, metadata={"help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens."}
    )
    top_k: int = field(
        default=-1, metadata={"help": "Float that controls the probability of other highly-scored candidates to be chosen"}
    )
    use_beam_search: bool = field(
        default=False, metadata={"help": "whether to enable beam search decoding"}
    )
    best_of: int = field(
        default=1, metadata={"help": "Integer that controls the number of candidate considered in the beam search decoding process"}
    )
    max_tokens: int = field(
        default=2048, metadata={"help": "Maximum number of tokens to generate per output sequence."}
    )
    # seed: Optional[int] = field(
    #     default=random.randint(1, 100000), metadata={"help": "seed of llm generation for reproducible"}
    # )
    seed: Optional[int] = field(
        default=None, metadata={"help": "seed of llm generation for reproducible"}
    )
    swap_space: Optional[int] = field(
        default=8, metadata={"help": "swap space for vllm"}
    )
    n_generate_sample: int = field(
        default=1, metadata={"help": "how many samples generated for each step. B2 in paper."}
    )
    stop: Optional[List[str]] = field(
        default=None, metadata={"help": "possible stop tokens for each step"}
    )
    step_beam_width: int = field(
        default=1, metadata={"help": "beam width for each step. B1 in paper."}
    )
    max_depth: int = field(
        default=4, metadata={"help": "maximum depth of the tree, ie., maximum steps of completion."}
    )
    iterations: int = field(
        default=1, metadata={"help": "number of simulations in mcts"}
    )
    positive_reward: float = field(
        default=1.0, metadata={"help": "reward for positive example"}
    )
    negative_reward: float = field(
       default=-1.0, metadata={"help": "reward for negative example"}
    )
    errors_threshold: int = field(
        default=0, metadata={"help": "maximum code errors allowed, ie., if errors_count > errors_threshold, the tree growth from this node should be terminated."}
    )
    need_value_func: bool = field(
        default=False, metadata={"help": "whether to use value head in decoding"}
    )
    update_leaf_value: bool = field(
        default=False, metadata={"help": "update leaf value in mcts"}
    )
    c_puct: float = field(
        default=2, metadata={"help": "weight of c_puct in mcts"}
    )
    is_sampling: bool = field(
        default=False, metadata={"help": "solution generation in mcts"}
    )
    # offline inferene args 
    prune: bool = field(
        default=False, metadata={"help": "prune the tree in a complete mcts tree"}
    )
    # other args
    batch_size: int = field(
        default=-1, metadata={"help": "batch size for batch inference"}
    )
    max_model_len: int = field(
        default=8192, metadata={"help": "maximum model length"}
    )
    terminal_sample: bool = field(
        default=False
    )
    llm_gpu_memory_utilization: float = field(
        default=0.5, metadata={"help": "gpu memory utilization for policy. rm will use 15G memory, so the remaining memory is used for llm."}
    ) # if your gpu has 80G memory, set this to 0.7; if your gpu has 40G memory, set this to 0.5
    tp: int = field(
        default=1, metadata={"help": "llm tensor parallel size"}
    )
    save_intermediate_rollouts: bool = field(
        default=True, metadata={"help": "save intermediate rollouts in ./rollout"}
    )