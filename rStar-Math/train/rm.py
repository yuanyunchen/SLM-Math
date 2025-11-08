# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from dataclasses import dataclass
from transformers import Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union, Any
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers import EvalPrediction
from datasets import load_dataset
import numpy as np
from numpy.typing import NDArray
IGNORE_INDEX = -100
@dataclass
class ComputeAccuracy:
    r"""
    Computes reward accuracy and supports `batch_eval_metrics`.
    """
    def numpify(inputs: Union["NDArray", "torch.Tensor"]):
        r"""
        Casts a torch tensor or a numpy array to a numpy array.
        """
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu()
            inputs = inputs.numpy()

        return inputs

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds, compute_result: bool = True) -> Optional[Dict[str, float]]:
        return {"accuracy": np.mean(eval_preds.predictions[0] > eval_preds.predictions[1])}

@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        concatenated_features = []
        for key in ("pos", "neg"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "factor": feature["factor"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


class ValueHead(nn.Module):
    
    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        #self.dropout = nn.Identity()
        hidden_size = 4096
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        linear_tpye = kwargs.get("linear_tpye", "single")
        if linear_tpye == "single":
            self.summary = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.summary.weight, mean=5e-7, std=1e-6)
            nn.init.constant_(self.summary.bias, 1e-6)
        else:
            raise ValueError("linear_tpye must be single or double")
        


    def forward(self, hidden_states):

        # if hidden_states.device != self.summary.weight.device:
        #     hidden_states = hidden_states.to(self.summary.weight.device)
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        try:
            if output.dtype != self.summary.weight.dtype:
                output = output.to(self.summary.weight.dtype)
        except:
            if output.dtype != self.summary[0].weight.dtype:
                output = output.to(self.summary[0].weight.dtype)

        output = self.summary(output)
        return output

class RewardModelWithValueHead(nn.Module):
    
    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        self.v_head = ValueHead(self.config, **kwargs)
        if hasattr(pretrained_model, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable = pretrained_model.gradient_checkpointing_disable

        if hasattr(pretrained_model, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable = pretrained_model.gradient_checkpointing_enable

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values
        if "factor" in kwargs:
            kwargs.pop('factor')
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]

        # if last_hidden_state.device != self.v_head.summary.weight.device:
        #     last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)

        if return_past_key_values:
            return (value, base_model_output.past_key_values)
        else:
            return value

    
def preprocess_value_dataset(
    examples,
    tokenizer,
    max_length=2048,
):
    model_inputs = {"pos_input_ids": [], "pos_attention_mask": [], "neg_input_ids": [], "neg_attention_mask": [], 
                    "pos_labels": [], "neg_labels": [],"factor": []}

    for i in range(len(examples["prompt"])):

        question = examples["prompt"][i]
        neg = examples["neg"][i]
        pos = examples["pos"][i]
        neg_count = examples["neg_count"][i]
        pos_count = examples["pos_count"][i]
        
        source_ids = tokenizer.encode(question, add_special_tokens=False, padding=False, truncation=False)
        source_mask = [1] * len(source_ids)

        pos_ids = tokenizer.encode(pos, add_special_tokens=False, padding=False, truncation=False)
        pos_mask = [1] * len(pos_ids)
        pos_ids = source_ids + pos_ids
        pos_mask = source_mask + pos_mask
        pos_labels = [IGNORE_INDEX] * len(source_ids) + pos_ids
        
        neg_ids = tokenizer.encode(neg, add_special_tokens=False, padding=False, truncation=False)
        neg_mask = [1] * len(neg_ids)
        neg_ids = source_ids + neg_ids
        neg_mask = source_mask + neg_mask
        neg_labels = [IGNORE_INDEX] * len(source_ids) + neg_ids
        
        
        if len(pos_ids) > max_length:
            pos_ids = pos_ids[:max_length]
            pos_mask = pos_mask[:max_length]
            pos_labels = pos_labels[:max_length]
        if len(neg_ids) > max_length:
            neg_ids = neg_ids[:max_length]
            neg_mask = neg_mask[:max_length]
            neg_labels = neg_labels[:max_length]
            
        if neg_count == 0 or pos_count == 0:
            factor = 1 / (neg_count + pos_count)
        else:
            factor = 1 / (neg_count * pos_count)
        
        model_inputs["pos_input_ids"].append(pos_ids)
        model_inputs["pos_attention_mask"].append(pos_mask)
        model_inputs["pos_labels"].append(pos_labels)
        model_inputs["neg_input_ids"].append(neg_ids)
        model_inputs["neg_attention_mask"].append(neg_mask)
        model_inputs["neg_labels"].append(neg_labels)
        model_inputs["factor"].append(factor)

    return model_inputs
    
class RMTrainer(Trainer):
    r"""
    Inherits Trainer to compute pairwise loss.
    """

    def __init__(
        self, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.can_return_loss = True  # override property to return eval_loss


    def compute_loss(
        self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        factor = inputs.get("factor", None)
        if factor is not None:
            del inputs["factor"]
        batch_size = inputs["input_ids"].size(0) // 2
        values = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        chosen_masks, rejected_masks = torch.split(inputs["attention_mask"], batch_size, dim=0)
        chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)
        factor, _ = torch.split(factor, batch_size, dim=0)
        chosen_scores = chosen_rewards.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1))
        rejected_scores = rejected_rewards.gather(dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1))
        chosen_scores, rejected_scores = chosen_scores.squeeze(), rejected_scores.squeeze()
        
        loss = -F.logsigmoid(chosen_scores.float() - rejected_scores.float())
        weighted_loss = loss * factor
        final_loss = weighted_loss.sum()

        if return_outputs:
            return final_loss, (final_loss, chosen_scores, rejected_scores)
        return final_loss
    
