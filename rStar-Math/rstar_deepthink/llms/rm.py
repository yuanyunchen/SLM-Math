# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch.nn as nn

class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
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
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = base_model_output.hidden_states[-1]
        value = self.v_head(last_hidden_state).squeeze(-1)
        if return_past_key_values:
            return (value, base_model_output.past_key_values)
        else:
            return value
