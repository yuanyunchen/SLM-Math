# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import random

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class PROMPT_RSTAR:
    def __init__(self, config):
        self.pot_format_instructions = None
        self.pot_suffix = None
        self.few_examples = None
        self.num_few_shot = config.num_few_shot
        self.load_prompt(config)

        assert self.num_few_shot <= len(self.few_examples), f"{self.num_few_shot} should less than few_examples."   


    def load_prompt(self, config):
        self.few_examples = load_json(config.few_shot_path)
        prompt = load_json(config.prompt_path)
        self.pot_format_instructions = prompt['pot_format_instructions']
        self.pot_suffix = prompt['pot_suffix']


    def random_examples(self):
        if self.num_few_shot == 1:
            return [self.few_examples[1]]
        selected_examples = random.sample(self.few_examples, min(len(self.few_examples), self.num_few_shot))
        return selected_examples