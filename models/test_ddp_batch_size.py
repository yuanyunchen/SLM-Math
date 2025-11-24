"""
Test DDP training with different batch sizes to find optimal configuration.
"""

import os
import sys
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

# System prompt
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve problems using step-by-step reasoning with clear explanations. "
    "Always provide your final answer in \\boxed{} format."
)

def formatting_prompts(example, tokenizer):
    """Format training examples into chat template."""
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                example["question"]
                + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            ),
        },
        {
            "role": "assistant",
            "content": example["raw_output"],
        },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def test_batch_size(batch_size, max_steps=3):
    """Test a specific batch size."""
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank <= 0:
        print(f"\n{'='*60}")
        print(f"Testing batch_size = {batch_size}")
        print(f"{'='*60}")
    
    # Load data
    data_path = "data/cot_generated/first_round_final_gsm8k_math/first_round_final_gsm8k_math.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Use small subset
    df = pd.DataFrame(data[:50])
    if "correct" in df.columns:
        df = df[df["correct"] == True].reset_index(drop=True)
    
    dataset = Dataset.from_pandas(df)
    
    # Load model
    model_name = "pretrained_models/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Move to cuda if single process
    if local_rank == -1 and torch.cuda.is_available():
        model = model.cuda()
    
    def format_func(example):
        return formatting_prompts(example, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/test_ddp_batch",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=False,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to=[],
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        max_steps=max_steps,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=format_func,
        args=training_args,
    )
    
    # Train
    import time
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    
    if local_rank <= 0:
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        steps_per_sec = max_steps / elapsed_time
        print(f"✓ SUCCESS")
        print(f"  Time: {elapsed_time:.2f}s ({steps_per_sec:.2f} steps/sec)")
        print(f"  GPU memory: {max_mem:.2f} GB")
    
    del trainer, model
    torch.cuda.empty_cache()
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])
        try:
            test_batch_size(batch_size)
            sys.exit(0)
        except RuntimeError as e:
            if "out of memory" in str(e):
                local_rank = int(os.environ.get("LOCAL_RANK", -1))
                if local_rank <= 0:
                    print(f"✗ OOM: batch_size={batch_size} too large")
                sys.exit(1)
            else:
                raise
    else:
        print("Usage: torchrun --nproc_per_node=2 test_ddp_batch_size.py <batch_size>")
        sys.exit(1)

