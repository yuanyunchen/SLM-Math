"""
Convert rstar_sft dataset to Interactive Agent format

rstar_sft format:
<code>
# Step 1: ...
<end_of_step>
# Step 2: ...
<end_of_code>
<output>result<end_of_output>
<answer>...\boxed{...}<end_of_answer>

Interactive format:
Let me solve this step by step using Python.

```python
# Step 1: ...
# Step 2: ...
print(answer)
```
```output
result
```

The answer is \boxed{...}.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import json
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm
from datasets import load_from_disk, Dataset


def convert_single_response(response: str) -> Optional[str]:
    """
    Convert a single rstar_sft response to interactive format.
    
    Returns None if conversion fails.
    """
    try:
        # Extract code block
        code_match = re.search(r'<code>\s*(.*?)\s*<end_of_code>', response, re.DOTALL)
        if not code_match:
            return None
        
        code_content = code_match.group(1)
        
        # Remove <end_of_step> tags (keep the structure as comments naturally flow)
        code_content = re.sub(r'\s*<end_of_step>\s*', '\n\n', code_content)
        code_content = code_content.strip()
        
        # Extract output
        output_match = re.search(r'<output>(.*?)<end_of_output>', response, re.DOTALL)
        output_content = output_match.group(1).strip() if output_match else "(No output)"
        
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)<end_of_answer>', response, re.DOTALL)
        answer_content = answer_match.group(1).strip() if answer_match else ""
        
        # Extract boxed answer for final statement
        boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', answer_content)
        
        # Build interactive format response
        parts = []
        
        # Add intro (varied slightly for diversity)
        intro_options = [
            "Let me solve this step by step using Python.",
            "I'll solve this problem using Python code.",
            "Let me work through this with Python.",
        ]
        # Use hash of code to select intro deterministically
        intro = intro_options[hash(code_content) % len(intro_options)]
        parts.append(intro)
        parts.append("")
        
        # Add code block
        parts.append("```python")
        parts.append(code_content)
        parts.append("```")
        
        # Add output block
        parts.append("```output")
        parts.append(output_content)
        parts.append("```")
        parts.append("")
        
        # Add answer statement
        if boxed_match:
            boxed_value = boxed_match.group(0)  # Full \boxed{...}
            parts.append(f"The answer is {boxed_value}.")
        elif answer_content:
            parts.append(answer_content)
        
        return "\n".join(parts)
        
    except Exception as e:
        return None


def convert_dataset(
    input_path: str = "data/rstar_sft",
    output_path: str = "data/rstar_sft_interactive",
    output_json: str = "data/rstar_sft_interactive/rstar_sft_interactive.json",
    max_samples: int = 0,
    save_failed: bool = True
):
    """
    Convert entire rstar_sft dataset to interactive format.
    
    Args:
        input_path: Path to rstar_sft dataset
        output_path: Path to save converted dataset (HuggingFace format)
        output_json: Path to save as JSON
        max_samples: Max samples to convert (0 = all)
        save_failed: Whether to save failed conversions for inspection
    """
    print(f"Loading dataset from {input_path}...")
    ds = load_from_disk(input_path)
    
    train_data = ds['train']
    total = len(train_data)
    
    if max_samples > 0:
        total = min(total, max_samples)
        print(f"Converting first {total} samples...")
    else:
        print(f"Converting all {total} samples...")
    
    converted = []
    failed = []
    
    for i in tqdm(range(total), desc="Converting"):
        item = train_data[i]
        query = item['query']
        response = item['response']
        
        converted_response = convert_single_response(response)
        
        if converted_response:
            converted.append({
                'query': query,
                'response': converted_response,
                'original_response': response  # Keep original for reference
            })
        else:
            failed.append({
                'index': i,
                'query': query,
                'response': response
            })
    
    print(f"\nConversion complete:")
    print(f"  Successfully converted: {len(converted)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Success rate: {len(converted)/total*100:.2f}%")
    
    # Save as JSON
    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON to: {output_json_path}")
    
    # Save as HuggingFace Dataset
    output_ds_path = Path(output_path)
    output_ds_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset without original_response for training
    train_records = [{'query': item['query'], 'response': item['response']} for item in converted]
    hf_dataset = Dataset.from_list(train_records)
    hf_dataset.save_to_disk(str(output_ds_path / "train"))
    
    print(f"Saved HuggingFace dataset to: {output_ds_path}")
    
    # Save failed conversions for inspection
    if save_failed and failed:
        failed_path = output_ds_path / "failed_conversions.json"
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(failed)} failed conversions to: {failed_path}")
    
    # Show sample conversions
    print("\n" + "="*80)
    print("Sample Conversions (first 2):")
    print("="*80)
    
    for i, item in enumerate(converted[:2]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Query: {item['query'][:100]}...")
        print(f"\nConverted Response:\n{item['response'][:600]}...")
    
    return converted, failed


def convert_for_sft_training(
    input_json: str = "data/rstar_sft_interactive/rstar_sft_interactive.json",
    output_file: str = "data/rstar_sft_interactive/train_sft.json",
    format_type: str = "chatml"
):
    """
    Convert to specific SFT training format (ChatML, Alpaca, etc.)
    
    Args:
        input_json: Path to converted interactive JSON
        output_file: Output file path
        format_type: "chatml" or "alpaca"
    """
    print(f"Loading converted data from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted = []
    
    for item in tqdm(data, desc=f"Formatting to {format_type}"):
        query = item['query']
        response = item['response']
        
        if format_type == "chatml":
            # ChatML format for Qwen
            formatted.append({
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response}
                ]
            })
        elif format_type == "alpaca":
            # Alpaca format
            formatted.append({
                "instruction": query,
                "input": "",
                "output": response
            })
        else:
            # Simple format
            formatted.append({
                "query": query,
                "response": response
            })
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(formatted)} samples to: {output_path}")
    
    return formatted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rstar_sft to interactive format")
    parser.add_argument("--input", type=str, default="data/rstar_sft", help="Input dataset path")
    parser.add_argument("--output", type=str, default="data/rstar_sft_interactive", help="Output path")
    parser.add_argument("--output-json", type=str, default="data/rstar_sft_interactive/rstar_sft_interactive.json")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--format-sft", action="store_true", help="Also create SFT training format")
    parser.add_argument("--sft-format", type=str, default="chatml", choices=["chatml", "alpaca", "simple"])
    
    args = parser.parse_args()
    
    # Convert dataset
    converted, failed = convert_dataset(
        input_path=args.input,
        output_path=args.output,
        output_json=args.output_json,
        max_samples=args.max_samples
    )
    
    # Optionally create SFT format
    if args.format_sft:
        sft_output = f"{args.output}/train_sft_{args.sft_format}.json"
        convert_for_sft_training(
            input_json=args.output_json,
            output_file=sft_output,
            format_type=args.sft_format
        )





