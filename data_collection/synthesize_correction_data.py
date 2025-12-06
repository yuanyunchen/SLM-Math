"""
Synthesize Self-Correction Training Data

This script generates synthetic "error + correction" trajectories
for teaching the model to recover from mistakes.

Common error types to simulate:
1. Syntax errors (missing parentheses, typos)
2. Logic errors (wrong formula, off-by-one)
3. Runtime errors (division by zero, type errors)
4. Wrong variable usage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import re
from typing import Dict, List, Tuple


# Error injection templates
ERROR_TEMPLATES = {
    "division_by_zero": {
        "original": "result = {a} / {b}",
        "error": "result = {a} / 0  # Bug: should be {b}",
        "error_output": "ZeroDivisionError: division by zero",
        "fix_response": "I see there's a division by zero error. Let me fix it:",
        "fixed": "result = {a} / {b}"
    },
    "syntax_missing_paren": {
        "original": "print({expr})",
        "error": "print({expr}",  # Missing closing paren
        "error_output": "SyntaxError: '(' was never closed",
        "fix_response": "There's a syntax error - missing closing parenthesis. Let me fix it:",
        "fixed": "print({expr})"
    },
    "off_by_one": {
        "original": "for i in range({n}):",
        "error": "for i in range({n} - 1):  # Bug: off by one",
        "error_output": "{wrong_result}",  # Will show wrong answer
        "fix_response": "The result seems off. Let me check the loop boundary:",
        "fixed": "for i in range({n}):"
    },
    "wrong_operator": {
        "original": "result = {a} * {b}",
        "error": "result = {a} + {b}  # Bug: should be multiplication",
        "error_output": "{wrong_result}",
        "fix_response": "That doesn't look right. I should use multiplication, not addition:",
        "fixed": "result = {a} * {b}"
    },
    "type_error": {
        "original": "total = sum([{nums}])",
        "error": "total = sum({nums})  # Bug: not a list",
        "error_output": "TypeError: 'int' object is not iterable",
        "fix_response": "I need to pass a list to sum(). Let me fix it:",
        "fixed": "total = sum([{nums}])"
    }
}


def inject_error_into_trajectory(
    question: str,
    correct_code: str,
    correct_output: str,
    correct_answer: str,
    error_type: str = None
) -> Dict:
    """
    Take a correct trajectory and inject an error + correction.
    
    Returns a trajectory with error -> correction flow.
    """
    
    if error_type is None:
        # Choose random error type based on code content
        if "/" in correct_code and "0" not in correct_code:
            error_type = "division_by_zero"
        elif "range(" in correct_code:
            error_type = "off_by_one"
        elif "*" in correct_code:
            error_type = "wrong_operator"
        elif "sum(" in correct_code:
            error_type = "type_error"
        else:
            error_type = "syntax_missing_paren"
    
    # Generate error trajectory
    template = ERROR_TEMPLATES.get(error_type, ERROR_TEMPLATES["syntax_missing_paren"])
    
    # Create the error version
    error_code = introduce_error(correct_code, error_type)
    error_output = template["error_output"]
    fix_response = template["fix_response"]
    
    # Build the full trajectory
    trajectory = f"""Let me solve this step by step using Python.

```python
{error_code}
```
```output
{error_output}
```

{fix_response}

```python
{correct_code}
```
```output
{correct_output}
```

Based on the calculation, the answer is \\boxed{{{correct_answer}}}.
"""
    
    return {
        "query": question,
        "response": trajectory,
        "category": "SYNTHESIZED_CORRECTION",
        "error_type": error_type,
        "correct_answer": correct_answer
    }


def introduce_error(code: str, error_type: str) -> str:
    """Introduce a specific type of error into code."""
    
    if error_type == "division_by_zero":
        # Find division and replace divisor with 0
        code = re.sub(r'/\s*(\d+)', '/ 0  # Bug', code, count=1)
    
    elif error_type == "syntax_missing_paren":
        # Remove a closing parenthesis
        if code.count(')') > 0:
            idx = code.rfind(')')
            code = code[:idx] + code[idx+1:]
    
    elif error_type == "off_by_one":
        # Change range(n) to range(n-1)
        code = re.sub(r'range\((\w+)\)', r'range(\1 - 1)', code, count=1)
    
    elif error_type == "wrong_operator":
        # Change * to +
        code = re.sub(r'\*', '+', code, count=1)
    
    elif error_type == "type_error":
        # Remove list brackets from sum
        code = re.sub(r'sum\(\[([^\]]+)\]\)', r'sum(\1)', code, count=1)
    
    return code


def create_correction_examples_from_gsm8k(
    n_examples: int = 100,
    output_file: str = "data/agent_sft/synthesized_corrections.json"
):
    """
    Create correction examples using GSM8K problems.
    """
    from evaluation.eval_pipeline import load_dataset_by_name
    
    dataset = load_dataset_by_name("gsm8k")
    
    examples = []
    
    # Sample problems
    indices = random.sample(range(len(dataset)), min(n_examples * 2, len(dataset)))
    
    for idx in indices:
        if len(examples) >= n_examples:
            break
            
        item = dataset[idx]
        question = item['question']
        answer = item['answer']
        
        # Generate a simple correct code for this problem
        # (In practice, you'd use actual collected trajectories)
        
        # Extract numbers from question for simple arithmetic
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            a, b = int(numbers[0]), int(numbers[1])
            
            # Create a simple calculation
            correct_code = f"""# Calculate the answer
a = {a}
b = {b}
result = a + b
print(result)"""
            
            correct_output = str(a + b)
            
            # Inject error
            trajectory = inject_error_into_trajectory(
                question=question,
                correct_code=correct_code,
                correct_output=correct_output,
                correct_answer=answer,
                error_type=random.choice(["division_by_zero", "wrong_operator", "syntax_missing_paren"])
            )
            
            examples.append(trajectory)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(examples)} synthesized correction examples")
    print(f"Saved to: {output_path}")
    
    return examples


def augment_real_trajectories_with_errors(
    trajectory_file: str,
    output_file: str,
    error_ratio: float = 0.3
):
    """
    Take real successful trajectories and augment some with injected errors.
    
    This creates more natural-looking error correction data.
    """
    
    with open(trajectory_file, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)
    
    augmented = []
    
    for traj in trajectories:
        # Keep original
        augmented.append(traj)
        
        # Randomly add error version
        if random.random() < error_ratio:
            response = traj.get('response', '')
            
            # Find code blocks
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            output_match = re.search(r'```output\n(.*?)\n```', response, re.DOTALL)
            
            if code_match and output_match:
                code = code_match.group(1)
                output = output_match.group(1)
                answer = traj.get('predicted_answer', '')
                
                if code and output and answer:
                    error_traj = inject_error_into_trajectory(
                        question=traj['query'],
                        correct_code=code,
                        correct_output=output,
                        correct_answer=answer
                    )
                    augmented.append(error_traj)
    
    # Save
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    
    print(f"Original: {len(trajectories)}, Augmented: {len(augmented)}")
    print(f"Saved to: {output_path}")
    
    return augmented


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthesize", "augment"], default="synthesize")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default="data/agent_sft/synthesized_corrections.json")
    
    args = parser.parse_args()
    
    if args.mode == "synthesize":
        create_correction_examples_from_gsm8k(
            n_examples=args.n_examples,
            output_file=args.output_file
        )
    else:
        if args.input_file:
            augment_real_trajectories_with_errors(
                trajectory_file=args.input_file,
                output_file=args.output_file
            )
        else:
            print("Error: --input-file required for augment mode")







