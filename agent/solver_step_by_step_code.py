"""
Solver with Step-by-Step Code Execution

This agent breaks down problems into sub-steps, solving each step with code.
The model sees the output of each step before moving to the next.

Flow:
1. Analyze the problem and identify steps
2. For each step: write code -> execute -> see result
3. Combine results for final answer
"""

import re
import torch
from typing import Dict, List, Tuple, Optional
from transformers import StoppingCriteria, StoppingCriteriaList
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY
)


class StopOnCodeBlockOrBoxed(StoppingCriteria):
    """Stop generation when ``` (code block end) or \\boxed is detected."""
    
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_length:].tolist()
        if len(generated_ids) < 3:
            return False
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = generated_text.rstrip()
        
        # Stop at code block end (but not ```python or ```output)
        if text.endswith("```") and not text.endswith("```python") and not text.endswith("```output"):
            return True
        
        # Stop at boxed answer
        if "\\boxed{" in text and text.count("{") <= text.count("}"):
            # Check if boxed is complete
            boxed_match = re.search(r'\\boxed\{[^}]+\}', text)
            if boxed_match:
                return True
        
        return False


def execute_code_safe(code: str, timeout: int = 10, shared_state: dict = None) -> dict:
    """Execute code safely with timeout and shared state."""
    from utils.python_code_execution import execute_python_code_with_state
    
    exec_globals = shared_state.copy() if shared_state else {}
    exec_locals = {}
    
    result = execute_python_code_with_state(
        code, 
        exec_globals=exec_globals,
        exec_locals=exec_locals,
        timeout=timeout
    )
    
    # Return state for sharing
    if result.get('success'):
        result['state'] = {**exec_globals, **exec_locals}
    
    return result


def format_output_block(exec_result: dict) -> str:
    """Format execution result as output block."""
    if exec_result['success']:
        output = exec_result['output'].strip()
        if output:
            return f"```output\n{output}\n```"
        else:
            return "```output\n[No output]\n```"
    else:
        error = exec_result.get('error', 'Unknown error')
        return f"```output\nError: {error}\n```"


def generate_step_by_step(
    question: str,
    model,
    tokenizer,
    dataset_name: str = "",
    max_new_tokens: int = 2048,
    max_steps: int = 8,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    detailed: bool = False
) -> Tuple[str, List[dict]]:
    """
    Generate solution step by step, executing code at each step.
    
    Returns:
        Tuple of (final_response, execution_results)
    """
    
    # Build prompt that encourages step-by-step solving
    prompt = f"""Solve this math problem step by step. Break it down into smaller steps.

Problem: {question}

Instructions:
- Break the problem into clear steps
- For each step, write Python code to calculate
- After seeing the output, move to the next step
- Finally, give your answer in \\boxed{{}}

Solve step by step:

Step 1: Let me identify what we need to find and the given information.
```python
"""
    
    execution_results = []
    shared_state = {}
    current_context = prompt
    total_generated_tokens = 0
    step_count = 0
    
    if detailed:
        print(f"\n[Step-by-Step Code Generation]")
        print(f"  Max tokens: {max_new_tokens}")
        print(f"  Max steps: {max_steps}")
    
    while total_generated_tokens < max_new_tokens and step_count < max_steps:
        step_count += 1
        
        # Tokenize current context
        inputs = tokenizer(current_context, return_tensors="pt").to(model.device)
        prompt_length = inputs['input_ids'].shape[1]
        remaining_tokens = max_new_tokens - total_generated_tokens
        
        if detailed:
            print(f"\n[Step {step_count}]")
            print(f"  Context: {prompt_length} tokens, Budget: {remaining_tokens} tokens")
        
        # Create stopping criteria
        stop_criteria = StoppingCriteriaList([
            StopOnCodeBlockOrBoxed(tokenizer, prompt_length)
        ])
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(remaining_tokens, 512),  # Limit per step
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop_criteria,
            )
        
        # Decode new tokens
        new_tokens = outputs[0][prompt_length:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens_generated = len(new_tokens)
        total_generated_tokens += tokens_generated
        
        if detailed:
            preview = new_text[:150].replace('\n', '\\n')
            print(f"  Generated: {tokens_generated} tokens")
            print(f"  Preview: {preview}...")
        
        # Update response
        response_so_far = current_context[len(prompt):] + new_text
        
        # Check if we have a complete answer
        if "\\boxed{" in new_text:
            if detailed:
                print(f"  Found \\boxed answer, stopping.")
            current_context = prompt + response_so_far
            break
        
        # Check for code block that needs execution
        # Look for code that ends with ```
        full_text = current_context + new_text
        
        # Find the last ```python block
        code_pattern = r'```python\s*\n(.*?)```'
        matches = list(re.finditer(code_pattern, full_text, re.DOTALL))
        
        if matches:
            last_match = matches[-1]
            block_end = last_match.end()
            
            # Check if this block already has output
            remaining_after_block = full_text[block_end:]
            if remaining_after_block.strip().startswith('```output'):
                # Already executed, continue
                current_context = full_text
                continue
            
            # Extract and execute code
            code = last_match.group(1).strip()
            
            if detailed:
                print(f"  Executing code: {code[:80]}...")
            
            exec_result = execute_code_safe(code, timeout=10, shared_state=shared_state)
            execution_results.append({
                'step': step_count,
                'code': code,
                'success': exec_result['success'],
                'output': exec_result['output'],
                'error': exec_result.get('error', '')
            })
            
            # Update shared state
            if exec_result.get('state'):
                for key, value in exec_result['state'].items():
                    if not callable(value) and not key.startswith('__'):
                        try:
                            import json
                            json.dumps(value)
                            shared_state[key] = value
                        except:
                            pass
            
            if detailed:
                if exec_result['success']:
                    print(f"  Output: {exec_result['output'][:80]}...")
                else:
                    print(f"  Error: {exec_result.get('error', '')[:80]}...")
            
            # Inject output and prompt for next step
            output_block = format_output_block(exec_result)
            current_context = full_text[:block_end] + "\n" + output_block + "\n\n"
            
            # Add prompt for next step if not at final answer
            if step_count < max_steps - 1:
                current_context += f"Step {step_count + 1}: "
        else:
            # No code block, might be reasoning text
            current_context = full_text
            
            # Check for EOS
            if tokens_generated < min(remaining_tokens, 512):
                if detailed:
                    print(f"  Generation ended (EOS)")
                break
    
    # Extract final response
    final_response = current_context[len(prompt):]
    
    if detailed:
        print(f"\n[Generation Complete]")
        print(f"  Total tokens: {total_generated_tokens}")
        print(f"  Steps executed: {len(execution_results)}")
    
    return final_response, execution_results


def run_solver_step_by_step_code(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    max_iterations: int = 1,
    max_steps: int = 8,
    detailed: bool = False,
    dataset_name: str = "",
) -> Dict:
    """
    Run solver with step-by-step code execution.
    """
    from utils.prompt_utils import extract_answer, check_answer
    
    if detailed:
        print("\n" + "=" * 80)
        print("SOLVER WITH STEP-BY-STEP CODE")
        print("=" * 80)
        print(f"Question: {question[:100]}...")
        print(f"Max steps: {max_steps}")
        print("=" * 80)
    
    iterations = []
    
    for iteration in range(max_iterations):
        if detailed:
            print(f"\n[Iteration {iteration + 1}/{max_iterations}]")
        
        response, execution_results = generate_step_by_step(
            question=question,
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            max_new_tokens=2048,
            max_steps=max_steps,
            detailed=detailed
        )
        
        # Extract answer
        predicted_answer = extract_answer(response)
        is_correct = check_answer(predicted_answer, ground_truth)
        
        iterations.append({
            'iteration': iteration + 1,
            'response': response,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'execution_results': execution_results
        })
        
        if detailed:
            print(f"\n[Extracted Answer]: {predicted_answer}")
        
        if is_correct:
            break
    
    # Get final result
    final_iteration = iterations[-1]
    
    result = {
        'question': question,
        'ground_truth': ground_truth,
        'predicted_answer': final_iteration['predicted_answer'],
        'final_correct': final_iteration['is_correct'],
        'first_try_correct': iterations[0]['is_correct'],
        'iterations': iterations,
        'total_iterations': len(iterations),
        'total_code_executions': sum(len(it.get('execution_results', [])) for it in iterations),
        'execution_results': final_iteration.get('execution_results', [])
    }
    
    return result

