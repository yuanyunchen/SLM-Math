"""
Solver with Interactive Code Execution
支持交互式代码执行的Solver

核心功能：
1. 模型生成时检测 ```python...``` 代码块
2. 代码块结束后立即执行
3. 将执行结果 ```output...``` 注入上下文
4. 模型继续生成，可以看到输出结果
5. 支持多次代码执行和错误处理

工作流程：
模型生成 → 检测代码块结束 → 执行代码 → 注入输出 → 继续生成 → ... → 最终答案
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import torch
from typing import Dict, List, Tuple, Optional
from transformers import StoppingCriteria, StoppingCriteriaList
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY
)


class StopOnCodeBlockEnd(StoppingCriteria):
    """Stop generation when ``` is detected (code block end)."""
    
    def __init__(self, tokenizer, prompt_length: int, prompt_ends_with_code_start: bool = False):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.prompt_ends_with_code_start = prompt_ends_with_code_start
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_length:].tolist()
        if len(generated_ids) < 3:
            return False
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # If prompt ends with ```python, we're generating code directly
        # Stop at first ``` that's not followed by python/output
        if self.prompt_ends_with_code_start:
            # Look for closing ```
            if "\n```" in generated_text:
                # Check it's not ```python or ```output
                match = re.search(r'\n```(?!python|output)', generated_text)
                if match:
                    return True
        else:
            # Normal case: look for complete ```python...``` block
            if "```python" in generated_text:
                last_python = generated_text.rfind("```python")
                after_python = generated_text[last_python + 9:]
                close_match = re.search(r'\n```(?!python|output)', after_python)
                if close_match:
                    return True
        
        return False


def detect_code_block_end(text: str, prompt_ends_with_code_start: bool = False) -> Tuple[bool, int, int, str]:
    """
    Detect if there's a complete ```python...``` block that needs execution.
    
    Returns:
        Tuple of (has_complete_block, block_start, block_end, code)
    """
    # If prompt ends with ```python, the text starts with code content
    if prompt_ends_with_code_start:
        # Look for closing ``` 
        match = re.search(r'^(.*?)\n```', text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            block_end = match.end()
            # Check if already has real output (from previous execution)
            remaining = text[block_end:]
            if remaining.strip().startswith('```output'):
                return False, -1, -1, ""  # Already executed
            return True, 0, block_end, code
        return False, -1, -1, ""
    
    # Find all ```python blocks
    pattern = r'```python\s*\n(.*?)\n```'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if not matches:
        return False, -1, -1, ""
    
    # Get the last code block
    last_match = matches[-1]
    block_end = last_match.end()
    
    # Check if already has output (from previous execution)
    remaining = text[block_end:]
    if remaining.strip().startswith('```output'):
        return False, -1, -1, ""  # Already executed
    
    code = last_match.group(1).strip()
    return True, last_match.start(), block_end, code


def execute_code_safe(code: str, timeout: int = 10, shared_state: Optional[Dict] = None) -> Dict:
    """Execute Python code safely with timeout and shared state."""
    import io
    import traceback
    from contextlib import redirect_stdout, redirect_stderr
    import signal
    
    result = {
        'success': False,
        'output': '',
        'error': '',
        'state': shared_state or {}
    }
    
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Set up timeout (Unix only)
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution timeout ({timeout}s)")
    
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    
    try:
        exec_globals = {
            '__builtins__': __builtins__,
        }
        if shared_state:
            exec_globals.update(shared_state)
        exec_locals = {}
        
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, exec_globals, exec_locals)
        
        # Update shared state with new variables
        if shared_state is not None:
            for key, value in exec_locals.items():
                if not key.startswith('_'):
                    try:
                        import json
                        json.dumps(value)  # Test if serializable
                        shared_state[key] = value
                    except (TypeError, ValueError):
                        pass
        
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        # Truncate long output
        max_output_len = 1000
        if len(stdout_output) > max_output_len:
            stdout_output = stdout_output[:max_output_len] + "\n... (output truncated)"
        
        result['success'] = True
        result['output'] = stdout_output.strip()
        if stderr_output:
            result['error'] = stderr_output.strip()
        result['state'] = shared_state or {}
        
    except TimeoutError as e:
        result['error'] = str(e)
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
    
    return result


def format_output_block(exec_result: Dict) -> str:
    """Format execution result as ```output``` block."""
    if exec_result['success']:
        output = exec_result['output']
        if not output:
            output = "(No output)"
        return f"```output\n{output}\n```"
    else:
        error = exec_result['error']
        return f"```output\n{error}\n```"


def generate_with_interactive_code(
    model,
    tokenizer,
    initial_prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_code_executions: int = 5,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    detailed: bool = False,
    share_variables: bool = True
) -> Tuple[str, List[Dict]]:
    """
    Generate response with interactive code execution.
    """
    execution_results = []
    shared_state = {} if share_variables else None
    
    # Check if prompt ends with ```python (expecting code to follow)
    prompt_ends_with_code_start = initial_prompt.rstrip().endswith('```python')
    
    # Current context = prompt + generated so far
    current_context = initial_prompt
    total_generated_tokens = 0
    code_execution_count = 0
    first_generation = True
    
    # Get device
    try:
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if detailed:
        print(f"\n[Interactive Code Generation]")
        print(f"  Max tokens: {max_new_tokens}")
        print(f"  Max code executions: {max_code_executions}")
    
    while total_generated_tokens < max_new_tokens and code_execution_count < max_code_executions:
        remaining_tokens = max_new_tokens - total_generated_tokens
        if remaining_tokens <= 0:
            break
        
        # Tokenize current context
        inputs = tokenizer(
            current_context,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_length = inputs['input_ids'].shape[1]
        
        if detailed:
            print(f"\n[Generation Round {code_execution_count + 1}]")
            print(f"  Context length: {prompt_length} tokens")
        
        # Check if this round starts with code (prompt ends with ```python)
        current_ends_with_code = current_context.rstrip().endswith('```python')
        
        # Create stopping criteria
        stop_criteria = StoppingCriteriaList([
            StopOnCodeBlockEnd(tokenizer, prompt_length, prompt_ends_with_code_start=current_ends_with_code)
        ])
        
        # Generate
        with torch.no_grad():
            if do_sample and temperature > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=remaining_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stop_criteria,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=remaining_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stop_criteria,
                )
        
        # Decode new tokens
        new_tokens = outputs[0][prompt_length:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens_generated = len(new_tokens)
        total_generated_tokens += tokens_generated
        
        if detailed:
            print(f"  Generated: {tokens_generated} tokens")
            preview = new_text[:150] if len(new_text) > 150 else new_text
            print(f"  Preview: {preview}...")
        
        # Update context
        response_so_far = current_context[len(initial_prompt):] + new_text
        
        # Check for complete code block
        check_prompt_code_start = prompt_ends_with_code_start and first_generation
        has_block, block_start, block_end, code = detect_code_block_end(
            response_so_far, 
            prompt_ends_with_code_start=check_prompt_code_start
        )
        first_generation = False
        
        if has_block and code:
            code_execution_count += 1
            
            if detailed:
                print(f"\n  [Code Block #{code_execution_count} Detected]")
                print(f"  Code: {code[:100]}...")
            
            # Execute code
            exec_result = execute_code_safe(code, timeout=10, shared_state=shared_state)
            execution_results.append({
                'code': code,
                'success': exec_result['success'],
                'output': exec_result['output'],
                'error': exec_result.get('error', ''),
                'round': code_execution_count
            })
            
            if detailed:
                if exec_result['success']:
                    print(f"  Result: Success - {exec_result['output'][:50]}...")
                else:
                    print(f"  Result: Error - {exec_result['error'][:50]}...")
            
            # Format output block
            output_block = format_output_block(exec_result)
            
            # Update shared state
            if share_variables and 'state' in exec_result:
                shared_state = exec_result['state']
            
            # Insert output block after code block
            response_with_output = response_so_far[:block_end] + "\n" + output_block + "\n"
            current_context = initial_prompt + response_with_output
            
            if detailed:
                print(f"  Output injected, continuing generation...")
            continue
        else:
            # No code block - update context and check if done
            current_context = initial_prompt + response_so_far
            
            if tokens_generated < remaining_tokens:
                if detailed:
                    print(f"  Generation completed (EOS)")
                break
            continue
    
    # Extract final response
    final_response = current_context[len(initial_prompt):]
    
    # Post-processing: truncate after boxed answer if followed by unrelated content
    boxed_pattern = r'(\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    boxed_matches = list(re.finditer(boxed_pattern, final_response))
    
    if boxed_matches:
        last_boxed = boxed_matches[-1]
        after_boxed = final_response[last_boxed.end():]
        
        # Check for unrelated content
        unrelated_patterns = [
            r'\n\n[A-Z][^.?!]*\?',  # New question
            r'\n\n(?:Given|Let|Consider|Find|Compute|Calculate|Determine|Prove|Show)\s',
            r'\n\n\$\\',  # New LaTeX block
        ]
        
        for pattern in unrelated_patterns:
            match = re.search(pattern, after_boxed)
            if match:
                truncate_pos = last_boxed.end() + match.start()
                final_response = final_response[:truncate_pos].rstrip()
                if detailed:
                    print(f"  [Truncated unrelated content]")
                break
    
    if detailed:
        print(f"\n[Generation Complete]")
        print(f"  Total tokens: {total_generated_tokens}")
        print(f"  Code executions: {code_execution_count}")
    
    return final_response, execution_results


# System prompt for chat template (should match training)
SYSTEM_PROMPT_AGENT = (
    "You are a mathematical reasoning assistant that uses Python code to solve problems. "
    "Write code to help with calculations. When you see an error or unexpected result, "
    "analyze what went wrong, explain your reasoning, and correct the code. "
    "Always provide your final answer in \\boxed{} format."
)


def run_solver_with_interactive_code(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    max_iterations: int = 1,
    max_code_executions: int = 5,
    detailed: bool = False,
    dataset_name: str = "",
    share_variables: bool = True,
    apply_chat_template: bool = False
) -> Dict:
    """Run solver with interactive code execution.
    
    Args:
        apply_chat_template: If True, use chat template format (for fine-tuned models).
                           If False, use plain text format (for base models).
    """
    from utils.prompt_utils import extract_answer, check_answer
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER WITH INTERACTIVE CODE")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Max code executions: {max_code_executions}")
        print(f"Apply chat template: {apply_chat_template}")
        print(f"{'='*80}\n")
    
    iterations = []
    predicted_answer = None
    
    for iteration_num in range(1, max_iterations + 1):
        if detailed:
            print(f"\n[Iteration {iteration_num}/{max_iterations}]")
        
        # Build prompt based on chat template setting
        if apply_chat_template:
            # Use chat template format (for fine-tuned models)
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_AGENT,
                },
                {
                    "role": "user",
                    "content": (
                        question
                        + "\nPlease solve this step by step using Python code. "
                        "Put your final answer within \\boxed{}."
                    ),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Use plain text format (for base models)
            from utils.prompt_utils import format_prompt_standard
            base_prompt = format_prompt_standard(question, dataset_name)
            tool_instruction = "\n\nYou may use Python code to help with calculations. Show your reasoning step by step."
            prompt = base_prompt + tool_instruction
        
        # Generate with interactive code execution
        response, exec_results = generate_with_interactive_code(
            model=model,
            tokenizer=tokenizer,
            initial_prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            max_code_executions=max_code_executions,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            detailed=detailed,
            share_variables=share_variables
        )
        
        # Extract answer - prioritize boxed answer (more reliable in interactive mode)
        # In interactive mode, model may print intermediate values, so boxed is more reliable
        boxed_answer = extract_answer(response)
        code_answer = None
        
        # Try to get answer from code execution outputs (collect all numbers, take last)
        if exec_results:
            all_numbers = []
            for result in exec_results:
                if result['success'] and result['output']:
                    output = result['output'].strip()
                    # Extract all numbers from output
                    numbers = re.findall(r'[-+]?\d*\.?\d+', output)
                    all_numbers.extend(numbers)
            
            if all_numbers:
                code_answer = all_numbers[-1]
                # Clean up: remove trailing .0 for integers
                if code_answer.endswith('.0'):
                    code_answer = code_answer[:-2]
        
        # Decision: prefer boxed answer (more reliable), fallback to code output
        if boxed_answer is not None:
            predicted_answer = boxed_answer
            if detailed:
                print(f"\n[Boxed Answer]: {boxed_answer}")
                if code_answer and code_answer != boxed_answer:
                    print(f"  (Code output: {code_answer}, using boxed instead)")
        elif code_answer is not None:
            predicted_answer = code_answer
            if detailed:
                print(f"\n[Code Output Answer]: {code_answer} (no boxed found)")
        else:
            predicted_answer = None
            if detailed:
                print(f"\n[No Answer Found]")
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "response": response,
            "predicted_answer": predicted_answer,
            "code_executions": len(exec_results),
            "exec_results": exec_results
        }
        iterations.append(iteration_data)
        
        if predicted_answer:
            break
    
    # Check correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    # Determine case type
    total_code_executions = sum(it['code_executions'] for it in iterations)
    
    if final_correct:
        case_type = "SUCCESS_WITH_CODE" if total_code_executions > 0 else "SUCCESS_NO_CODE"
    else:
        case_type = "FAILED_WITH_CODE" if total_code_executions > 0 else "FAILED_NO_CODE"
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "final_correct": final_correct,
        "final_verdict": "CORRECT" if final_correct else "INCORRECT",
        "first_answer": predicted_answer,
        "first_correct": final_correct,
        "case_type": case_type,
        "total_iterations": len(iterations),
        "total_code_executions": total_code_executions,
        "iterations": iterations,
        "config": {
            "max_iterations": max_iterations,
            "max_code_executions": max_code_executions,
            "share_variables": share_variables
        }
    }
