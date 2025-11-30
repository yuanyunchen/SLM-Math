"""
Python Code Execution Utility
从模型输出中提取Python代码，执行并返回结果

用于帮助小模型完成精确计算：
1. 模型生成 ```python ... ``` 代码块
2. 自动执行代码
3. 返回 ```output ... ``` 格式的结果
4. 模型基于准确结果继续推理
"""

import re
import sys
import io
import traceback
from typing import List, Dict, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
import signal
from functools import wraps


class TimeoutException(Exception):
    """Timeout exception for code execution"""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutException("Code execution timeout")


def extract_python_code_blocks(text: str) -> List[str]:
    """
    从文本中提取所有Python代码块
    
    Args:
        text: 包含代码块的文本
    
    Returns:
        List of code strings
    """
    # Pattern to match ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Also try to match code blocks without language tag
    if not matches:
        pattern = r'```\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
    
    return [m.strip() for m in matches if m.strip()]


def execute_python_code(
    code: str, 
    timeout: int = 5,
    max_output_length: int = 1000
) -> Dict[str, any]:
    """
    安全地执行Python代码并捕获输出
    
    Args:
        code: Python代码字符串
        timeout: 执行超时时间（秒）
        max_output_length: 最大输出长度
    
    Returns:
        Dictionary with:
            - success: bool
            - output: str (stdout)
            - error: str (stderr or exception)
            - result: any (last expression result if available)
    """
    result_dict = {
        'success': False,
        'output': '',
        'error': '',
        'result': None
    }
    
    # Create string buffers to capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Set timeout (Unix only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    
    try:
        # Prepare execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'print': print,
        }
        exec_locals = {}
        
        # Redirect stdout and stderr
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Execute the code
            exec(code, exec_globals, exec_locals)
        
        # Get output
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        # Truncate if too long
        if len(stdout_output) > max_output_length:
            stdout_output = stdout_output[:max_output_length] + "\n... (output truncated)"
        
        result_dict['success'] = True
        result_dict['output'] = stdout_output.strip()
        if stderr_output:
            result_dict['error'] = stderr_output.strip()
        
        # Try to get the last expression value
        # (not always possible, but useful for simple cases)
        
    except TimeoutException:
        result_dict['error'] = f"Execution timeout ({timeout}s)"
    except Exception as e:
        result_dict['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Cancel alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
    
    return result_dict


def format_execution_result(exec_result: Dict) -> str:
    """
    将执行结果格式化为模型可读的输出
    
    Args:
        exec_result: execute_python_code返回的结果字典
    
    Returns:
        Formatted string like:
        ```output
        <result>
        ```
    """
    if exec_result['success']:
        output = exec_result['output']
        if not output:
            output = "(no output)"
        return f"```output\n{output}\n```"
    else:
        error = exec_result['error']
        return f"```output\n{error}\n```"


def process_text_with_code_execution(
    text: str,
    execute_code: bool = True,
    timeout: int = 5,
    share_variables: bool = True
) -> Tuple[str, List[Dict]]:
    """
    处理包含代码块的文本，执行代码并插入结果
    
    Args:
        text: 包含```python```代码块的文本
        execute_code: 是否实际执行代码
        timeout: 代码执行超时时间
        share_variables: 多个代码块之间是否共享变量（默认True）
    
    Returns:
        Tuple of:
            - processed_text: 插入了执行结果的文本
            - execution_results: 所有执行结果的列表
    """
    if not execute_code:
        return text, []
    
    # Find all code blocks with their positions
    pattern = r'(```python\s*\n.*?```)'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if not matches:
        return text, []
    
    # Shared execution environment if requested
    shared_globals = None
    shared_locals = None
    if share_variables:
        shared_globals = {
            '__builtins__': __builtins__,
            'print': print,
        }
        shared_locals = {}
    
    # Process from end to start to maintain correct positions
    execution_results = []
    processed_text = text
    
    # Need to process in order for shared variables, but insert in reverse order
    # So we first execute all in order, then insert results in reverse
    temp_results = []
    
    for match in matches:
        code_block = match.group(1)
        
        # Extract code
        code = extract_python_code_blocks(code_block)
        if not code:
            temp_results.append(None)
            continue
        
        code_str = code[0]
        
        # Execute code (with or without shared state)
        if share_variables and shared_globals is not None:
            exec_result = execute_python_code_with_state(
                code_str, 
                shared_globals, 
                shared_locals, 
                timeout=timeout
            )
        else:
            exec_result = execute_python_code(code_str, timeout=timeout)
        
        temp_results.append(exec_result)
    
    # Now insert results in reverse order
    for match, exec_result in zip(reversed(matches), reversed(temp_results)):
        if exec_result is None:
            continue
            
        end_pos = match.end()
        
        execution_results.insert(0, exec_result)
        
        # Format result
        result_str = format_execution_result(exec_result)
        
        # Check if there's already an output block right after the code block
        # Look for ```output...``` pattern immediately following
        remaining_text = processed_text[end_pos:]
        output_pattern = r'^\s*```output\s*\n(.*?)```'
        output_match = re.match(output_pattern, remaining_text, re.DOTALL)
        
        if output_match:
            # Found existing output block - replace it with real execution result
            output_end = end_pos + output_match.end()
            processed_text = (
                processed_text[:end_pos] +
                '\n' + result_str +
                processed_text[output_end:]
            )
        else:
            # No existing output block - insert result after code block
            processed_text = (
                processed_text[:end_pos] + 
                '\n' + result_str + 
                processed_text[end_pos:]
            )
    
    return processed_text, execution_results


def execute_python_code_with_state(
    code: str,
    exec_globals: dict,
    exec_locals: dict,
    timeout: int = 5,
    max_output_length: int = 1000
) -> Dict[str, any]:
    """
    在指定的全局/局部环境中执行代码（支持状态持久化）
    
    Args:
        code: Python代码字符串
        exec_globals: 全局变量字典
        exec_locals: 局部变量字典
        timeout: 执行超时时间（秒）
        max_output_length: 最大输出长度
    
    Returns:
        Same as execute_python_code
    """
    result_dict = {
        'success': False,
        'output': '',
        'error': '',
        'result': None
    }
    
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, exec_globals, exec_locals)
        
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        if len(stdout_output) > max_output_length:
            stdout_output = stdout_output[:max_output_length] + "\n... (output truncated)"
        
        result_dict['success'] = True
        result_dict['output'] = stdout_output.strip()
        if stderr_output:
            result_dict['error'] = stderr_output.strip()
        
    except TimeoutException:
        result_dict['error'] = f"Execution timeout ({timeout}s)"
    except Exception as e:
        result_dict['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
    
    return result_dict


def interactive_code_execution_demo():
    """交互式演示"""
    print("="*80)
    print("Python Code Execution Tool - Interactive Demo")
    print("="*80)
    
    sample_text = """
Let's calculate how much money Janet makes:

```python
# Define the given values
total_eggs_per_day = 16
eggs_eaten_for_breakfast = 3
muffin_baking_eggs = 4

# Calculate the number of fresh eggs available for sale
fresh_eggs_available = total_eggs_per_day - (eggs_eaten_for_breakfast + muffin_baking_eggs)

# The price per fresh egg is $2
price_per_egg = 2

# Calculate the total revenue from selling the fresh eggs
revenue_from_fresh_eggs = fresh_eggs_available * price_per_egg

print(revenue_from_fresh_eggs)
```

So Janet makes some money each day.
"""
    
    print("\n" + "="*80)
    print("ORIGINAL TEXT:")
    print("="*80)
    print(sample_text)
    
    print("\n" + "="*80)
    print("PROCESSING...")
    print("="*80)
    
    # Process text
    processed_text, results = process_text_with_code_execution(sample_text)
    
    print("\n" + "="*80)
    print("PROCESSED TEXT (with execution results):")
    print("="*80)
    print(processed_text)
    
    print("\n" + "="*80)
    print("EXECUTION RESULTS SUMMARY:")
    print("="*80)
    for i, result in enumerate(results, 1):
        print(f"\nCode Block {i}:")
        print(f"  Success: {result['success']}")
        print(f"  Output: {result['output'][:100]}")
        if result['error']:
            print(f"  Error: {result['error'][:100]}")


# Test with error handling
def test_code_execution():
    """测试各种情况"""
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    # Test 1: Normal execution
    print("\n[Test 1] Normal execution:")
    code1 = "x = 10\ny = 20\nprint(x + y)"
    result1 = execute_python_code(code1)
    print(f"Success: {result1['success']}, Output: {result1['output']}")
    
    # Test 2: Error
    print("\n[Test 2] Syntax error:")
    code2 = "x = 10\nprint(x +"
    result2 = execute_python_code(code2)
    print(f"Success: {result2['success']}, Error: {result2['error'][:50]}...")
    
    # Test 3: Import and use libraries
    print("\n[Test 3] Using math library:")
    code3 = "import math\nprint(math.sqrt(16))"
    result3 = execute_python_code(code3)
    print(f"Success: {result3['success']}, Output: {result3['output']}")
    
    # Test 4: No output
    print("\n[Test 4] No output:")
    code4 = "x = 10\ny = 20"
    result4 = execute_python_code(code4)
    print(f"Success: {result4['success']}, Output: '{result4['output']}'")
    
    # Test 5: Multiple prints
    print("\n[Test 5] Multiple prints:")
    code5 = "for i in range(3):\n    print(f'Number {i}')"
    result5 = execute_python_code(code5)
    print(f"Success: {result5['success']}, Output:\n{result5['output']}")


if __name__ == "__main__":
    # Run demo
    interactive_code_execution_demo()
    
    # Run tests
    test_code_execution()
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)

