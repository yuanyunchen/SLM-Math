"""
Agent with Python Tools - 单次推理 + 代码执行
带Python工具的单次推理Agent

不同于Base Direct:
- Base Direct: 单次生成，无工具
- Agent with Python Tools: 单次生成 + 自动代码执行 + 结果注入

工作流程：
1. Agent生成reasoning + Python代码
2. 自动提取并执行```python```代码块
3. 将执行结果以```output```形式注入回文本
4. 提取最终答案

适用场景：
- 需要精确计算但不需要迭代的问题
- 小模型容易算错但能写对代码的场景
- 比Base Direct准确，比Solver-Checker简单

预期效果：
- Baseline (无工具): ~65%
- With Python Tools: ~70-75% (代码执行消除计算错误)
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict


# Prompt to encourage code usage - compatible with Qwen2.5-Math's default system prompt
# Note: Qwen2.5-Math already has "Please reason step by step, and put your final answer within \boxed{}"
# as its default system prompt, so we just add the code instruction
TOOL_INSTRUCTION = """

You MUST solve this problem by writing Python code. Follow these steps:
1. Analyze the problem carefully
2. Write Python code in ```python``` blocks to calculate the answer
3. The code will be automatically executed and results shown
4. Put your final numerical answer in \\boxed{}

IMPORTANT: Always use Python code for calculations. Do NOT skip the code step."""


def extract_answer_from_code_output(exec_results: list, detailed: bool = False) -> str:
    """
    Extract numerical answer from code execution results.
    
    Strategy:
    1. For tuple outputs like (13, 98), take the FIRST element (usually the answer)
    2. For simple number outputs, use directly
    3. For multiple code blocks, prefer the last one that produces a clear answer
    """
    if not exec_results:
        return None
    
    # Try to extract from last successful execution
    for result in reversed(exec_results):
        if result['success'] and result['output']:
            output_text = result['output'].strip()
            
            # Handle multiple lines - take the last line with a number
            lines = output_text.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check for tuple format: (x, y) or (x, y, z) - take FIRST element
                tuple_match = re.match(r'^\(([+-]?\d+\.?\d*),\s*[+-]?\d+\.?\d*(?:,\s*[+-]?\d+\.?\d*)*\)$', line)
                if tuple_match:
                    first_elem = tuple_match.group(1)
                    if detailed:
                        print(f"  [Code output (tuple first elem)]: {first_elem}")
                    return first_elem
                    
                # Try to parse as a single number
                try:
                    # Handle scientific notation and decimals
                    num = float(line)
                    if detailed:
                        print(f"  [Code output]: {num}")
                    return str(num)
                except ValueError:
                    pass
                
                # Extract numbers from the line - take the FIRST clear number
                # (often printed answers are at the start)
                numbers = re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', line)
                if numbers:
                    # If it looks like a simple printout, take the first number
                    if detailed:
                        print(f"  [Code output extracted]: {numbers[0]}")
                    return numbers[0]
    
    return None


def run_agent_with_python_tools(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    detailed: bool = False,
    dataset_name: str = "",
    enable_tools: bool = True
) -> Dict:
    """
    Run single-shot agent with Python code execution
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        model: Language model
        tokenizer: Tokenizer
        detailed: Verbose output
        dataset_name: Dataset name
        enable_tools: Enable Python code execution
    
    Returns:
        Dict with results
    """
    from utils.prompt_utils import extract_answer, check_answer
    from models.inference import generate_response
    from utils.python_code_execution import process_text_with_code_execution
    from agent.unified_config import FIRST_ROUND_SOLVER_CONFIG
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"AGENT WITH PYTHON TOOLS")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Python Tools: {'enabled' if enable_tools else 'disabled'}")
        print(f"{'='*80}\n")
    
    # Build prompt - Qwen2.5-Math already has "reason step by step" in system prompt
    # Just add the question, and optionally the code instruction
    if enable_tools:
        prompt = f"{question}{TOOL_INSTRUCTION}"
    else:
        prompt = question
    
    if detailed:
        print(f"[Generating response with Unified Config...]")
    
    # Generate response with unified config (single-shot agent, use first round config)
    response = generate_response(
        model,
        tokenizer,
        prompt,
        "standard",
        detailed,
        temperature=FIRST_ROUND_SOLVER_CONFIG['temperature'],
        do_sample=FIRST_ROUND_SOLVER_CONFIG['do_sample'],
        top_p=FIRST_ROUND_SOLVER_CONFIG['top_p']
    )
    
    # Execute code if tools enabled
    code_executed = False
    exec_results = []
    
    if enable_tools:
        # Check if response contains code
        if "```python" in response:
            if detailed:
                print(f"\n[Executing Python code...]")
            
            # Execute code and get response with outputs
            response_with_output, exec_results = process_text_with_code_execution(
                response,
                share_variables=True
            )
            
            if exec_results:
                code_executed = True
                if detailed:
                    print(f"\n[Code Execution Results]")
                    for i, result in enumerate(exec_results, 1):
                        if result['success']:
                            output_preview = result['output'][:100] if result['output'] else "(no output)"
                            print(f"  Block {i}: SUCCESS -> {output_preview}")
                        else:
                            print(f"  Block {i}: ERROR -> {result['error'][:80]}")
                
                # Use response with execution results
                response = response_with_output
            else:
                if detailed:
                    print(f"  No code blocks found or execution failed")
        else:
            if detailed:
                print(f"\n[WARNING] Model did not generate Python code!")
    
    # Extract final answer with priority:
    # 1. Code output (most reliable for calculations)
    # 2. Boxed answer from response
    predicted_answer = None
    answer_source = "none"
    
    if code_executed and exec_results:
        # Priority 1: Extract from code output
        predicted_answer = extract_answer_from_code_output(exec_results, detailed)
        if predicted_answer:
            answer_source = "code_output"
            if detailed:
                print(f"  [Answer from code output]: {predicted_answer}")
    
    # Fallback to boxed answer if no answer from code
    if not predicted_answer:
        predicted_answer = extract_answer(response)
        if predicted_answer:
            answer_source = "boxed"
            if detailed:
                print(f"  [Answer from boxed]: {predicted_answer}")
    
    if detailed:
        print(f"\n[Final Answer]: {predicted_answer} (source: {answer_source})")
    
    # Check correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    # Determine case type (simpler for single-shot)
    case_type = "FIRST_TRY_SUCCESS" if final_correct else "FAILED"
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "final_correct": final_correct,
        "first_answer": predicted_answer,  # Same as final for single-shot
        "first_correct": final_correct,
        "final_verdict": "N/A",  # No checker for single-shot agent
        "response": response,
        "case_type": case_type,
        "code_executed": code_executed,
        "exec_results": exec_results,
        "num_code_blocks": len(exec_results),
        "answer_source": answer_source,
        "tools_config": {
            "enable_tools": enable_tools,
            "code_executed": code_executed
        }
    }


# For testing
if __name__ == "__main__":
    print("Agent with Python Tools - Quick Test")
    print("=" * 80)
    
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_ground_truth = "18"
    
    try:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        from models.inference import load_model
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning Agent with Python Tools...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_agent_with_python_tools(
            question=test_question,
            ground_truth=test_ground_truth,
            model=model,
            tokenizer=tokenizer,
            detailed=True,
            dataset_name="gsm8k",
            enable_tools=True
        )
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Expected: {result['ground_truth']}")
        print(f"Correct: {'✅' if result['final_correct'] else '❌'}")
        print(f"Case Type: {result['case_type']}")
        print(f"Code Executed: {result['code_executed']}")
        print(f"Number of Code Blocks: {result['num_code_blocks']}")
        if result['code_executed']:
            print(f"\nCode Execution Summary:")
            for i, exec_result in enumerate(result['exec_results'], 1):
                status = '✓' if exec_result['success'] else '✗'
                print(f"  Block {i}: {status}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

