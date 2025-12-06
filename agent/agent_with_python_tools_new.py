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
- Baseline (无工具): ~69%
- With Python Tools: ~75-80% (代码执行消除计算错误)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
from models.generation_config import MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY


def apply_chat_template_if_enabled(prompt: str, tokenizer, apply_chat_template: bool) -> str:
    """Wrap prompt with chat template if enabled."""
    if not apply_chat_template:
        return prompt
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def run_agent_with_python_tools(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    detailed: bool = False,
    dataset_name: str = "",
    enable_tools: bool = True,
    apply_chat_template: bool = False
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
        apply_chat_template: Whether to apply chat template to prompts
    
    Returns:
        Dict with results
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response
    from utils.python_code_execution import process_text_with_code_execution
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"AGENT WITH PYTHON TOOLS")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Python Tools: {'✓' if enable_tools else '✗'}")
        print(f"{'='*80}\n")
    
    # Build prompt with tool instruction
    prompt = format_prompt_standard(question, dataset_name)
    
    if enable_tools:
        prompt += "\n\nYou can write Python code in ```python``` blocks to help with calculations. The code will be executed automatically and you'll see the results. Use code for any arithmetic operations to ensure accuracy."
    
    # Apply chat template if enabled
    prompt = apply_chat_template_if_enabled(prompt, tokenizer, apply_chat_template)
    
    if detailed:
        print(f"[Generating response...]")
    
    # Generate response - check if model is an inference engine or raw model
    if hasattr(model, 'generate_single'):
        # Using inference engine (from load_inference_engine_wrapper)
        response = model.generate_single(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            detailed=detailed
        )
    else:
        # Using standard model (from load_model)
        response = generate_response(
            model,
            tokenizer,
            prompt,
            "standard",
            detailed
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
                            print(f"  Block {i}: ✓ Output: {result['output'][:50]}")
                        else:
                            print(f"  Block {i}: ✗ Error: {result['error'][:50]}")
                
                # Use response with execution results
                response = response_with_output
            else:
                if detailed:
                    print(f"  No code blocks found or execution failed")
    
    # Extract final answer
    predicted_answer = extract_answer(response)
    
    if detailed:
        print(f"\n[Final Answer]: {predicted_answer}")
    
    # Check correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    # Determine case type (simpler for single-shot)
    case_type = "FIRST_TRY_SUCCESS" if final_correct else "FAILED"
    final_verdict = "CORRECT" if final_correct else "INCORRECT"
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "final_correct": final_correct,
        "final_verdict": final_verdict,
        "first_answer": predicted_answer,  # Same as final for single-shot
        "first_correct": final_correct,
        "response": response,
        "case_type": case_type,
        "code_executed": code_executed,
        "exec_results": exec_results,
        "num_code_blocks": len(exec_results),
        "tools_config": {
            "enable_tools": enable_tools,
            "code_executed": code_executed
        }
    }
