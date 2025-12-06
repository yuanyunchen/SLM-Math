"""
Agent Code as Answer - 直接使用代码执行结果作为答案

工作流程：
1. 模型生成 reasoning + Python 代码
2. 执行代码获取结果
3. 直接使用执行结果作为最终答案（不再让模型生成）

与其他 agent 的区别：
- agent_with_python_tools: 模型生成答案，代码执行只是辅助
- agent_with_code_feedback: 模型看到执行结果后再生成答案
- agent_code_as_answer: 直接用代码执行结果作为答案

适用场景：
- 信任代码计算结果
- 模型容易在最后一步算错
- 需要精确数值计算的问题
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
from models.generation_config import MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY


def run_agent_code_as_answer(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    detailed: bool = False,
    dataset_name: str = "",
    enable_tools: bool = True,
    greedy: bool = True,
    apply_chat_template: bool = False
) -> Dict:
    """
    Run agent that uses code execution result as the final answer
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        model: Language model
        tokenizer: Tokenizer
        detailed: Verbose output
        dataset_name: Dataset name
        enable_tools: Enable Python code execution
        greedy: Use greedy decoding
        apply_chat_template: Whether to use chat template format
    
    Returns:
        Dict with results
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response
    from utils.python_code_execution import extract_python_code_blocks, execute_python_code
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"AGENT CODE AS ANSWER")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Python Tools: {'Enabled' if enable_tools else 'Disabled'}")
        print(f"Greedy: {greedy}")
        print(f"{'='*80}\n")
    
    # Build prompt (avoid triple backticks in instruction which confuse some models)
    tool_instruction = ""
    if enable_tools:
        tool_instruction = "\n\nYou may use Python code to help with calculations. Show your reasoning step by step."
    
    prompt = format_prompt_standard(question, dataset_name) + tool_instruction
    
    if detailed:
        print(f"[Step 1: Generating reasoning + code...]")
    
    # Generate response
    if hasattr(model, 'generate_single'):
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
        response = generate_response(
            model,
            tokenizer,
            prompt,
            "standard",
            detailed,
            greedy=greedy
        )
    
    if detailed:
        print(f"\n[Response Preview]:\n{response[:300]}...")
    
    # Extract model's answer (for comparison)
    model_answer = extract_answer(response)
    has_code = "```python" in response
    
    # Execute code if present
    exec_results = []
    code_answer = None
    
    if enable_tools and has_code:
        if detailed:
            print(f"\n[Step 2: Executing code...]")
        
        code_blocks = extract_python_code_blocks(response)
        
        for i, code in enumerate(code_blocks, 1):
            result = execute_python_code(code, timeout=10)
            exec_results.append(result)
            
            if result['success']:
                output = result['output'].strip()
                if detailed:
                    print(f"  Block {i}: Success - {output[:50]}")
                
                # Extract numeric result from output (last number)
                numbers = re.findall(r'[-+]?\d*\.?\d+', output)
                if numbers:
                    code_answer = numbers[-1]
                    # Clean up: remove trailing .0 for integers
                    if code_answer.endswith('.0'):
                        code_answer = code_answer[:-2]
            else:
                if detailed:
                    print(f"  Block {i}: Error - {result['error'][:50]}")
    
    # Determine final answer: prefer code result, fallback to model answer
    if code_answer is not None:
        final_answer = code_answer
        answer_source = "code_execution"
    else:
        final_answer = model_answer
        answer_source = "model_output"
    
    if detailed:
        print(f"\n[Answer Source]: {answer_source}")
        print(f"[Model Answer]: {model_answer}")
        print(f"[Code Answer]: {code_answer}")
        print(f"[Final Answer]: {final_answer}")
    
    # Check correctness
    final_correct = check_answer(final_answer, ground_truth) if final_answer else False
    model_correct = check_answer(model_answer, ground_truth) if model_answer else False
    
    # Determine case type
    if model_correct and final_correct:
        case_type = "BOTH_CORRECT"
    elif not model_correct and final_correct:
        case_type = "CODE_SAVED"  # Code execution saved the answer
    elif model_correct and not final_correct:
        case_type = "CODE_WRONG"  # Code was wrong, model was right
    else:
        case_type = "BOTH_WRONG"
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"FINAL RESULT")
        print(f"{'='*80}")
        print(f"Final Answer: {final_answer} (from {answer_source})")
        print(f"Model Answer: {model_answer} (correct: {model_correct})")
        print(f"Ground Truth: {ground_truth}")
        print(f"Final Correct: {final_correct}")
        print(f"Case Type: {case_type}")
        print(f"{'='*80}\n")
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": final_answer,
        "final_correct": final_correct,
        "final_verdict": "CORRECT" if final_correct else "INCORRECT",
        "first_answer": model_answer,  # Model's original answer
        "first_correct": model_correct,
        "response": response,
        "case_type": case_type,
        "code_executed": len(exec_results) > 0,
        "exec_results": exec_results,
        "num_code_blocks": len(exec_results),
        "code_answer": code_answer,
        "model_answer": model_answer,
        "answer_source": answer_source,
        "tools_config": {
            "enable_tools": enable_tools,
            "greedy": greedy,
            "use_chat_template": apply_chat_template
        }
    }


# For testing
if __name__ == "__main__":
    print("Agent Code as Answer - Quick Test")
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
        
        print(f"\nRunning Agent Code as Answer...")
        result = run_agent_code_as_answer(
            question=test_question,
            ground_truth=test_ground_truth,
            model=model,
            tokenizer=tokenizer,
            detailed=True,
            dataset_name="gsm8k",
            enable_tools=True,
            greedy=True
        )
        
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print(f"Final Answer: {result['predicted_answer']} (from {result['answer_source']})")
        print(f"Model Answer: {result['model_answer']}")
        print(f"Correct: {result['final_correct']}")
        print(f"Case Type: {result['case_type']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()














