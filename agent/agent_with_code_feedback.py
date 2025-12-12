"""
Agent with Code Execution Feedback
Agent that uses code execution feedback.

Workflow:
1. The model generates reasoning + Python code.
2. Execute the code automatically and inject the results.
3. The model sees the execution results and, based on them, provides the \boxed{} answer.

Difference from single-pass inference:
- Single-pass: model generates reasoning + code, then we extract the answer directly after execution (model does not see the execution result).
- Code-feedback: model first generates code → sees the execution result → then generates the \boxed{} answer.

When to use:
- The model can write correct code but may not know how to use the execution result.
- Problems that require further reasoning based on computed results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
from models.generation_config import MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY


def run_agent_with_code_feedback(
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
    Run agent with code execution feedback
    
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
        print(f"AGENT WITH CODE EXECUTION FEEDBACK")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Python Tools: {'Enabled' if enable_tools else 'Disabled'}")
        print(f"Greedy: {greedy}")
        print(f"Chat Template: {apply_chat_template}")
        print(f"{'='*80}\n")
    
    # Build initial prompt (avoid triple backticks in instruction which confuse some models)
    tool_instruction = ""
    if enable_tools:
        tool_instruction = "\n\nYou may use Python code to help with calculations. Show your reasoning step by step."
    
    # Build prompt (standard mode works for both base and chat models)
    prompt = format_prompt_standard(question, dataset_name) + tool_instruction
    
    if detailed:
        print(f"[Step 1: Generating reasoning + code...]")
    
    # First generation: reasoning + code
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
        print(f"\n[Initial Response]:\n{response[:300]}...")
    
    # Check if already has \boxed{} answer (no code needed)
    initial_answer = extract_answer(response)
    has_code = "```python" in response
    
    if initial_answer and not has_code:
        # Model gave answer directly without code
        if detailed:
            print(f"\n[Model gave answer directly: {initial_answer}]")
        
        final_correct = check_answer(initial_answer, ground_truth)
        
        return {
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": initial_answer,
            "final_correct": final_correct,
            "final_verdict": "CORRECT" if final_correct else "INCORRECT",
            "first_answer": initial_answer,
            "first_correct": final_correct,
            "response": response,
            "case_type": "FIRST_TRY_SUCCESS" if final_correct else "FAILED",
            "code_executed": False,
            "exec_results": [],
            "num_code_blocks": 0,
            "used_feedback": False,
            "tools_config": {
                "enable_tools": enable_tools,
                "greedy": greedy,
                "use_chat_template": apply_chat_template
            }
        }
    
    # Execute code if present
    exec_results = []
    code_output = ""
    
    if enable_tools and has_code:
        if detailed:
            print(f"\n[Step 2: Executing code...]")
        
        # Extract and execute code blocks
        code_blocks = extract_python_code_blocks(response)
        
        output_parts = []
        for i, code in enumerate(code_blocks, 1):
            result = execute_python_code(code, timeout=10)
            exec_results.append(result)
            
            if result['success']:
                if result['output'].strip():
                    output_parts.append(f"Code block {i} output:\n{result['output']}")
                else:
                    output_parts.append(f"Code block {i}: Executed successfully (no output)")
                if detailed:
                    output_preview = result['output'][:100] if result['output'] else "(no output)"
                    print(f"  Block {i}: Success - {output_preview}")
            else:
                output_parts.append(f"Code block {i} error:\n{result['error']}")
                if detailed:
                    print(f"  Block {i}: Error - {result['error'][:100]}")
        
        code_output = "\n".join(output_parts)
    
    # If no code or code failed, return initial response
    if not exec_results or not code_output:
        if detailed:
            print(f"\n[No code executed, using initial response]")
        
        final_correct = check_answer(initial_answer, ground_truth) if initial_answer else False
        
        return {
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": initial_answer,
            "final_correct": final_correct,
            "final_verdict": "CORRECT" if final_correct else "INCORRECT",
            "first_answer": initial_answer,
            "first_correct": final_correct,
            "response": response,
            "case_type": "FIRST_TRY_SUCCESS" if final_correct else "FAILED",
            "code_executed": len(exec_results) > 0,
            "exec_results": exec_results,
            "num_code_blocks": len(exec_results),
            "used_feedback": False,
            "tools_config": {
                "enable_tools": enable_tools,
                "greedy": greedy,
                "use_chat_template": apply_chat_template
            }
        }
    
    # Build feedback prompt with code execution results
    if detailed:
        print(f"\n[Step 3: Generating final answer based on execution results...]")
    
    # Build feedback prompt: original prompt + model's response + code output
    # Model can see its own reasoning and code, plus the execution result
    feedback_prompt = f"{prompt}\n\n{response}\n\n```output\n{code_output}\n```\n\nBased on the execution result above, give your final answer in \\boxed{{}}."
    
    # Second generation: final answer based on execution results
    if hasattr(model, 'generate_single'):
        final_response = model.generate_single(
            feedback_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            detailed=detailed
        )
    else:
        final_response = generate_response(
            model,
            tokenizer,
            feedback_prompt,
            "standard",
            detailed,
            greedy=greedy
        )
    
    if detailed:
        print(f"\n[Final Response]:\n{final_response[:300]}...")
    
    # Extract final answer from model's response
    final_answer = extract_answer(final_response)
    
    # If extraction failed, try extracting from code output
    if not final_answer:
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', code_output)
        if numbers:
            final_answer = numbers[-1]
            if detailed:
                print(f"\n[Using code output as answer: {final_answer}]")
        else:
            if detailed:
                print(f"\n[No answer found]")
    else:
        if detailed:
            print(f"\n[Final Answer]: {final_answer}")
    
    # Check correctness
    final_correct = check_answer(final_answer, ground_truth) if final_answer else False
    first_correct = check_answer(initial_answer, ground_truth) if initial_answer else False
    
    # Determine case type
    if first_correct and final_correct:
        case_type = "FIRST_TRY_SUCCESS"
    elif not first_correct and final_correct:
        case_type = "IMPROVED"
    elif first_correct and not final_correct:
        case_type = "DEGRADED"
    else:
        case_type = "FAILED"
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"FINAL RESULT")
        print(f"{'='*80}")
        print(f"Predicted Answer: {final_answer}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correct: {final_correct}")
        print(f"Code Blocks: {len(exec_results)}")
        print(f"Used Feedback: True")
        print(f"Case Type: {case_type}")
        print(f"{'='*80}\n")
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": final_answer,
        "final_correct": final_correct,
        "final_verdict": "CORRECT" if final_correct else "INCORRECT",
        "first_answer": initial_answer,
        "first_correct": first_correct,
        "response": response + f"\n\n```output\n{code_output}\n```\n\n" + final_response,
        "case_type": case_type,
        "code_executed": True,
        "exec_results": exec_results,
        "num_code_blocks": len(exec_results),
        "used_feedback": True,
        "tools_config": {
            "enable_tools": enable_tools,
            "greedy": greedy,
            "use_chat_template": apply_chat_template
        }
    }

