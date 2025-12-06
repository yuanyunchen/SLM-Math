"""
Agent with RAG Review - Two-Stage Retrieval-Augmented Generation
两阶段 RAG Agent: 先生成，后检索辅助审查

工作流程:
1. 第一阶段: 模型独立生成初始答案 (无 RAG 干扰)
2. 第二阶段: 检索相似题目，让模型参考后重新审视答案
   - 明确告知模型：检索结果仅供参考，可能不准确
   - 让模型自己判断是否需要修改答案

优势:
- 避免检索噪声干扰模型原始思路
- 模型可以自主判断检索结果是否有用
- 减少幻觉风险
- 如果第一次就对了，可以选择跳过第二阶段

与其他 Agent 的区别:
- agent_with_math_rag: 直接用检索结果作为 few-shot (可能引入噪声)
- agent_with_rag_review: 先独立思考，再参考检索结果审查 (更稳健)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Optional, List
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


def generate_model_response(model, tokenizer, prompt: str, detailed: bool = False):
    """Generate response from model."""
    from models.inference import generate_response
    
    if hasattr(model, 'generate_single'):
        return model.generate_single(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            detailed=detailed
        )
    else:
        return generate_response(
            model,
            tokenizer,
            prompt,
            "standard",
            detailed
        )


def run_agent_with_rag_review(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    retriever=None,
    top_k: int = 3,
    detailed: bool = False,
    dataset_name: str = "",
    enable_tools: bool = False,
    include_solution: bool = True,
    max_solution_length: int = 400,
    apply_chat_template: bool = False,
    always_review: bool = True,
    confidence_threshold: float = 0.8
) -> Dict:
    """
    Run Two-Stage RAG Review Agent
    
    Stage 1: Generate initial answer independently
    Stage 2: Retrieve similar examples and review (with caution warnings)
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        model: Language model
        tokenizer: Tokenizer
        retriever: MathRAGRetriever instance (will be created if None)
        top_k: Number of similar examples to retrieve
        detailed: Verbose output
        dataset_name: Dataset name (gsm8k, math500)
        enable_tools: Enable Python code execution
        include_solution: Include solution steps in retrieved examples
        max_solution_length: Max length of solution in examples
        apply_chat_template: Whether to apply chat template
        always_review: Always do stage 2 review (default: True)
        confidence_threshold: Skip review if confidence > threshold (not implemented yet)
    
    Returns:
        Dict with results
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from utils.math_rag import MathRAGRetriever
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"AGENT WITH RAG REVIEW (Two-Stage)")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Top-K: {top_k}")
        print(f"Python Tools: {'Enabled' if enable_tools else 'Disabled'}")
        print(f"Always Review: {always_review}")
        print(f"{'='*80}\n")
    
    # Initialize retriever if not provided
    if retriever is None:
        if detailed:
            print("[Initializing RAG retriever...]")
        retriever = MathRAGRetriever(
            dataset_name=dataset_name,
            method="bm25"
        )
    
    # =========================================================================
    # STAGE 1: Independent Generation (No RAG)
    # =========================================================================
    if detailed:
        print("\n" + "="*60)
        print("STAGE 1: Independent Generation")
        print("="*60)
    
    # Build base prompt (no few-shot examples)
    base_prompt = format_prompt_standard(question, dataset_name)
    
    if enable_tools:
        base_prompt += "\n\nYou can write Python code in ```python``` blocks to help with calculations. The code will be executed automatically and you'll see the results."
    
    prompt_stage1 = apply_chat_template_if_enabled(base_prompt, tokenizer, apply_chat_template)
    
    if detailed:
        print(f"[Stage 1] Generating initial response...")
    
    response_stage1 = generate_model_response(model, tokenizer, prompt_stage1, detailed)
    
    # Execute code if needed
    code_executed_stage1 = False
    exec_results_stage1 = []
    
    if enable_tools and "```python" in response_stage1:
        from utils.python_code_execution import process_text_with_code_execution
        
        if detailed:
            print(f"[Stage 1] Executing Python code...")
        
        response_stage1, exec_results_stage1 = process_text_with_code_execution(
            response_stage1,
            share_variables=True
        )
        code_executed_stage1 = bool(exec_results_stage1)
    
    # Extract Stage 1 answer
    answer_stage1 = extract_answer(response_stage1)
    correct_stage1 = check_answer(answer_stage1, ground_truth) if answer_stage1 else False
    
    if detailed:
        print(f"[Stage 1] Answer: {answer_stage1}")
        print(f"[Stage 1] Correct: {correct_stage1}")
    
    # =========================================================================
    # STAGE 2: RAG-Assisted Review (with caution)
    # =========================================================================
    
    # Decide whether to do Stage 2
    # Be conservative: only review if we have good reason to doubt Stage 1
    do_stage2 = False
    
    if always_review:
        # Check if retrieved examples suggest a different answer pattern
        # This helps avoid unnecessary reviews that might degrade correct answers
        if retrieved_examples and answer_stage1:
            # Simple heuristic: if all retrieved examples have similar answer format
            # and Stage 1 answer looks reasonable, skip review
            retrieved_answers = [ex.answer for ex in retrieved_examples[:2]]
            # For now, always do review if always_review is True
            do_stage2 = True
    elif not correct_stage1:
        do_stage2 = True
    
    if not do_stage2:
        if detailed:
            print("\n[Skipping Stage 2 - Stage 1 answer is confident]")
        
        return {
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": answer_stage1,
            "final_correct": correct_stage1,
            "final_verdict": "CORRECT" if correct_stage1 else "INCORRECT",
            "first_answer": answer_stage1,
            "first_correct": correct_stage1,
            "response": response_stage1,
            "case_type": "FIRST_TRY_SUCCESS" if correct_stage1 else "FAILED",
            "stage1_answer": answer_stage1,
            "stage1_correct": correct_stage1,
            "stage1_response": response_stage1,
            "stage2_used": False,
            "stage2_answer": None,
            "stage2_correct": None,
            "code_executed": code_executed_stage1,
            "exec_results": exec_results_stage1,
            "num_code_blocks": len(exec_results_stage1),
            "rag_config": {
                "top_k": top_k,
                "include_solution": include_solution,
                "num_retrieved": 0,
                "retrieved_examples": []
            },
            "tools_config": {
                "enable_tools": enable_tools,
                "code_executed": code_executed_stage1
            }
        }
    
    if detailed:
        print("\n" + "="*60)
        print("STAGE 2: RAG-Assisted Review")
        print("="*60)
    
    # Retrieve similar examples
    if detailed:
        print(f"[Stage 2] Retrieving top-{top_k} similar examples...")
    
    retrieved_examples = retriever.retrieve(question, top_k=top_k)
    
    if detailed:
        print(f"[Stage 2] Retrieved {len(retrieved_examples)} examples")
        for i, ex in enumerate(retrieved_examples, 1):
            print(f"  Example {i}: score={ex.score:.4f}, answer={ex.answer}")
    
    # Format retrieved examples with CAUTION warning
    retrieved_info_text = format_retrieved_examples_with_caution(
        retrieved_examples,
        include_solution=include_solution,
        max_solution_length=max_solution_length
    )
    
    # Build Stage 2 prompt: review with retrieved examples
    review_prompt = build_review_prompt(
        question=question,
        initial_answer=answer_stage1,
        initial_reasoning=response_stage1,
        retrieved_info=retrieved_info_text,
        enable_tools=enable_tools
    )
    
    prompt_stage2 = apply_chat_template_if_enabled(review_prompt, tokenizer, apply_chat_template)
    
    if detailed:
        print(f"[Stage 2] Generating review response...")
    
    response_stage2 = generate_model_response(model, tokenizer, prompt_stage2, detailed)
    
    # Execute code if needed
    code_executed_stage2 = False
    exec_results_stage2 = []
    
    if enable_tools and "```python" in response_stage2:
        from utils.python_code_execution import process_text_with_code_execution
        
        if detailed:
            print(f"[Stage 2] Executing Python code...")
        
        response_stage2, exec_results_stage2 = process_text_with_code_execution(
            response_stage2,
            share_variables=True
        )
        code_executed_stage2 = bool(exec_results_stage2)
    
    # Extract Stage 2 answer
    answer_stage2 = extract_answer(response_stage2)
    correct_stage2 = check_answer(answer_stage2, ground_truth) if answer_stage2 else False
    
    if detailed:
        print(f"[Stage 2] Answer: {answer_stage2}")
        print(f"[Stage 2] Correct: {correct_stage2}")
    
    # Determine final answer (use Stage 2 if available, else Stage 1)
    final_answer = answer_stage2 if answer_stage2 else answer_stage1
    final_correct = correct_stage2 if answer_stage2 else correct_stage1
    final_response = response_stage2 if answer_stage2 else response_stage1
    
    # Determine case type
    if correct_stage1 and correct_stage2:
        case_type = "BOTH_CORRECT"
    elif not correct_stage1 and correct_stage2:
        case_type = "IMPROVED"  # Stage 2 fixed the error
    elif correct_stage1 and not correct_stage2:
        case_type = "DEGRADED"  # Stage 2 broke it
    else:
        case_type = "BOTH_WRONG"
    
    if detailed:
        print(f"\n[Final] Answer: {final_answer}")
        print(f"[Final] Correct: {final_correct}")
        print(f"[Case Type]: {case_type}")
    
    # Prepare retrieved examples info for logging
    retrieved_info = [
        {
            "index": ex.index,
            "question": ex.question[:200],
            "answer": ex.answer,
            "score": ex.score
        }
        for ex in retrieved_examples
    ]
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": final_answer,
        "final_correct": final_correct,
        "final_verdict": "CORRECT" if final_correct else "INCORRECT",
        "first_answer": answer_stage1,
        "first_correct": correct_stage1,
        "response": final_response,
        "case_type": case_type,
        "stage1_answer": answer_stage1,
        "stage1_correct": correct_stage1,
        "stage1_response": response_stage1,
        "stage2_used": True,
        "stage2_answer": answer_stage2,
        "stage2_correct": correct_stage2,
        "stage2_response": response_stage2,
        "code_executed": code_executed_stage1 or code_executed_stage2,
        "exec_results": exec_results_stage1 + exec_results_stage2,
        "num_code_blocks": len(exec_results_stage1) + len(exec_results_stage2),
        "rag_config": {
            "top_k": top_k,
            "include_solution": include_solution,
            "num_retrieved": len(retrieved_examples),
            "retrieved_examples": retrieved_info
        },
        "tools_config": {
            "enable_tools": enable_tools,
            "code_executed": code_executed_stage1 or code_executed_stage2
        }
    }


def format_retrieved_examples_with_caution(
    examples,
    include_solution: bool = True,
    max_solution_length: int = 400
) -> str:
    """
    Format retrieved examples with CAUTION warnings.
    """
    if not examples:
        return ""
    
    parts = []
    parts.append("=" * 50)
    parts.append("REFERENCE EXAMPLES (Use with CAUTION)")
    parts.append("=" * 50)
    parts.append("")
    parts.append("WARNING: The following examples are retrieved based on keyword similarity.")
    parts.append("They may NOT be directly applicable to your problem.")
    parts.append("- Do NOT blindly copy the solution approach")
    parts.append("- Do NOT assume the answer pattern is the same")
    parts.append("- Use these ONLY as a sanity check for your reasoning")
    parts.append("- Trust your own calculation if it differs from these examples")
    parts.append("")
    
    for i, ex in enumerate(examples, 1):
        parts.append(f"--- Reference {i} (similarity: {ex.score:.2f}) ---")
        parts.append(f"Problem: {ex.question}")
        
        if include_solution:
            solution = ex.solution
            if len(solution) > max_solution_length:
                solution = solution[:max_solution_length] + "..."
            parts.append(f"Solution: {solution}")
        
        parts.append(f"Answer: {ex.answer}")
        parts.append("")
    
    parts.append("=" * 50)
    parts.append("END OF REFERENCE EXAMPLES")
    parts.append("=" * 50)
    
    return "\n".join(parts)


def build_review_prompt(
    question: str,
    initial_answer: str,
    initial_reasoning: str,
    retrieved_info: str,
    enable_tools: bool = False
) -> str:
    """
    Build the Stage 2 review prompt.
    Make it conservative - only change if there's a clear error.
    """
    prompt = f"""Double-check this math problem. Your previous answer was {initial_answer}.

Problem: {question}

{retrieved_info}

CRITICAL: Only change your answer if you find a CLEAR calculation error.
- If unsure, keep your original answer: {initial_answer}
- Do NOT change your answer just because reference examples have different answers
- Reference examples may be for DIFFERENT problem types

Re-solve the problem step by step. Final answer in \\boxed{{}}.
"""
    
    if enable_tools:
        prompt += "\nUse Python code to verify calculations."
    
    return prompt


# For testing
if __name__ == "__main__":
    print("Agent with RAG Review - Quick Test")
    print("=" * 80)
    
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_ground_truth = "18"
    
    try:
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        from models.inference import load_model
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning Agent with RAG Review...")
        print(f"Question: {test_question[:80]}...")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_agent_with_rag_review(
            question=test_question,
            ground_truth=test_ground_truth,
            model=model,
            tokenizer=tokenizer,
            retriever=None,
            top_k=3,
            detailed=True,
            dataset_name="gsm8k",
            enable_tools=False,
            apply_chat_template=True
        )
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Stage 1 Answer: {result['stage1_answer']} ({'Correct' if result['stage1_correct'] else 'Wrong'})")
        print(f"Stage 2 Answer: {result['stage2_answer']} ({'Correct' if result['stage2_correct'] else 'Wrong'})")
        print(f"Final Answer: {result['predicted_answer']}")
        print(f"Expected: {result['ground_truth']}")
        print(f"Final Correct: {result['final_correct']}")
        print(f"Case Type: {result['case_type']}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

