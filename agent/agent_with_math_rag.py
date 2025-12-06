"""
Agent with Math RAG - Few-Shot Retrieval-Augmented Generation
从训练集检索相似题目作为 few-shot examples，辅助模型解题

工作流程:
1. 接收用户问题
2. 从训练集中检索 Top-K 相似题目
3. 将检索到的例题作为 few-shot examples 注入 prompt
4. 模型基于 few-shot examples 生成解答
5. (可选) 执行 Python 代码确保计算准确

优势:
- 利用训练集中的解题范例
- 帮助小模型学习"如何解这类题"
- 不需要额外训练，即插即用
- 预计提升 5-10% 准确率

与其他 Agent 的区别:
- Base Direct: 无 few-shot，无工具
- Agent with Python Tools: 无 few-shot，有代码执行
- Agent with Math RAG: 有 few-shot，可选代码执行
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


def run_agent_with_math_rag(
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
    max_solution_length: int = 500,
    apply_chat_template: bool = False
) -> Dict:
    """
    Run Few-Shot RAG Agent
    
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
        include_solution: Include solution steps in few-shot examples
        max_solution_length: Max length of solution in examples
        apply_chat_template: Whether to apply chat template
    
    Returns:
        Dict with results
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response
    from utils.math_rag import MathRAGRetriever
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"AGENT WITH MATH RAG (Few-Shot)")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Top-K: {top_k}")
        print(f"Python Tools: {'Enabled' if enable_tools else 'Disabled'}")
        print(f"{'='*80}\n")
    
    # Initialize retriever if not provided
    if retriever is None:
        if detailed:
            print("[Initializing RAG retriever...]")
        retriever = MathRAGRetriever(
            dataset_name=dataset_name,
            method="bm25"
        )
    
    # Retrieve similar examples
    if detailed:
        print(f"[Retrieving top-{top_k} similar examples...]")
    
    retrieved_examples = retriever.retrieve(question, top_k=top_k)
    
    if detailed:
        print(f"[Retrieved {len(retrieved_examples)} examples]")
        for i, ex in enumerate(retrieved_examples, 1):
            print(f"  Example {i}: score={ex.score:.4f}, answer={ex.answer}")
    
    # Build prompt with few-shot examples
    few_shot_prompt = retriever.format_examples_as_prompt(
        retrieved_examples,
        include_solution=include_solution,
        max_solution_length=max_solution_length
    )
    
    # Combine few-shot prompt with question
    base_prompt = format_prompt_standard(question, dataset_name)
    
    if few_shot_prompt:
        prompt = few_shot_prompt + base_prompt
    else:
        prompt = base_prompt
    
    # Add tool instruction if enabled
    if enable_tools:
        prompt += "\n\nYou can write Python code in ```python``` blocks to help with calculations. The code will be executed automatically and you'll see the results. Use code for any arithmetic operations to ensure accuracy."
    
    # Apply chat template if enabled
    prompt = apply_chat_template_if_enabled(prompt, tokenizer, apply_chat_template)
    
    if detailed:
        print(f"\n[Prompt length: {len(prompt)} chars]")
        print(f"[Generating response...]")
    
    # Generate response
    if hasattr(model, 'generate_single'):
        # Using inference engine
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
        # Using standard model
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
        from utils.python_code_execution import process_text_with_code_execution
        
        if "```python" in response:
            if detailed:
                print(f"\n[Executing Python code...]")
            
            response_with_output, exec_results = process_text_with_code_execution(
                response,
                share_variables=True
            )
            
            if exec_results:
                code_executed = True
                if detailed:
                    print(f"[Code Execution Results]")
                    for i, result in enumerate(exec_results, 1):
                        if result['success']:
                            print(f"  Block {i}: Success - {result['output'][:50]}")
                        else:
                            print(f"  Block {i}: Error - {result['error'][:50]}")
                
                response = response_with_output
    
    # Extract final answer
    predicted_answer = extract_answer(response)
    
    if detailed:
        print(f"\n[Final Answer]: {predicted_answer}")
        print(f"[Ground Truth]: {ground_truth}")
    
    # Check correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    if detailed:
        status = "CORRECT" if final_correct else "INCORRECT"
        print(f"[Result]: {status}")
    
    # Determine case type
    case_type = "FIRST_TRY_SUCCESS" if final_correct else "FAILED"
    final_verdict = "CORRECT" if final_correct else "INCORRECT"
    
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
        "predicted_answer": predicted_answer,
        "final_correct": final_correct,
        "final_verdict": final_verdict,
        "first_answer": predicted_answer,
        "first_correct": final_correct,
        "response": response,
        "case_type": case_type,
        "code_executed": code_executed,
        "exec_results": exec_results,
        "num_code_blocks": len(exec_results),
        "rag_config": {
            "top_k": top_k,
            "include_solution": include_solution,
            "num_retrieved": len(retrieved_examples),
            "retrieved_examples": retrieved_info
        },
        "tools_config": {
            "enable_tools": enable_tools,
            "code_executed": code_executed
        }
    }


# For testing
if __name__ == "__main__":
    print("Agent with Math RAG - Quick Test")
    print("=" * 80)
    
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_ground_truth = "18"
    
    try:
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        from models.inference import load_model
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning Agent with Math RAG...")
        print(f"Question: {test_question[:80]}...")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_agent_with_math_rag(
            question=test_question,
            ground_truth=test_ground_truth,
            model=model,
            tokenizer=tokenizer,
            retriever=None,  # Will be created automatically
            top_k=3,
            detailed=True,
            dataset_name="gsm8k",
            enable_tools=False
        )
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Expected: {result['ground_truth']}")
        print(f"Correct: {result['final_correct']}")
        print(f"Case Type: {result['case_type']}")
        print(f"Retrieved Examples: {result['rag_config']['num_retrieved']}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()














