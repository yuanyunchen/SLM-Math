"""
Solver-Checker-Summarizer Multi-Agent Workflow (Chat Mode)
带总结层的Solver-Checker工作流 - Chat模式

ARCHITECTURE:
Similar to stateless but maintains conversation history with [ROLE] tags

CHAT MODE FEATURES:
- Maintains full conversation history
- Uses [SOLVER], [CHECKER], [SUMMARIZER] tags
- Better for larger models (>2B parameters)
- Warning: May cause hallucination in small models

适用场景：
- 大模型 (>=2B)
- 需要上下文连贯性
- 实验性功能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import torch


def run_solver_checker_summarizer_chat_workflow(
    question: str,
    ground_truth: str,
    model,  # Shared model for all roles
    tokenizer,
    max_iterations: int = 5,
    detailed: bool = False,
    dataset_name: str = ""
) -> Dict:
    """
    Run Solver-Checker-Summarizer workflow (Chat Mode)
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        model: Shared model for all roles
        tokenizer: Tokenizer
        max_iterations: Maximum iterations
        detailed: Verbose output
        dataset_name: Dataset name
    
    Returns:
        Dict with workflow results
    """
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    from models.inference import generate_response
    
    if detailed:
        print(f"\n{'='*80}")
        print(f"SOLVER-CHECKER-SUMMARIZER WORKFLOW (Chat)")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"Mode: Chat with conversation history")
        print(f"{'='*80}\n")
    
    # Conversation history
    conversation_history = []
    
    iterations = []
    solver_responses = []
    solver_answers = []
    checker_responses = []
    checker_verdicts = []
    
    predicted_answer = None
    final_verdict = None
    
    # Iterative loop
    for iteration_num in range(1, max_iterations + 1):
        if detailed:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration_num}/{max_iterations}")
            print(f"{'='*80}")
        
        # ========== SOLVER TURN ==========
        if detailed:
            print(f"\n[Solver Turn]")
        
        if iteration_num == 1:
            # First turn: add question
            solver_prompt = format_prompt_standard(question, dataset_name)
            conversation_history.append({
                'role': 'user',
                'content': f"Problem: {question}\n\nPlease solve this step by step and put your answer in \\boxed{{}}."
            })
        else:
            # Subsequent turns: conversation continues
            pass
        
        # Build conversation prompt
        conversation_text = ""
        for msg in conversation_history:
            role_label = msg['role'].upper()
            conversation_text += f"[{role_label}]: {msg['content']}\n\n"
        
        conversation_text += "[SOLVER]: "
        
        # Generate solver response
        inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, max_length=2048)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        solver_response = full_output[len(conversation_text):].strip()
        
        if detailed:
            print(solver_response[:200] + "...")
        
        solver_answer = extract_answer(solver_response)
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        # Add to conversation
        conversation_history.append({
            'role': 'solver',
            'content': solver_response
        })
        
        if detailed:
            print(f"\n[Solver Answer]: {solver_answer}")
        
        # ========== SUMMARIZER (Solver → Checker) ==========
        if detailed:
            print(f"\n[Summarizer: Preparing summary for Checker]")
        
        conversation_text = ""
        for msg in conversation_history:
            role_label = msg['role'].upper()
            conversation_text += f"[{role_label}]: {msg['content']}\n\n"
        
        conversation_text += "[SUMMARIZER]: Now I'll create a concise summary of the solver's solution for the checker to evaluate:\n\n"
        
        inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.5,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        solver_summary = full_output[len(conversation_text):].strip()
        
        if detailed:
            print(f"[Summary]: {solver_summary[:150]}...")
        
        # Add to conversation
        conversation_history.append({
            'role': 'summarizer',
            'content': solver_summary
        })
        
        # ========== CHECKER TURN ==========
        if detailed:
            print(f"\n[Checker Turn]")
        
        conversation_text = ""
        for msg in conversation_history:
            role_label = msg['role'].upper()
            conversation_text += f"[{role_label}]: {msg['content']}\n\n"
        
        conversation_text += "[CHECKER]: Based on the summary, my evaluation is:\n\n"
        
        inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        checker_response = full_output[len(conversation_text):].strip()
        
        if detailed:
            print(checker_response[:200] + "...")
        
        checker_responses.append(checker_response)
        
        # Parse verdict
        from agent.utils import parse_checker_verdict
        checker_verdict = parse_checker_verdict(checker_response)
        
        if checker_verdict not in ["CORRECT", "INCORRECT"]:
            checker_verdict = "INCORRECT"  # Default to INCORRECT
        
        checker_verdicts.append(checker_verdict)
        
        # Add to conversation
        conversation_history.append({
            'role': 'checker',
            'content': checker_response
        })
        
        if detailed:
            print(f"\n[Checker Verdict]: {checker_verdict}")
        
        # ========== SUMMARIZER (Checker → Solver) ==========
        if checker_verdict != "CORRECT":
            if detailed:
                print(f"\n[Summarizer: Preparing feedback for Solver]")
            
            conversation_text = ""
            for msg in conversation_history:
                role_label = msg['role'].upper()
                conversation_text += f"[{role_label}]: {msg['content']}\n\n"
            
            conversation_text += "[SUMMARIZER]: Let me summarize the checker's feedback concisely for the solver:\n\n"
            
            inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            checker_summary = full_output[len(conversation_text):].strip()
            
            if detailed:
                print(f"[Feedback Summary]: {checker_summary[:100]}...")
            
            # Add to conversation
            conversation_history.append({
                'role': 'summarizer',
                'content': checker_summary
            })
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "conversation_length": len(conversation_history)
        }
        
        # Check actual correctness
        is_actually_correct = False
        if solver_answer:
            is_actually_correct = check_answer(solver_answer, ground_truth)
        iteration_data["is_actually_correct"] = is_actually_correct
        
        iterations.append(iteration_data)
        
        # Decision logic
        if checker_verdict == "CORRECT":
            if solver_answer:
                predicted_answer = solver_answer
                final_verdict = "CORRECT"
                if detailed:
                    print(f"\n✓ Checker confirmed CORRECT, breaking")
                break
        
        if iteration_num >= max_iterations:
            if detailed:
                print(f"\nReached max iterations ({max_iterations})")
            break
    
    # If no CORRECT verdict, use last valid answer
    if predicted_answer is None:
        for i in range(len(iterations) - 1, -1, -1):
            iter_answer = iterations[i]['solver_answer']
            if iter_answer and iter_answer.strip():
                predicted_answer = iter_answer
                final_verdict = f"LAST_VALID_ITER_{iterations[i]['iteration']}"
                break
        
        if predicted_answer is None:
            final_verdict = "NO_VALID_ANSWER"
    
    # Check final correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    # Determine case type
    first_answer = solver_answers[0] if solver_answers else None
    first_correct = check_answer(first_answer, ground_truth) if first_answer else False
    
    case_type = None
    if first_correct and final_correct and len(iterations) == 1:
        case_type = "FIRST_TRY_SUCCESS"
    elif not first_correct and final_correct:
        case_type = "IMPROVED"
    elif first_correct and not final_correct:
        case_type = "DEGRADED"
    elif not first_correct and not final_correct:
        case_type = "FAILED"
    else:
        case_type = "OTHER"
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "final_correct": final_correct,
        "final_verdict": final_verdict,
        "total_iterations": len(iterations),
        "iterations": iterations,
        "solver_answers": solver_answers,
        "checker_verdicts": checker_verdicts,
        "first_answer": first_answer,
        "first_correct": first_correct,
        "case_type": case_type,
        "conversation_history": conversation_history,
        "config": {
            "use_summarizer": True,
            "mode": "chat"
        }
    }


# For testing
if __name__ == "__main__":
    print("Solver-Checker-Summarizer (Chat) - Quick Test")
    print("=" * 80)
    
    test_question = "A bakery sells 3 types of bread. Type A costs $2, Type B costs $3, and Type C costs $5. If someone buys 2 of Type A, 1 of Type B, and 1 of Type C, how much do they spend?"
    test_ground_truth = "12"
    
    try:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        from models.inference import load_model
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning Solver-Checker-Summarizer Chat workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_solver_checker_summarizer_chat_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            model=model,
            tokenizer=tokenizer,
            max_iterations=2,
            detailed=True,
            dataset_name="gsm8k"
        )
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Expected: {result['ground_truth']}")
        print(f"Correct: {result['final_correct']}")
        print(f"Case Type: {result['case_type']}")
        print(f"Iterations: {result['total_iterations']}")
        print(f"Conversation Length: {len(result['conversation_history'])} messages")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

