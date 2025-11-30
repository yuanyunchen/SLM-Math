"""
Solver-Checker Chat-based Multi-Agent Workflow (Optimized)
使用chat template的优化对话实现

OPTIMIZED CHAT MODE - 使用模型的chat template
- 使用 tokenizer.apply_chat_template()
- 正确的消息格式 [{"role": "user/assistant", "content": "..."}]
- 减少幻觉风险
- 更符合模型训练方式

适用: 支持chat template的模型（Qwen系列支持）
"""

from typing import Dict, List
import torch
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
from agent.unified_config import FIRST_ROUND_SOLVER_CONFIG


class StopAfterBoxed(StoppingCriteria):
    """Stop generation after \\boxed{} is complete"""
    def __init__(self, tokenizer, min_new_tokens: int = 50):
        self.tokenizer = tokenizer
        self.min_new_tokens = min_new_tokens
        self.initial_length = None
        self.boxed_start_seen = False
        self.brace_count = 0
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Record initial length on first call
        if self.initial_length is None:
            self.initial_length = input_ids.shape[1]
        
        # Must generate at least min_new_tokens
        new_tokens_count = input_ids.shape[1] - self.initial_length
        if new_tokens_count < self.min_new_tokens:
            return False
        
        # Decode the last few tokens to check for \boxed{}
        last_tokens = input_ids[0, -30:]  # Check last 30 tokens
        text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        
        # Check if we have \boxed{...}
        if '\\boxed{' in text or 'boxed{' in text:
            self.boxed_start_seen = True
            # Count braces after 'boxed{'
            boxed_pos = max(text.rfind('\\boxed{'), text.rfind('boxed{'))
            text_after_boxed = text[boxed_pos:]
            self.brace_count = text_after_boxed.count('{') - text_after_boxed.count('}')
        
        # Stop if we've closed all braces after seeing \boxed{
        if self.boxed_start_seen and self.brace_count == 0:
                    return True
        
        return False
    

class StopAfterVerdict(StoppingCriteria):
    """Stop generation after VERDICT: CORRECT/INCORRECT"""
    def __init__(self, tokenizer, min_length: int = 50):
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.initial_length = None
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Record initial length on first call
        if self.initial_length is None:
            self.initial_length = input_ids.shape[1]
        
        # Must generate at least min_length tokens
        new_tokens_count = input_ids.shape[1] - self.initial_length
        if new_tokens_count < self.min_length:
            return False
        
        # Check recent tokens for verdict
        recent_tokens = input_ids[0, -40:]  # Check last 40 tokens
        text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
        text_upper = text.upper()
        
        # Look for verdict patterns (more flexible)
        verdict_patterns = [
            'VERDICT: CORRECT',
            'VERDICT: INCORRECT', 
            'VERDICT:CORRECT',
            'VERDICT:INCORRECT',
            'ANSWER: CORRECT',
            'ANSWER: INCORRECT'
        ]
        
        for pattern in verdict_patterns:
            if pattern in text_upper:
                return True
        
        # Also check if we have standalone verdict at end
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1].strip().upper()
            if last_line in ['CORRECT', 'INCORRECT']:
                return True
        
        # Stop if too long (safety)
        if new_tokens_count > 300:
            return True
        
        return False


def run_solver_checker_chat_workflow(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    max_iterations: int = 5,
    detailed: bool = False,
    dataset_name: str = ""
) -> Dict:
    """
    Run solver-checker chat-based workflow using chat template.
    
    Args:
        question: Math problem question
        ground_truth: Ground truth answer
        model: Shared model for both solver and checker
        tokenizer: Shared tokenizer
        max_iterations: Maximum number of iterations
        detailed: Whether to show detailed output
        dataset_name: Dataset name
    
    Returns:
        Dictionary with workflow results
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from utils.prompt_utils import extract_answer, check_answer
    from agent.utils import parse_checker_verdict, parse_checker_tip
    
    # Check if model supports chat template
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    
    if not has_chat_template:
        if detailed:
            print("⚠️  Model doesn't have chat_template, falling back to simple format")
    
    # Conversation messages in OpenAI format
    messages = []
    
    # Storage for all iterations
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
            print(f"\n{'='*40}")
            print(f"Iteration {iteration_num}/{max_iterations}")
            print(f"{'='*40}")
        
        # Step 1: Solver turn
        if iteration_num == 1:
            # First turn: ask the question (use standard format for consistency)
            from utils.prompt_utils import format_prompt_standard
            user_message = format_prompt_standard(question, dataset_name)
            use_unified_config = True
        else:
            # Subsequent turns: provide feedback with FULL problem context
            # Key fix: Always include the original question to prevent hallucination
            user_message = f"""The previous solution was incorrect.

Original Problem: {question}

Feedback: {checker_feedback}

Please solve this problem again from scratch. Show your reasoning step by step, and put your final answer within \\boxed{{}}."""
            use_unified_config = False
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Generate solver response
        try:
            if use_unified_config:
                # First round: use unified config (deterministic)
                solver_response = generate_with_chat_template(
                    model, tokenizer, messages, 
                    max_new_tokens=FIRST_ROUND_SOLVER_CONFIG.get('max_new_tokens', 2048),
                    temperature=FIRST_ROUND_SOLVER_CONFIG['temperature'],
                    detailed=detailed,
                    has_chat_template=has_chat_template,
                    top_p=FIRST_ROUND_SOLVER_CONFIG['top_p'],
                    do_sample=FIRST_ROUND_SOLVER_CONFIG['do_sample']
                )
            else:
                # Subsequent rounds: use agent's own config (allow randomness)
                from agent.unified_config import SUBSEQUENT_ROUND_CONFIG
                solver_response = generate_with_chat_template(
                    model, tokenizer, messages, 
                    max_new_tokens=SUBSEQUENT_ROUND_CONFIG.get('max_new_tokens', 2048),
                    temperature=SUBSEQUENT_ROUND_CONFIG['temperature'],
                    detailed=detailed,
                    has_chat_template=has_chat_template,
                    role="solver",
                    top_p=SUBSEQUENT_ROUND_CONFIG['top_p'],
                    do_sample=SUBSEQUENT_ROUND_CONFIG['do_sample']
                )
        except Exception as e:
            solver_response = f"Error: {e}"
        
        # Fallback if empty
        if not solver_response.strip():
            if detailed:
                print("⚠️  Empty solver response, using fallback")
            from utils.prompt_utils import format_prompt_standard
            from models.inference import generate_response
            fallback_prompt = format_prompt_standard(question, dataset_name)
            try:
                solver_response = generate_response(model, tokenizer, fallback_prompt, "standard", detailed)
            except Exception as e:
                solver_response = f"Error: {e}"
        
        solver_answer = extract_answer(solver_response)
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        # Add solver response to conversation
        messages.append({
            "role": "assistant",
            "content": solver_response
        })
        
        if detailed:
            print(f"\n{'─'*40}")
            print(f"Solver answer: {solver_answer}")
        
        # Step 2: Checker turn - Use model to verify reasoning
        checker_verdict = "CORRECT"  # Default to CORRECT (trust solver)
        checker_feedback = ""
        checker_response = ""
        
        try:
            # Build checker prompt and have model verify
            from agent.utils import format_prompt_checker, parse_checker_verdict, generate_response_checker
            
            checker_prompt = format_prompt_checker(question, solver_response, dataset_name)
            
            # Add checker message to conversation for context
            messages.append({
                "role": "user", 
                "content": f"Please verify this solution:\n{checker_prompt}"
            })
            
            # Generate checker response using model (not ground truth!)
            checker_response = generate_response_checker(
                model, tokenizer, checker_prompt, detailed
            )
            
            # Parse verdict from model's response
            checker_verdict = parse_checker_verdict(checker_response)
            
            # Remove the checker message from conversation (keep only solver messages)
            messages.pop()
                
        except Exception as e:
            checker_response = f"Error: {e}"
            checker_verdict = "CORRECT"  # Default to CORRECT on error (trust solver)
            checker_feedback = ""
        
        if detailed:
            print(f"\nChecker verdict: {checker_verdict}")
            if checker_feedback:
                print(f"Feedback: {checker_feedback}")
        
        checker_responses.append(checker_response)
        checker_verdicts.append(checker_verdict)
        
        # Extract feedback for next iteration
        checker_feedback = ""
        if checker_verdict == "INCORRECT":
            checker_feedback = parse_checker_tip(checker_response)
            if not checker_feedback:
                checker_feedback = "The previous solution was incorrect. Please reconsider the problem."
        
        if detailed:
            print(f"Checker verdict: {checker_verdict}")
            if checker_feedback:
                print(f"Feedback: {checker_feedback[:100]}...")
        
        # Store iteration data
        iteration_data = {
            "iteration": iteration_num,
            "solver_response": solver_response,
            "solver_answer": solver_answer,
            "checker_response": checker_response,
            "checker_verdict": checker_verdict,
            "checker_feedback": checker_feedback,
            "conversation_length": len(messages)
        }
        
        # Check correctness
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
                    print(f"✓ Checker confirmed CORRECT, breaking")
                break
        
        # If not correct and not at max iterations, continue
        if iteration_num >= max_iterations:
            if detailed:
                print(f"Reached max iterations ({max_iterations})")
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
    if first_correct and final_correct:
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
        "messages": messages  # Store full conversation
    }


def generate_with_chat_template(
    model,
    tokenizer,
    messages: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    detailed: bool = False,
    has_chat_template: bool = True,
    role: str = "solver",  # "solver" or "checker"
    top_p: float = 0.95,
    do_sample: bool = None  # None means auto-detect from temperature
) -> str:
    """
    Generate response using chat template if available.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        messages: List of messages in OpenAI format
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        detailed: Whether to show detailed output
        has_chat_template: Whether tokenizer has chat template
        role: "solver" or "checker" (for appropriate stopping criteria)
    
    Returns:
        Generated response text
    """
    # Apply chat template if available
    if has_chat_template:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            if detailed:
                print(f"Chat template failed: {e}, using fallback")
            # Fallback to simple format
            prompt = format_messages_simple(messages)
    else:
        # Fallback to simple format
        prompt = format_messages_simple(messages)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Get model device
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # Setup streamer if detailed
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None
    
    # Setup stopping criteria based on role - ALWAYS use StopAfterBoxed for solver
    stopping_criteria = StoppingCriteriaList()
    input_length = inputs['input_ids'].shape[1]
    
    if role == "solver":
        # Use StopAfterBoxed to prevent hallucination
        stopping_criteria.append(StopAfterBoxed(tokenizer, min_new_tokens=30))
    elif role == "checker":
        stopping_criteria.append(StopAfterVerdict(tokenizer, min_length=50))
    
    # Determine do_sample: if not specified, auto-detect from temperature
    if do_sample is None:
        do_sample = temperature > 0
    
    generation_kwargs = {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature if temperature > 0 else 1.0,  # Avoid division by zero
        'do_sample': do_sample,
        'top_p': top_p,
        'pad_token_id': tokenizer.eos_token_id,
        'repetition_penalty': 1.2,  # Increased to reduce repetition
        'streamer': streamer,
        'stopping_criteria': stopping_criteria  # Always apply stopping criteria
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    # Decode only the new tokens (excluding the input prompt)
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return response


def format_messages_simple(messages: List[Dict]) -> str:
    """
    Fallback: format messages without chat template.
    
    Args:
        messages: List of messages
    
    Returns:
        Formatted prompt string
    """
    prompt = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"
        elif role == "system":
            prompt += f"System: {content}\n\n"
    
    # Add prompt for next assistant response
    if messages and messages[-1]['role'] != 'assistant':
        prompt += "Assistant: "
    
    return prompt


def build_checker_prompt(question: str, solver_response: str, solver_answer: str) -> str:
    """
    Build checker prompt for verification.
    
    Args:
        question: Original question
        solver_response: Solver's full response
        solver_answer: Extracted answer
    
    Returns:
        Checker prompt text
    """
    # Extract reasoning from solver response
    import re
    reasoning = solver_response
    # Remove code blocks
    reasoning = re.sub(r'```.*?```', '', reasoning, flags=re.DOTALL)
    # Truncate if too long
    if len(reasoning) > 400:
        reasoning = reasoning[:400] + "..."
    
    # Direct prompt that guides model to solve then compare
    prompt = f"""Problem: {question}

Let me solve this step by step to verify the answer of {solver_answer}.

Solution:"""
    
    return prompt


def generate_checker_direct(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
    detailed: bool = False
) -> str:
    """
    Generate checker response using direct completion (no chat template).
    This is more reliable for verification tasks.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Direct prompt string
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        detailed: Whether to show detailed output
    
    Returns:
        Generated response text
    """
    import torch
    
    # Tokenize directly
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Get model device
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    input_length = inputs['input_ids'].shape[1]
    
    # Setup streamer if detailed
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None
    
    # Setup stopping criteria for verdict
    stopping_criteria = StoppingCriteriaList([
        StopAfterVerdict(tokenizer, min_length=50)
    ])
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            stopping_criteria=stopping_criteria,
            streamer=streamer
        )
    
    # Decode only new tokens
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return response


def parse_checker_verdict_enhanced(response: str) -> str:
    """
    Enhanced verdict parser that handles multiple response formats.
    
    Args:
        response: Checker's response
    
    Returns:
        "CORRECT" or "INCORRECT" (defaults to INCORRECT if unclear)
    """
    import re
    upper = response.upper()
    
    # Method 1: Look for explicit "VERDICT: XXX" pattern
    verdict_match = re.search(r'VERDICT\s*:\s*(CORRECT|INCORRECT)', upper)
    if verdict_match:
        return verdict_match.group(1)
    
    # Method 2: Look for "CORRECT ANSWER: XXX" or "ANSWER: XXX"  
    answer_match = re.search(r'(?:CORRECT\s+)?ANSWER\s*:\s*(CORRECT|INCORRECT)', upper)
    if answer_match:
        return answer_match.group(1)
    
    # Method 3: Look for verdict as last word/line
    lines = [line.strip() for line in upper.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        if 'CORRECT' in last_line and 'INCORRECT' not in last_line:
            return "CORRECT"
        elif 'INCORRECT' in last_line:
            return "INCORRECT"
    
    # Method 4: Look anywhere in response but prioritize end
    # Split into sentences and check from end
    sentences = re.split(r'[.!?]\s+', upper)
    for sent in reversed(sentences[-3:]):  # Check last 3 sentences
        if 'CORRECT' in sent and 'INCORRECT' not in sent and len(sent) < 100:
            # Make sure it's not part of explaining the problem
            if any(keyword in sent for keyword in ['VERDICT', 'ANSWER', 'RESPONSE', 'CONCLUSION']):
                return "CORRECT"
        elif 'INCORRECT' in sent and len(sent) < 100:
            if any(keyword in sent for keyword in ['VERDICT', 'ANSWER', 'RESPONSE', 'CONCLUSION']):
                return "INCORRECT"
    
    # Method 5: Check overall tone (last resort)
    # If response is short and mentions correctness positively
    if len(response) < 300:
        positive_indicators = upper.count('CORRECT') + upper.count('ACCURATE') + upper.count('RIGHT')
        negative_indicators = upper.count('INCORRECT') + upper.count('WRONG') + upper.count('ERROR')
        
        if positive_indicators > 0 and negative_indicators == 0:
            return "CORRECT"
        elif negative_indicators > 0 and positive_indicators == 0:
            return "INCORRECT"
    
    # Default to INCORRECT if can't determine (conservative)
    return "INCORRECT"


# For testing
if __name__ == "__main__":
    print("Solver-Checker Chat (Optimized) - Quick Test")
    print("=" * 80)
    
    test_question = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
    test_ground_truth = "3"
    
    try:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        from agent.solver_checker_base import load_model
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning optimized chat workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_solver_checker_chat_workflow(
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
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
