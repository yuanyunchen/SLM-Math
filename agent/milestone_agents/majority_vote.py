"""
Majority Vote Agent Workflow
通过不同随机种子运行多轮，然后进行多数投票
"""

import torch
from typing import Dict, List
from collections import Counter
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


def generate_response_with_seed(
    model,
    tokenizer,
    prompt: str,
    seed: int,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    detailed: bool = False
):
    """
    Generate response with specific random seed.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Input prompt
        seed: Random seed
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        detailed: Whether to show detailed output
    
    Returns:
        Generated response string
    """
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Check if using inference engine or standard model
    if hasattr(model, 'generate_single'):
        # Using inference engine (vLLM or TransformersEngine)
        return model.generate_single(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=DO_SAMPLE,
            top_p=top_p,
            repetition_penalty=REPETITION_PENALTY,
            detailed=detailed
        )
    
    # Using standard PyTorch model
    from transformers import TextStreamer, StoppingCriteriaList
    from models.inference import StopOnBoxedAnswer
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    prompt_length = inputs['input_ids'].shape[1]
    
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None
    
    stopping_criteria = StoppingCriteriaList([StopOnBoxedAnswer(tokenizer, prompt_length)])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=DO_SAMPLE,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
            stopping_criteria=stopping_criteria,
            streamer=streamer
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response


def run_majority_vote_workflow(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    num_runs: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.95,
    detailed: bool = False,
    dataset_name: str = "",
    apply_chat_template: bool = False
) -> Dict:
    """
    Run majority vote workflow with multiple runs using different seeds.
    
    Args:
        question: Math problem question
        ground_truth: Ground truth answer
        model: Model instance
        tokenizer: Tokenizer instance
        num_runs: Number of runs with different seeds
        temperature: Sampling temperature (default 0.7)
        top_p: Nucleus sampling parameter (default 0.95)
        detailed: Whether to show detailed output
        dataset_name: Dataset name
        apply_chat_template: Whether to apply chat template to prompts
    
    Returns:
        Dictionary with workflow results
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    
    # Format prompt (use standard prompt)
    prompt = format_prompt_standard(question, dataset_name)
    prompt = apply_chat_template_if_enabled(prompt, tokenizer, apply_chat_template)
    
    # Run multiple times with different seeds
    runs = []
    answers = []
    
    for run_num in range(1, num_runs + 1):
        seed = run_num  # Use run number as seed
        
        try:
            response = generate_response_with_seed(
                model, tokenizer, prompt, seed, temperature, top_p, detailed
            )
        except Exception as e:
            response = f"Error: {e}"
        
        answer = extract_answer(response)
        
        run_data = {
            "run": run_num,
            "seed": seed,
            "prompt": prompt,
            "response": response,
            "answer": answer,
            "temperature": temperature,
            "top_p": top_p
        }
        runs.append(run_data)
        
        if answer:
            answers.append(answer)
    
    # Majority vote
    predicted_answer = None
    final_verdict = "MAJORITY_VOTE"
    
    if answers:
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        predicted_answer = most_common[0]
        vote_count = most_common[1]
        total_runs = len(answers)
        
        # Check if there's a clear majority (more than 50%)
        if vote_count / total_runs > 0.5:
            final_verdict = f"MAJORITY_VOTE_{vote_count}/{total_runs}"
        else:
            # No clear majority, use most common anyway
            final_verdict = f"PLURALITY_VOTE_{vote_count}/{total_runs}"
    else:
        # No valid answers
        final_verdict = "NO_VALID_ANSWER"
    
    # Check final correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    # Check each run's correctness
    for run_data in runs:
        if run_data['answer']:
            run_data['is_correct'] = check_answer(run_data['answer'], ground_truth)
        else:
            run_data['is_correct'] = False
    
    # Determine case type
    first_answer = answers[0] if answers else None
    first_correct = check_answer(first_answer, ground_truth) if first_answer else False
    
    case_type = None
    if final_correct:
        if first_correct:
            case_type = "FIRST_RUN_SUCCESS"
        else:
            case_type = "MAJORITY_IMPROVED"
    else:
        if first_correct:
            case_type = "MAJORITY_DEGRADED"
        else:
            case_type = "MAJORITY_FAILED"
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "final_correct": final_correct,
        "final_verdict": final_verdict,
        "num_runs": num_runs,
        "runs": runs,
        "answers": answers,
        "answer_counts": dict(Counter(answers)) if answers else {},
        "first_answer": first_answer,
        "first_correct": first_correct,
        "case_type": case_type
    }

