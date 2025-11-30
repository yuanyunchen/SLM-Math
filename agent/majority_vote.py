"""
Majority Vote Agent Workflow
通过不同随机种子运行多轮，然后进行多数投票

关键设计：
- 第一个run使用FIRST_ROUND_SOLVER_CONFIG（确定性），与其他agent保持一致
- 后续runs使用不同seed的随机配置
- 这样保证first_try_accuracy在不同agent间可比较
"""

import torch
from typing import Dict, List
from collections import Counter
from agent.unified_config import (
    FIRST_ROUND_SOLVER_CONFIG,
    MAJORITY_VOTE_OTHER_RUNS_CONFIG
)


def generate_response_deterministic(
    model,
    tokenizer,
    prompt: str,
    config: dict,
    detailed: bool = False
):
    """
    Generate response with deterministic config (no sampling).
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Input prompt
        config: Generation config dict
        detailed: Whether to show detailed output
    
    Returns:
        Generated response string
    """
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
            max_new_tokens=config.get('max_new_tokens', 2048),
            temperature=config.get('temperature', 0.0),
            do_sample=config.get('do_sample', False),
            top_p=config.get('top_p', 1.0),
            top_k=config.get('top_k', 1),
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=config.get('repetition_penalty', 1.1),
            stopping_criteria=stopping_criteria,
            streamer=streamer
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response


def generate_response_with_seed(
    model,
    tokenizer,
    prompt: str,
    seed: int,
    config: dict,
    detailed: bool = False
):
    """
    Generate response with specific random seed and config.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Input prompt
        seed: Random seed
        config: Generation config dict
        detailed: Whether to show detailed output
    
    Returns:
        Generated response string
    """
    from transformers import TextStreamer, StoppingCriteriaList
    from models.inference import StopOnBoxedAnswer
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
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
            max_new_tokens=config.get('max_new_tokens', 2048),
            temperature=config.get('temperature', 0.7),
            do_sample=config.get('do_sample', True),
            top_p=config.get('top_p', 0.95),
            top_k=config.get('top_k', 50),
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=config.get('repetition_penalty', 1.2),
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
    dataset_name: str = ""
) -> Dict:
    """
    Run majority vote workflow with multiple runs using different seeds.
    
    关键设计：
    - Run 1: 使用FIRST_ROUND_SOLVER_CONFIG（确定性，与其他agent一致）
    - Run 2+: 使用不同seed的随机配置
    
    Args:
        question: Math problem question
        ground_truth: Ground truth answer
        model: Model instance
        tokenizer: Tokenizer instance
        num_runs: Number of runs with different seeds
        temperature: Sampling temperature for runs 2+ (default 0.7)
        top_p: Nucleus sampling parameter for runs 2+ (default 0.95)
        detailed: Whether to show detailed output
        dataset_name: Dataset name
    
    Returns:
        Dictionary with workflow results
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from utils.prompt_utils import extract_answer, check_answer, format_prompt_standard
    
    # Format prompt (use standard prompt)
    prompt = format_prompt_standard(question, dataset_name)
    
    # Run multiple times with different configs
    runs = []
    answers = []
    
    for run_num in range(1, num_runs + 1):
        if detailed:
            print(f"\n--- Run {run_num}/{num_runs} ---")
        
        try:
            if run_num == 1:
                # First run: use deterministic config (FIRST_ROUND_SOLVER_CONFIG)
                # This ensures first_try_accuracy is comparable across all agents
                if detailed:
                    print("[Using deterministic config for first run]")
                response = generate_response_deterministic(
                    model, tokenizer, prompt, 
                    config=FIRST_ROUND_SOLVER_CONFIG,
                    detailed=detailed
                )
                seed_used = None  # No seed for deterministic
                config_used = "FIRST_ROUND_DETERMINISTIC"
            else:
                # Subsequent runs: use random config with different seeds
                seed = run_num * 42  # Different seed for each run
                config = MAJORITY_VOTE_OTHER_RUNS_CONFIG.copy()
                # Override with user-specified temperature and top_p
                config['temperature'] = temperature
                config['top_p'] = top_p
                
                if detailed:
                    print(f"[Using random config with seed={seed}, temp={temperature}]")
                response = generate_response_with_seed(
                    model, tokenizer, prompt, seed,
                    config=config,
                    detailed=detailed
                )
                seed_used = seed
                config_used = f"RANDOM_SEED_{seed}"
                
        except Exception as e:
            response = f"Error: {e}"
            seed_used = run_num if run_num > 1 else None
            config_used = "ERROR"
        
        answer = extract_answer(response)
        
        run_data = {
            "run": run_num,
            "seed": seed_used,
            "config": config_used,
            "prompt": prompt,
            "response": response,
            "answer": answer,
            "temperature": 0.0 if run_num == 1 else temperature,
            "top_p": 1.0 if run_num == 1 else top_p,
            "is_deterministic": run_num == 1
        }
        runs.append(run_data)
        
        if answer:
            answers.append(answer)
            
        if detailed:
            print(f"Answer: {answer}")
    
    # Majority vote with conservative first-run preference
    # Strategy: Only override first run when there's clear consensus (>50%)
    # This reduces degradation from noisy random runs
    predicted_answer = None
    final_verdict = "MAJORITY_VOTE"
    
    if answers:
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        most_common_answer = most_common[0]
        vote_count = most_common[1]
        total_runs = len(answers)
        
        # Get first run's answer
        first_run_answer = runs[0]['answer'] if runs and runs[0]['answer'] else None
        first_run_vote_count = answer_counts.get(first_run_answer, 0) if first_run_answer else 0
        
        # Decision logic (conservative - prioritize first run):
        # 1. If most common has clear majority (>50%), use it (strong consensus)
        # 2. If first run answer is in a tie with most common, prefer first run
        # 3. If no clear majority, default to first run (trust deterministic baseline)
        if vote_count / total_runs > 0.5:
            # Clear majority - strong signal to use consensus answer
            predicted_answer = most_common_answer
            final_verdict = f"MAJORITY_VOTE_{vote_count}/{total_runs}"
        elif first_run_answer and first_run_vote_count == vote_count:
            # Tie between first run and another answer - prefer first run
            predicted_answer = first_run_answer
            final_verdict = f"FIRST_TIE_{first_run_vote_count}/{total_runs}"
        elif first_run_answer:
            # No clear majority - default to first run (conservative)
            # This prevents weak plurality from overriding deterministic baseline
            predicted_answer = first_run_answer
            final_verdict = f"FIRST_DEFAULT_{first_run_vote_count}/{total_runs}"
        else:
            # First run has no answer, fall back to most common
            predicted_answer = most_common_answer
            final_verdict = f"PLURALITY_FALLBACK_{vote_count}/{total_runs}"
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
    
    # Determine first_answer and first_correct
    # first_answer is from Run 1 (deterministic, comparable to other agents)
    first_answer = runs[0]['answer'] if runs else None
    first_correct = runs[0]['is_correct'] if runs else False
    
    # Determine case type
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
