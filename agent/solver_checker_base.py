"""
Solver-Checker Base Implementation (Archive Version)
独立实现的solver-checker工作流，完全保留原始archive逻辑
不依赖外部utils文件，所有代码自包含
"""

import re
import torch
from pathlib import Path
from typing import Optional, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from collections import Counter
from models.generation_config import (
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY,
    CHECKER_MAX_TOKENS, CHECKER_TEMPERATURE, CHECKER_TOP_P, CHECKER_REPETITION_PENALTY
)


# ============================================================================
# STOPPING CRITERIA
# ============================================================================

class StopOnBoxedAnswer(StoppingCriteria):
    """Halts generation once a \\boxed{} answer is produced."""
    
    def __init__(self, tokenizer, prompt_token_len: int):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len
    
    def _has_boxed_answer(self, text: str) -> bool:
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return False
        return "}" in text[idx:]
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_token_len:]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if self._has_boxed_answer(text):
            return True
        return False


class StopAfterCheckerConclusion(StoppingCriteria):
    """Halts checker generation once a VERDICT line is produced."""
    
    def __init__(self, tokenizer, prompt_token_len: int):
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_token_len:]
        if generated_ids.numel() == 0:
            return False
        
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        upper = text.upper()
        
        # Check for complete VERDICT pattern
        verdict_pattern = r'VERDICT\s*:\s*(CORRECT|INCORRECT|UNCLEAR)'
        match = re.search(verdict_pattern, upper)
        
        if match:
            return True
        
        # Also stop if we see verdict words as standalone
        if len(text.strip()) > 5:
            if re.search(r'(^|\n|\s)(CORRE+CT|INCORREE+CT|UNCLEAR)(\s|$|\n)', upper):
                return True
            if re.search(r'(^|\n|\s)(COR+ECT|INCOR+ECT)(\s|$|\n)', upper):
                return True
        
        # Stop if response is getting too long
        if len(text) > 150:
            return True
        
        return False


# ============================================================================
# MODEL LOADING AND GENERATION
# ============================================================================

def load_model(model_name: str, base_path: Path):
    """Load model and tokenizer from disk"""
    model_dir = base_path / 'pretrained_models' / model_name
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} not found!")
    
    model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}!")
    
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "cuda" if device == "cuda" else "cpu"
    
    if device == "cuda":
        try:
            cuda_count = torch.cuda.device_count()
            cuda_name = torch.cuda.get_device_name(0) if cuda_count > 0 else "<unknown>"
        except Exception:
            cuda_count = 0
            cuda_name = "<unknown>"
        print(f"CUDA available: True | GPU count: {cuda_count} | Using device: {cuda_name}")
    else:
        print("CUDA available: False | Using CPU for inference")
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    print(f"Model loaded successfully on {device}\n")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, mode: str, detailed: bool = False):
    """Generate response from model given a prompt."""
    
    # Check if using inference engine
    if hasattr(model, 'generate_single'):
        # Using inference engine (vLLM or TransformersEngine)
        # Set parameters based on mode
        if mode == "checker":
            max_new_tokens = CHECKER_MAX_TOKENS
            temperature = CHECKER_TEMPERATURE
            do_sample = DO_SAMPLE
            top_p = CHECKER_TOP_P
            repetition_penalty = CHECKER_REPETITION_PENALTY
        else:
            max_new_tokens = MAX_NEW_TOKENS
            temperature = TEMPERATURE
            do_sample = DO_SAMPLE
            top_p = TOP_P
            repetition_penalty = REPETITION_PENALTY
        
        return model.generate_single(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            detailed=detailed
        )
    
    # Using standard PyTorch model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    
    # Get model device - handle both inference engines and standard models
    try:
        if hasattr(model, 'device'):
            model_device = model.device
        else:
            model_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    prompt_length = inputs['input_ids'].shape[1]
    
    if detailed:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None
    
    stopping_criteria = StoppingCriteriaList()
    
    gen_kwargs = {
        "temperature": 0.1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,
        "streamer": streamer,
    }
    
    model_max_length = getattr(tokenizer, 'model_max_length', None)
    if model_max_length and model_max_length > 0:
        available_for_generation = max(2048, model_max_length - prompt_length - 100)
    else:
        available_for_generation = 2048
    
    if mode in {"standard", "thinking", "solver"}:
        stopping_criteria.append(StopOnBoxedAnswer(tokenizer, prompt_length))
        gen_kwargs["max_new_tokens"] = available_for_generation
        gen_kwargs["stopping_criteria"] = stopping_criteria
    elif mode == "checker":
        stopping_criteria.append(StopAfterCheckerConclusion(tokenizer, prompt_length))
        gen_kwargs["max_new_tokens"] = min(CHECKER_MAX_TOKENS, available_for_generation)
        gen_kwargs["stopping_criteria"] = stopping_criteria
        gen_kwargs["temperature"] = CHECKER_TEMPERATURE
        gen_kwargs["do_sample"] = DO_SAMPLE
        gen_kwargs["top_p"] = CHECKER_TOP_P
        gen_kwargs["repetition_penalty"] = CHECKER_REPETITION_PENALTY
    else:
        gen_kwargs["max_new_tokens"] = available_for_generation
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response


# ============================================================================
# ANSWER EXTRACTION AND CHECKING
# ============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text."""
    patterns = [
        r'Final Answer[:\s]*\\boxed\{([^}]+)\}',
        r'Final Answer[:\s]+\$?([+-]?\d+\.?\d*)',
        r'####\s*([+-]?\d+\.?\d*)',
        r'(?:answer|Answer|ANSWER)[\s:=]+\$?([+-]?\d+\.?\d*)',
        r'\\boxed\{([^}]+)\}',
        r'\$([+-]?\d+\.?\d*)\s*$',
        r'([+-]?\d+\.?\d*)\s*(?:dollars?|eggs?|meters?|bolts?|people?|students?|items?)\s*(?:\.|$)',
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).strip()
            if answer:
                return normalize_answer(answer)
    
    last_numbers = re.findall(r'([+-]?\d+\.?\d*)', text[-200:])
    if last_numbers:
        return normalize_answer(last_numbers[-1])
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer to standardized format."""
    answer = answer.strip()
    answer = answer.replace(',', '').replace('$', '').replace('%', '')
    answer = answer.replace(' ', '')
    
    try:
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                if den != 0:
                    answer = str(num / den)
        
        float_val = float(answer)
        if float_val.is_integer():
            answer = str(int(float_val))
        else:
            answer = str(float_val)
    except:
        pass
    
    return answer.lower()


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if not predicted or not ground_truth:
        return False
    
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)
    
    if pred_normalized == gt_normalized:
        return True
    
    try:
        pred_num = float(pred_normalized)
        gt_num = float(gt_normalized)
        return abs(pred_num - gt_num) < 1e-3
    except:
        pass
    
    return pred_normalized in gt_normalized or gt_normalized in pred_normalized


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def format_prompt_solver(question: str, checker_feedback: str = "", dataset_name: str = "") -> str:
    """Format prompt for solver agent."""
    if checker_feedback and checker_feedback.strip():
        return f"""{question}

Previous attempt feedback: {checker_feedback}

Please reconsider the problem with the feedback in mind. Reason step by step, and put your final answer within \\boxed{{}}."""
    else:
        return f"""{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""


def extract_solver_cot(solver_response: str) -> str:
    """Extract Chain of Thought reasoning from solver response."""
    cot = re.sub(r'```python.*?```', '', solver_response, flags=re.DOTALL)
    cot = re.sub(r'```output.*?```', '', cot, flags=re.DOTALL)
    cot = re.sub(r'```.*?```', '', cot, flags=re.DOTALL)
    cot = re.sub(r'\\boxed\{[^}]+\}', '[final answer]', cot)
    
    lines = [line.strip() for line in cot.split('\n') if line.strip()]
    cot = ' '.join(lines)
    
    if len(cot) > 400:
        sentences = re.split(r'[.!?]\s+', cot)
        shortened = ""
        for sent in sentences:
            if len(shortened) + len(sent) < 400:
                shortened += sent + ". "
            else:
                break
        cot = shortened.strip()
    
    return cot.strip()


def format_prompt_checker(question: str, solver_response: str, dataset_name: str = "") -> str:
    """Format prompt for checker agent."""
    solver_answer = extract_answer(solver_response)
    
    if not solver_answer:
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solver_response)
        if boxed_match:
            solver_answer = boxed_match.group(1).strip()
        else:
            solver_answer = "No answer found"
    
    solver_cot = extract_solver_cot(solver_response)
    
    return f"""Problem: {question}

Solution approach: {solver_cot}
Final answer: {solver_answer}

Review checklist:
1. Does the solution use the right numbers from the problem?
2. Are the calculations logical?
3. Does the final answer make sense?

If you see an error or something wrong, respond: INCORRECT
If everything seems correct, respond: CORRECT
If unsure, respond: UNCLEAR

One word only:"""


# ============================================================================
# CHECKER RESPONSE PARSING
# ============================================================================

def parse_checker_verdict(response: str) -> str:
    """Parse checker verdict from response."""
    upper = response.upper()
    
    match = re.search(r'VERDICT\s*:\s*(CORRECT|INCORRECT|UNCLEAR)', upper)
    if match:
        return match.group(1).upper()
    
    standalone_match = re.search(r'(^|\n|VERDICT[:\s]+)(CORRECT|INCORRECT|UNCLEAR)(\s|$|\n)', upper)
    if standalone_match:
        return standalone_match.group(2).upper()
    
    if re.search(r'\bCORRE+CT\b', upper):
        return "CORRECT"
    if re.search(r'\bCOR+ECT\b', upper):
        return "CORRECT"
    if re.search(r'\bINCORRE+CT\b', upper):
        return "INCORRECT"
    if re.search(r'\bINCOR+ECT\b', upper):
        return "INCORRECT"
    if re.search(r'\bUNCLE+AR\b', upper):
        return "UNCLEAR"
    
    head = upper[:200]
    
    if "INCORRECT" in head and "CORRECT" not in head:
        return "INCORRECT"
    if "CORRECT" in head and "INCORRECT" not in head:
        return "CORRECT"
    if "UNCLEAR" in head:
        return "UNCLEAR"
    
    return "UNCLEAR"


def parse_checker_tip(response: str) -> str:
    """Extract tip/feedback from checker response."""
    verdict_match = re.search(r'VERDICT\s*:\s*(?:CORRECT|INCORRECT|UNCLEAR)', response, re.IGNORECASE)
    if verdict_match:
        tip_end = verdict_match.start()
        tip = response[:tip_end].strip()
        
        tip = re.sub(r'^(tip|feedback|suggestion|hint|note|reasoning)[:\s]*', '', tip, flags=re.IGNORECASE)
        tip = re.sub(r'\n\s*\n', ' ', tip)
        tip = tip.strip()
        
        if len(tip) > 300:
            sentences = re.split(r'[.!?]\s+', tip)
            if len(sentences) > 1:
                tip = '. '.join(sentences[:2]) + '.'
            else:
                tip = tip[:300]
        
        return tip
    
    return ""


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_solver_checker_workflow(
    question: str,
    ground_truth: str,
    solver_model,
    solver_tokenizer,
    checker_model,
    checker_tokenizer,
    max_iterations: int = 5,
    detailed: bool = False,
    dataset_name: str = ""
) -> Dict:
    """
    Run solver-checker iterative workflow (original archive version).
    
    Args:
        question: Math problem question
        ground_truth: Ground truth answer
        solver_model: Solver model
        solver_tokenizer: Solver tokenizer
        checker_model: Checker model (can be same as solver)
        checker_tokenizer: Checker tokenizer
        max_iterations: Maximum number of iterations
        detailed: Whether to show detailed output
        dataset_name: Dataset name
    
    Returns:
        Dictionary with workflow results
    """
    # Storage for all iterations
    solver_responses = []
    solver_answers = []
    checker_responses = []
    checker_verdicts = []
    
    checker_feedback = ""  # Initially empty
    predicted_answer = None
    
    for iteration in range(max_iterations):
        iteration_num = iteration + 1
        
        if detailed:
            print("\n" + "=" * 40)
            print(f"[Iteration {iteration_num}/{max_iterations}]")
            print("=" * 40)
        
        # Step 1: Solver generates response
        solver_prompt = format_prompt_solver(question, checker_feedback, dataset_name)
        
        if detailed:
            print("\n" + "-" * 40)
            print(f"[Iteration {iteration_num}] Generating Solver response...")
            if checker_feedback:
                print(f"Checker feedback: {checker_feedback}")
            print("-" * 40)
        
        try:
            solver_response = generate_response(solver_model, solver_tokenizer, solver_prompt, "solver", detailed)
        except Exception as e:
            solver_response = f"Error: {e}"
        
        # Fallback if solver produces empty response
        if not solver_response.strip():
            if detailed:
                print("Solver response empty, falling back to standard prompt.")
            fallback_prompt = format_prompt_solver(question, "", dataset_name)
            try:
                solver_response = generate_response(solver_model, solver_tokenizer, fallback_prompt, "solver", detailed)
            except Exception as e:
                solver_response = f"Error: {e}"
        
        solver_answer = extract_answer(solver_response)
        solver_responses.append(solver_response)
        if solver_answer:
            solver_answers.append(solver_answer)
        
        # Step 2: Checker evaluates solver response
        checker_prompt = format_prompt_checker(question, solver_response, dataset_name)
        
        if detailed:
            print("\n" + "-" * 40)
            print(f"[Iteration {iteration_num}] Generating Checker response...")
            print("-" * 40)
        
        try:
            checker_response = generate_response(checker_model, checker_tokenizer, checker_prompt, "checker", detailed)
        except Exception as e:
            checker_response = f"Error: {e}"
        
        # If checker responded with nothing, retry with simpler prompt
        if not checker_response.strip():
            if detailed:
                print("Checker produced empty response; retrying with simpler prompt...")
            simple_checker_prompt = f"""Q: {question}\nA: {solver_answer}\n\nVERDICT:"""
            try:
                checker_response = generate_response(checker_model, checker_tokenizer, simple_checker_prompt, "checker", detailed)
            except Exception as e:
                checker_response = "VERDICT: UNCLEAR"
            
            if not checker_response.strip():
                if detailed:
                    print("Checker still empty after retry; defaulting to VERDICT: UNCLEAR")
                checker_response = "VERDICT: UNCLEAR"
        
        checker_responses.append(checker_response)
        checker_verdict = parse_checker_verdict(checker_response)
        
        # Ensure valid verdict
        if checker_verdict not in ["CORRECT", "INCORRECT", "UNCLEAR"]:
            checker_verdict = "UNCLEAR"
        
        checker_verdicts.append(checker_verdict)
        
        if detailed:
            print(f"\nChecker Verdict: {checker_verdict}")
        
        # Condition 1: If checker says CORRECT, use this answer and break
        if checker_verdict == "CORRECT":
            if solver_answer:
                predicted_answer = solver_answer
                if detailed:
                    print(f"\n✓ Checker confirmed solution is CORRECT. Using answer: {predicted_answer}")
                break
            else:
                if detailed:
                    print("Warning: Checker says CORRECT but no answer extracted. Continuing...")
        
        # Condition 2: If checker says INCORRECT, extract tip and loop back
        elif checker_verdict == "INCORRECT":
            checker_feedback = parse_checker_tip(checker_response)
            
            if checker_feedback:
                if detailed:
                    print(f"\n✗ Checker says INCORRECT. Tip: {checker_feedback}")
            else:
                if detailed:
                    print("\n✗ Checker says INCORRECT. No tip provided.")
                checker_feedback = "The previous solution was incorrect. Please reconsider the problem."
            
            if iteration_num < max_iterations:
                if detailed:
                    print(f"Looping back to solver with feedback (iteration {iteration_num + 1}/{max_iterations})...")
                continue
            else:
                if detailed:
                    print("\nReached maximum iterations. Proceeding to majority vote...")
                break
        
        # Condition 3: If UNCLEAR, continue to next iteration or break if at max
        else:  # UNCLEAR
            if detailed:
                print(f"\n? Checker verdict is UNCLEAR.")
            if iteration_num < max_iterations:
                if detailed:
                    print(f"Continuing to next iteration...")
                checker_feedback = "The previous solution was unclear. Please try again with clearer reasoning."
                continue
            else:
                if detailed:
                    print("Reached maximum iterations. Proceeding to majority vote...")
                break
    
    # If we've exhausted all iterations without CORRECT verdict, use majority vote
    if predicted_answer is None:
        if detailed:
            print("\n" + "=" * 40)
            print("Using majority vote from all iterations")
            print("=" * 40)
        
        if solver_answers:
            answer_counts = Counter(solver_answers)
            most_common = answer_counts.most_common(1)[0]
            predicted_answer = most_common[0]
            if detailed:
                print(f"Majority vote: {predicted_answer} (appeared {most_common[1]} times out of {len(solver_answers)} attempts)")
        else:
            # Fallback: use last answer if no answers extracted
            predicted_answer = solver_answers[-1] if solver_answers else None
            if detailed:
                print(f"No answers extracted. Using last attempt.")
    
    # Check final correctness
    final_correct = False
    if predicted_answer:
        final_correct = check_answer(predicted_answer, ground_truth)
    
    # Determine case type
    first_answer = solver_answers[0] if solver_answers else None
    first_correct = check_answer(first_answer, ground_truth) if first_answer else False
    
    case_type = None
    if first_correct and final_correct and len(solver_responses) == 1:
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
        "total_iterations": len(solver_responses),
        "solver_responses": solver_responses,
        "solver_answers": solver_answers,
        "checker_responses": checker_responses,
        "checker_verdicts": checker_verdicts,
        "first_answer": first_answer,
        "first_correct": first_correct,
        "case_type": case_type
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Solver-Checker Base (Archive Version) - Quick Test")
    print("=" * 80)
    
    # Simple test question
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_ground_truth = "18"
    
    # Try to load model
    try:
        base_path = Path(__file__).parent.parent
        model_name = "Qwen2.5-Math-1.5B"
        
        print(f"\nLoading model: {model_name}")
        model, tokenizer = load_model(model_name, base_path)
        
        print(f"\nRunning solver-checker workflow...")
        print(f"Question: {test_question[:100]}...")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_solver_checker_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            solver_model=model,
            solver_tokenizer=tokenizer,
            checker_model=model,
            checker_tokenizer=tokenizer,
            max_iterations=3,
            detailed=True,
            dataset_name="gsm8k"
        )
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Predicted Answer: {result['predicted_answer']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Correct: {result['final_correct']}")
        print(f"Case Type: {result['case_type']}")
        print(f"Total Iterations: {result['total_iterations']}")
        print(f"First Answer: {result['first_answer']} (Correct: {result['first_correct']})")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during test: {e}")
        print("Note: This test requires model files in pretrained_models/")

