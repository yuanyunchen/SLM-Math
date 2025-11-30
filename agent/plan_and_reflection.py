"""
Plan-and-Reflection Multi-Agent Workflow (Simplified Version)
Optimized for math-focused models like Qwen2.5-Math

SIMPLIFIED ARCHITECTURE:
- Phase 1: Solve directly with step-by-step reasoning
- Phase 2: Verify by re-solving with different approach hint  
- Phase 3: If disagreement, solve once more and take majority

This approach works better with math-specialized models that don't handle
abstract planning/reflection prompts well.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Tuple
import re
import torch


class PlanAndReflectionWorkflow:
    """Simplified Plan-and-Reflection Agent Workflow for Math Models"""
    
    def __init__(
        self,
        model,
        tokenizer,
        max_iterations: int = 3,
        max_subproblems: int = 5,
        detailed: bool = False,
        dataset_name: str = ""
    ):
        """
        Initialize workflow
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            max_iterations: Maximum solve-verify iterations
            max_subproblems: Not used in simplified version
            detailed: Verbose output
            dataset_name: Dataset name for prompting
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.max_subproblems = max_subproblems
        self.detailed = detailed
        self.dataset_name = dataset_name
        
        # Import utilities
        from utils.prompt_utils import extract_answer, check_answer
        from models.inference import generate_response
        from agent.unified_config import FIRST_ROUND_SOLVER_CONFIG, SUBSEQUENT_ROUND_CONFIG
        self.extract_answer = extract_answer
        self.check_answer = check_answer
        self.generate_response = generate_response
        self.first_round_config = FIRST_ROUND_SOLVER_CONFIG
        self.subsequent_round_config = SUBSEQUENT_ROUND_CONFIG
        
        # Track if first answer has been generated (for unified config)
        self._first_answer_generated = False
    
    def run(self, question: str, ground_truth: str) -> Dict:
        """
        Execute the simplified solve-verify workflow
        
        Args:
            question: Problem to solve
            ground_truth: Ground truth answer
        
        Returns:
            Dict with results
        """
        # Reset first answer flag for each new problem
        self._first_answer_generated = False
        
        if self.detailed:
            print(f"\n{'='*80}")
            print(f"PLAN-AND-REFLECTION WORKFLOW (SIMPLIFIED)")
            print(f"{'='*80}")
            print(f"Question: {question[:100]}...")
            print(f"Ground Truth: {ground_truth}")
            print(f"{'='*80}\n")
        
        iterations = []
        answers_collected = []
        
        # Phase 1: Initial solve with standard prompt
        if self.detailed:
            print(f"\n{'='*80}")
            print(f"ITERATION 1/{self.max_iterations}: INITIAL SOLVE")
            print(f"{'='*80}\n")
        
        result1 = self._solve_standard(question)
        answer1 = result1['answer']
        is_correct1 = self.check_answer(answer1, ground_truth) if answer1 else False
        
        iterations.append({
            'iteration': 1,
            'phase': 'initial_solve',
            'prompt': result1['prompt'],
            'response': result1['response'],
            'answer': answer1,
            'is_correct': is_correct1,
            'plan': {'prompt': '', 'response': 'Direct solve', 'reasoning': 'Step-by-step solution', 'subproblems': [], 'num_subproblems': 0},
            'execute': {'solutions': [{'subproblem': question, 'response': result1['response'], 'answer': answer1}], 'num_solved': 1},
            'reflect': {'prompt': '', 'response': '', 'verdict': 'PENDING', 'reasoning': '', 'suggestions': ''},
            'integrate': {'prompt': result1['prompt'], 'response': result1['response'], 'final_answer': answer1, 'is_correct': is_correct1}
        })
        answers_collected.append(answer1)
        
        if self.detailed:
            print(f"\n[Initial Answer]: {answer1}")
            print(f"[Correct]: {is_correct1}")
        
        # If max_iterations is 1, return now
        if self.max_iterations <= 1:
            return self._compile_results(question, ground_truth, iterations, answers_collected)
        
        # Phase 2: Verify by re-solving with verification prompt
        if self.detailed:
            print(f"\n{'='*80}")
            print(f"ITERATION 2/{self.max_iterations}: VERIFICATION SOLVE")
            print(f"{'='*80}\n")
        
        result2 = self._solve_with_verification(question, answer1)
        answer2 = result2['answer']
        is_correct2 = self.check_answer(answer2, ground_truth) if answer2 else False
        
        iterations.append({
            'iteration': 2,
            'phase': 'verification_solve',
            'prompt': result2['prompt'],
            'response': result2['response'],
            'answer': answer2,
            'is_correct': is_correct2,
            'plan': {'prompt': '', 'response': 'Verification solve', 'reasoning': 'Re-solve to verify', 'subproblems': [], 'num_subproblems': 0},
            'execute': {'solutions': [{'subproblem': question, 'response': result2['response'], 'answer': answer2}], 'num_solved': 1},
            'reflect': {'prompt': '', 'response': '', 'verdict': 'ACCEPT' if answer1 == answer2 else 'REJECT', 'reasoning': f'First answer: {answer1}, Second answer: {answer2}', 'suggestions': ''},
            'integrate': {'prompt': result2['prompt'], 'response': result2['response'], 'final_answer': answer2, 'is_correct': is_correct2}
        })
        answers_collected.append(answer2)
        
        if self.detailed:
            print(f"\n[Verification Answer]: {answer2}")
            print(f"[Correct]: {is_correct2}")
            print(f"[Agreement]: {'YES' if answer1 == answer2 else 'NO'}")
        
        # If answers agree, we're done
        if answer1 == answer2:
            if self.detailed:
                print(f"\n[CONSENSUS REACHED] Both attempts agree on: {answer1}")
            return self._compile_results(question, ground_truth, iterations, answers_collected)
        
        # If max_iterations is 2, return best answer
        if self.max_iterations <= 2:
            return self._compile_results(question, ground_truth, iterations, answers_collected)
        
        # Phase 3: Tiebreaker - solve once more with careful reasoning prompt
        if self.detailed:
            print(f"\n{'='*80}")
            print(f"ITERATION 3/{self.max_iterations}: TIEBREAKER SOLVE")
            print(f"{'='*80}\n")
        
        result3 = self._solve_careful(question, answer1, answer2)
        answer3 = result3['answer']
        is_correct3 = self.check_answer(answer3, ground_truth) if answer3 else False
        
        iterations.append({
            'iteration': 3,
            'phase': 'tiebreaker_solve',
            'prompt': result3['prompt'],
            'response': result3['response'],
            'answer': answer3,
            'is_correct': is_correct3,
            'plan': {'prompt': '', 'response': 'Tiebreaker solve', 'reasoning': 'Careful re-solve', 'subproblems': [], 'num_subproblems': 0},
            'execute': {'solutions': [{'subproblem': question, 'response': result3['response'], 'answer': answer3}], 'num_solved': 1},
            'reflect': {'prompt': '', 'response': '', 'verdict': 'ACCEPT', 'reasoning': 'Final decision', 'suggestions': ''},
            'integrate': {'prompt': result3['prompt'], 'response': result3['response'], 'final_answer': answer3, 'is_correct': is_correct3}
        })
        answers_collected.append(answer3)
        
        if self.detailed:
            print(f"\n[Tiebreaker Answer]: {answer3}")
            print(f"[Correct]: {is_correct3}")
        
        return self._compile_results(question, ground_truth, iterations, answers_collected)
    
    def _solve_standard(self, question: str) -> Dict:
        """Solve with standard step-by-step prompt (same as baseline)"""
        # Use the same prompt as format_prompt_standard for fair comparison
        prompt = f"""{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""
        
        if self.detailed:
            print(f"{'─'*80}")
            print(f"SOLVING (Standard Prompt)")
            print(f"{'─'*80}")
        
        # Use unified config for first answer
        response = self.generate_response(
            self.model,
            self.tokenizer,
            prompt,
            "standard",
            self.detailed,
            temperature=self.first_round_config['temperature'],
            do_sample=self.first_round_config['do_sample'],
            top_p=self.first_round_config['top_p']
        )
        self._first_answer_generated = True
        
        answer = self.extract_answer(response)
        
        return {
            'prompt': prompt,
            'response': response,
            'answer': answer
        }
    
    def _solve_with_verification(self, question: str, previous_answer: str) -> Dict:
        """Solve again to verify the first answer"""
        prompt = f"""{question}
Please solve this carefully step by step, and put your final answer within \\boxed{{}}."""
        
        if self.detailed:
            print(f"{'─'*80}")
            print(f"SOLVING (Verification Prompt)")
            print(f"{'─'*80}")
        
        # Use subsequent round config (with some randomness for diversity)
        response = self.generate_response(
            self.model,
            self.tokenizer,
            prompt,
            "standard",
            self.detailed,
            temperature=self.subsequent_round_config['temperature'],
            do_sample=self.subsequent_round_config['do_sample'],
            top_p=self.subsequent_round_config['top_p']
        )
        
        answer = self.extract_answer(response)
        
        return {
            'prompt': prompt,
            'response': response,
            'answer': answer
        }
    
    def _solve_careful(self, question: str, answer1: str, answer2: str) -> Dict:
        """Solve once more with extra care when previous answers disagree"""
        prompt = f"""{question}
Let me solve this step by step very carefully, and put my final answer within \\boxed{{}}."""
        
        if self.detailed:
            print(f"{'─'*80}")
            print(f"SOLVING (Careful Prompt - Tiebreaker)")
            print(f"{'─'*80}")
        
        # Use lower temperature for more careful reasoning
        response = self.generate_response(
            self.model,
            self.tokenizer,
            prompt,
            "standard",
            self.detailed,
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )
        
        answer = self.extract_answer(response)
        
        return {
            'prompt': prompt,
            'response': response,
            'answer': answer
        }
    
    def _compile_results(
        self, 
        question: str, 
        ground_truth: str, 
        iterations: List[Dict],
        answers_collected: List[str]
    ) -> Dict:
        """Compile final results using smart answer selection
        
        Strategy (prioritized):
        1. If two or more answers agree, use the majority answer
        2. If first answer appears again later (even once), use first (confirmed)
        3. Otherwise, use FIRST answer (trust initial deterministic reasoning)
        
        This protects against degradation while still allowing improvement
        when later attempts reach consensus.
        """
        from collections import Counter
        valid_answers = [a for a in answers_collected if a]
        answer_counts = Counter(valid_answers)
        
        if not answer_counts:
            final_answer = None
        elif len(answer_counts) == 1:
            # All answers agree
            final_answer = valid_answers[0]
        else:
            # Multiple different answers
            most_common = answer_counts.most_common(1)[0]
            if most_common[1] >= 2:
                # True majority (2+ agree) - use it
                final_answer = most_common[0]
            else:
                # No majority (all different)
                # Use FIRST answer (trust deterministic initial reasoning)
                final_answer = valid_answers[0] if valid_answers else None
        
        final_correct = self.check_answer(final_answer, ground_truth) if final_answer else False
        
        # First answer info
        first_answer = iterations[0]['answer'] if iterations else None
        first_correct = iterations[0]['is_correct'] if iterations else False
        
        # Determine case type
        case_type = self._determine_case_type(first_correct, final_correct, len(iterations))
        
        if self.detailed:
            print(f"\n{'='*80}")
            print(f"FINAL RESULT")
            print(f"{'='*80}")
            print(f"Answers collected: {answers_collected}")
            print(f"Final Answer (majority): {final_answer}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Correct: {final_correct}")
            print(f"Case Type: {case_type}")
            print(f"{'='*80}\n")
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'predicted_answer': final_answer,
            'final_correct': final_correct,
            'first_answer': first_answer,
            'first_correct': first_correct,
            'total_iterations': len(iterations),
            'iterations': iterations,
            'case_type': case_type,
            'final_verdict': 'CORRECT' if final_correct else 'INCORRECT',
            'answers_collected': answers_collected,
            'answer_counts': dict(answer_counts)
        }
    
    def _determine_case_type(self, first_correct: bool, final_correct: bool, num_iterations: int) -> str:
        """Determine the type of case"""
        if first_correct and final_correct:
            return "FIRST_TRY_SUCCESS"
        elif not first_correct and final_correct:
            return "IMPROVED"
        elif first_correct and not final_correct:
            return "DEGRADED"
        elif not first_correct and not final_correct:
            return "FAILED"
        else:
            return "OTHER"


def run_plan_and_reflection_workflow(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    max_iterations: int = 3,
    max_subproblems: int = 5,
    detailed: bool = False,
    dataset_name: str = ""
) -> Dict:
    """
    Run simplified Plan-and-Reflection workflow
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        model: Language model
        tokenizer: Tokenizer
        max_iterations: Maximum solve-verify cycles (1-3)
        max_subproblems: Not used in simplified version
        detailed: Verbose output
        dataset_name: Dataset name
    
    Returns:
        Dict with workflow results
    """
    workflow = PlanAndReflectionWorkflow(
        model=model,
        tokenizer=tokenizer,
        max_iterations=max_iterations,
        max_subproblems=max_subproblems,
        detailed=detailed,
        dataset_name=dataset_name
    )
    
    result = workflow.run(question, ground_truth)
    
    return result


# For testing
if __name__ == "__main__":
    print("Plan-and-Reflection Workflow (Simplified) - Quick Test")
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
        
        print(f"\nRunning simplified workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_plan_and_reflection_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            model=model,
            tokenizer=tokenizer,
            max_iterations=3,
            max_subproblems=3,
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
