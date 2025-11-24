"""
Plan-and-Reflection Multi-Agent Workflow
规划-执行-反思智能体工作流

ARCHITECTURE:
┌─────────────────────────────────────────────────────┐
│                  Main Controller                    │
│            (Orchestrates the workflow)              │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌────────┐      ┌──────────┐     ┌──────────┐
   │ Planner│ ───> │ Executor │ ──> │Reflector │
   └────────┘      └──────────┘     └──────────┘
        │                │                │
        │                │                │
        └────────────────┴────────────────┘
                         │
                    ┌─────────┐
                    │Integrator│
                    └─────────┘

WORKFLOW:
1. PLAN Phase
   - Analyze problem
   - Decompose into sub-problems
   - Create execution plan

2. EXECUTE Phase
   - Solve each sub-problem
   - Collect partial results
   - Monitor progress

3. REFLECT Phase
   - Evaluate solution quality
   - Identify errors/gaps
   - Decide: accept or re-plan

4. INTEGRATE Phase
   - Combine sub-solutions
   - Generate final answer
   - Verify consistency

适用场景：
- 复杂的多步推理问题
- 需要分解的大问题
- 需要自我修正的场景
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Tuple
import re
import torch


class PlanAndReflectionWorkflow:
    """Plan-and-Reflection Agent Workflow"""
    
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
            max_iterations: Maximum plan-execute-reflect iterations
            max_subproblems: Maximum number of sub-problems
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
        from models.inference import generate_response as _generate_response
        self.extract_answer = extract_answer
        self.check_answer = check_answer
        self._generate_response_fn = _generate_response
        
        # Create a wrapper to handle both inference engine and raw model
        if hasattr(model, 'generate_single'):
            # Using inference engine
            self.generate_response = lambda m, t, prompt, mode, detailed: m.generate_single(
                prompt,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.2,
                detailed=detailed
            )
        else:
            # Using standard model
            self.generate_response = _generate_response
    
    def run(self, question: str, ground_truth: str) -> Dict:
        """
        Execute the complete workflow
        
        Args:
            question: Problem to solve
            ground_truth: Ground truth answer
        
        Returns:
            Dict with results
        """
        if self.detailed:
            print(f"\n{'='*80}")
            print(f"PLAN-AND-REFLECTION WORKFLOW")
            print(f"{'='*80}")
            print(f"Question: {question[:100]}...")
            print(f"Ground Truth: {ground_truth}")
            print(f"{'='*80}\n")
        
        iterations = []
        final_answer = None
        final_correct = False
        
        for iteration in range(1, self.max_iterations + 1):
            if self.detailed:
                print(f"\n{'='*80}")
                print(f"ITERATION {iteration}/{self.max_iterations}")
                print(f"{'='*80}\n")
            
            # Phase 1: PLAN
            plan_result = self._plan_phase(question, iterations)
            
            # Phase 2: EXECUTE
            execute_result = self._execute_phase(question, plan_result)
            
            # Phase 3: REFLECT
            reflect_result = self._reflect_phase(
                question, 
                plan_result, 
                execute_result,
                ground_truth
            )
            
            # Phase 4: INTEGRATE
            integrate_result = self._integrate_phase(
                question,
                plan_result,
                execute_result,
                reflect_result
            )
            
            # Store iteration data
            iteration_data = {
                'iteration': iteration,
                'plan': plan_result,
                'execute': execute_result,
                'reflect': reflect_result,
                'integrate': integrate_result,
                'answer': integrate_result['final_answer'],
                'is_correct': integrate_result['is_correct']
            }
            iterations.append(iteration_data)
            
            # Check if we should continue
            if reflect_result['verdict'] == 'ACCEPT':
                if self.detailed:
                    print(f"\n✓ Solution accepted at iteration {iteration}")
                final_answer = integrate_result['final_answer']
                final_correct = integrate_result['is_correct']
                break
            elif iteration >= self.max_iterations:
                if self.detailed:
                    print(f"\n⚠ Reached max iterations")
                final_answer = integrate_result['final_answer']
                final_correct = integrate_result['is_correct']
                break
            else:
                if self.detailed:
                    print(f"\n↻ Re-planning needed: {reflect_result['reason']}")
        
        # Compile final results
        first_answer = iterations[0]['answer'] if iterations else None
        first_correct = iterations[0]['is_correct'] if iterations else False
        
        final_verdict = "CORRECT" if final_correct else "INCORRECT"
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'predicted_answer': final_answer,
            'final_correct': final_correct,
            'final_verdict': final_verdict,
            'first_answer': first_answer,
            'first_correct': first_correct,
            'total_iterations': len(iterations),
            'iterations': iterations,
            'case_type': self._determine_case_type(first_correct, final_correct, len(iterations))
        }
    
    def _plan_phase(self, question: str, previous_iterations: List[Dict]) -> Dict:
        """
        Phase 1: Planning
        Analyze and decompose the problem
        """
        if self.detailed:
            print(f"{'─'*80}")
            print(f"PHASE 1: PLANNING")
            print(f"{'─'*80}")
        
        # Build planner prompt
        prompt = self._build_planner_prompt(question, previous_iterations)
        
        # Generate plan
        plan_response = self.generate_response(
            self.model,
            self.tokenizer,
            prompt,
            "standard",
            self.detailed
        )
        
        # Parse plan
        subproblems = self._parse_subproblems(plan_response)
        reasoning = self._extract_planning_reasoning(plan_response)
        
        if self.detailed:
            print(f"\n[Planning Reasoning]")
            print(f"{reasoning[:200]}...")
            print(f"\n[Sub-problems identified: {len(subproblems)}]")
            for i, sp in enumerate(subproblems, 1):
                print(f"  {i}. {sp[:80]}...")
        
        return {
            'prompt': prompt,
            'response': plan_response,
            'reasoning': reasoning,
            'subproblems': subproblems,
            'num_subproblems': len(subproblems)
        }
    
    def _execute_phase(self, question: str, plan_result: Dict) -> Dict:
        """
        Phase 2: Execution
        Solve each sub-problem
        """
        if self.detailed:
            print(f"\n{'─'*80}")
            print(f"PHASE 2: EXECUTION")
            print(f"{'─'*80}")
        
        subproblems = plan_result['subproblems']
        solutions = []
        
        for i, subproblem in enumerate(subproblems, 1):
            if self.detailed:
                print(f"\n[Sub-problem {i}/{len(subproblems)}]")
                print(f"  {subproblem[:100]}...")
            
            # Build executor prompt
            prompt = self._build_executor_prompt(question, subproblem, i)
            
            # Generate solution
            solution_response = self.generate_response(
                self.model,
                self.tokenizer,
                prompt,
                "standard",
                self.detailed
            )
            
            # Extract answer from solution
            solution_answer = self.extract_answer(solution_response)
            
            if self.detailed:
                print(f"  → Solution: {solution_answer}")
            
            solutions.append({
                'subproblem': subproblem,
                'prompt': prompt,
                'response': solution_response,
                'answer': solution_answer
            })
        
        return {
            'solutions': solutions,
            'num_solved': len(solutions)
        }
    
    def _reflect_phase(
        self,
        question: str,
        plan_result: Dict,
        execute_result: Dict,
        ground_truth: str
    ) -> Dict:
        """
        Phase 3: Reflection
        Evaluate solution quality and decide next action
        """
        if self.detailed:
            print(f"\n{'─'*80}")
            print(f"PHASE 3: REFLECTION")
            print(f"{'─'*80}")
        
        # Build reflector prompt
        prompt = self._build_reflector_prompt(
            question,
            plan_result,
            execute_result
        )
        
        # Generate reflection
        reflection_response = self.generate_response(
            self.model,
            self.tokenizer,
            prompt,
            "standard",
            self.detailed
        )
        
        # Parse reflection
        verdict = self._parse_reflection_verdict(reflection_response)
        reasoning = self._extract_reflection_reasoning(reflection_response)
        suggestions = self._extract_reflection_suggestions(reflection_response)
        
        if self.detailed:
            print(f"\n[Reflection Verdict]: {verdict}")
            print(f"[Reasoning]: {reasoning[:150]}...")
            if suggestions:
                print(f"[Suggestions]: {suggestions[:150]}...")
        
        return {
            'prompt': prompt,
            'response': reflection_response,
            'verdict': verdict,
            'reasoning': reasoning,
            'suggestions': suggestions
        }
    
    def _integrate_phase(
        self,
        question: str,
        plan_result: Dict,
        execute_result: Dict,
        reflect_result: Dict
    ) -> Dict:
        """
        Phase 4: Integration
        Combine sub-solutions into final answer
        """
        if self.detailed:
            print(f"\n{'─'*80}")
            print(f"PHASE 4: INTEGRATION")
            print(f"{'─'*80}")
        
        # Build integrator prompt
        prompt = self._build_integrator_prompt(
            question,
            plan_result,
            execute_result
        )
        
        # Generate integrated solution
        integration_response = self.generate_response(
            self.model,
            self.tokenizer,
            prompt,
            "standard",
            self.detailed
        )
        
        # Extract final answer
        final_answer = self.extract_answer(integration_response)
        
        # Check correctness
        is_correct = False
        if final_answer:
            is_correct = self.check_answer(final_answer, question)  # Will be checked against ground truth in caller
        
        if self.detailed:
            print(f"\n[Final Answer]: {final_answer}")
            print(f"[Reasoning]: {integration_response[:200]}...")
        
        return {
            'prompt': prompt,
            'response': integration_response,
            'final_answer': final_answer,
            'is_correct': is_correct
        }
    
    # ========== PROMPT BUILDERS ==========
    
    def _build_planner_prompt(self, question: str, previous_iterations: List[Dict]) -> str:
        """Build prompt for planner agent"""
        prompt = f"""You are a planning agent. Analyze the problem and break it down into sub-problems.

Problem: {question}

Your task:
1. Understand what the problem is asking
2. Identify the key information and unknowns
3. Break the problem into 2-{self.max_subproblems} clear sub-problems
4. Each sub-problem should be independently solvable

"""
        
        if previous_iterations:
            last_iter = previous_iterations[-1]
            prompt += f"""Previous attempt failed. Issues identified:
{last_iter['reflect']['reasoning']}

Suggestions for improvement:
{last_iter['reflect']['suggestions']}

"""
        
        prompt += """Format your response as:

ANALYSIS:
[Your analysis of the problem]

SUB-PROBLEMS:
1. [First sub-problem]
2. [Second sub-problem]
...

PLAN:
[How solving these sub-problems will lead to the final answer]
"""
        
        return prompt
    
    def _build_executor_prompt(self, question: str, subproblem: str, index: int) -> str:
        """Build prompt for executor agent"""
        prompt = f"""You are solving a sub-problem as part of a larger problem.

Main Problem: {question}

Sub-problem {index}: {subproblem}

Solve this sub-problem step by step. Show your work and put your final answer in \\boxed{{}}.

Solution:"""
        
        return prompt
    
    def _build_reflector_prompt(
        self,
        question: str,
        plan_result: Dict,
        execute_result: Dict
    ) -> str:
        """Build prompt for reflector agent"""
        # Compile solutions
        solutions_text = ""
        for i, sol in enumerate(execute_result['solutions'], 1):
            solutions_text += f"\nSub-problem {i}: {sol['subproblem']}\n"
            solutions_text += f"Solution: {sol['answer']}\n"
        
        prompt = f"""You are a reflection agent. Evaluate the quality of the problem-solving approach.

Original Problem: {question}

Plan:
{plan_result['reasoning']}

Sub-problems:
{chr(10).join(f"{i}. {sp}" for i, sp in enumerate(plan_result['subproblems'], 1))}

Execution Results:
{solutions_text}

Your task:
1. Check if the sub-problems are appropriate
2. Check if the solutions are correct
3. Identify any gaps or errors
4. Decide: ACCEPT (solution is good) or REJECT (needs revision)

Format your response as:

EVALUATION:
[Your evaluation]

VERDICT: [ACCEPT or REJECT]

REASON:
[Why you accept or reject]

SUGGESTIONS (if REJECT):
[How to improve]
"""
        
        return prompt
    
    def _build_integrator_prompt(
        self,
        question: str,
        plan_result: Dict,
        execute_result: Dict
    ) -> str:
        """Build prompt for integrator agent"""
        # Compile solutions
        solutions_text = ""
        for i, sol in enumerate(execute_result['solutions'], 1):
            solutions_text += f"\nSub-problem {i}: {sol['subproblem']}\n"
            solutions_text += f"Answer: {sol['answer']}\n"
            solutions_text += f"Reasoning: {sol['response'][:200]}...\n"
        
        prompt = f"""You are an integration agent. Combine the sub-solutions to answer the original problem.

Original Problem: {question}

Sub-solutions:
{solutions_text}

Your task:
1. Combine the sub-solutions logically
2. Derive the final answer to the original problem
3. Verify the answer makes sense

Show your reasoning and put the final answer in \\boxed{{}}.

Final Solution:"""
        
        return prompt
    
    # ========== PARSERS ==========
    
    def _parse_subproblems(self, plan_response: str) -> List[str]:
        """Extract sub-problems from plan"""
        subproblems = []
        
        # Look for numbered list
        pattern = r'(?:SUB-PROBLEMS?:|SUBPROBLEMS?:)?\s*\n\s*(\d+[\.\)]\s*.+?)(?=\n\d+[\.\)]|\nPLAN:|\nAPPROACH:|\Z)'
        matches = re.findall(pattern, plan_response, re.IGNORECASE | re.DOTALL)
        
        if matches:
            for match in matches:
                # Clean up
                subproblem = re.sub(r'^\d+[\.\)]\s*', '', match.strip())
                if len(subproblem) > 10:  # Valid subproblem
                    subproblems.append(subproblem)
        
        # Fallback: look for any numbered list
        if not subproblems:
            pattern = r'^\s*\d+[\.\)]\s*(.+)$'
            for line in plan_response.split('\n'):
                match = re.match(pattern, line)
                if match and len(match.group(1)) > 10:
                    subproblems.append(match.group(1).strip())
        
        # Limit to max_subproblems
        return subproblems[:self.max_subproblems]
    
    def _extract_planning_reasoning(self, plan_response: str) -> str:
        """Extract planning reasoning"""
        # Look for ANALYSIS section
        match = re.search(r'ANALYSIS:\s*\n(.+?)(?=\nSUB-PROBLEMS?:|\n\d+[\.\)]|\Z)', 
                         plan_response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: first paragraph
        paragraphs = [p.strip() for p in plan_response.split('\n\n') if len(p.strip()) > 20]
        return paragraphs[0] if paragraphs else plan_response[:200]
    
    def _parse_reflection_verdict(self, reflection_response: str) -> str:
        """Extract verdict from reflection"""
        match = re.search(r'VERDICT:\s*(ACCEPT|REJECT)', reflection_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Heuristic: look for accept/reject keywords
        if 'accept' in reflection_response.lower() and 'reject' not in reflection_response.lower():
            return 'ACCEPT'
        else:
            return 'REJECT'
    
    def _extract_reflection_reasoning(self, reflection_response: str) -> str:
        """Extract reasoning from reflection"""
        match = re.search(r'REASON:\s*\n(.+?)(?=\nSUGGESTIONS?:|\Z)', 
                         reflection_response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback
        match = re.search(r'EVALUATION:\s*\n(.+?)(?=\nVERDICT:|\Z)',
                         reflection_response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return reflection_response[:200]
    
    def _extract_reflection_suggestions(self, reflection_response: str) -> str:
        """Extract suggestions from reflection"""
        match = re.search(r'SUGGESTIONS?.*?:\s*\n(.+?)$',
                         reflection_response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _determine_case_type(self, first_correct: bool, final_correct: bool, num_iterations: int) -> str:
        """Determine the type of case"""
        if first_correct and num_iterations == 1:
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
    Run Plan-and-Reflection workflow
    
    Args:
        question: Problem to solve
        ground_truth: Ground truth answer
        model: Language model
        tokenizer: Tokenizer
        max_iterations: Maximum plan-execute-reflect cycles
        max_subproblems: Maximum sub-problems per plan
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
    
    # Fix is_correct to use ground_truth
    if result['predicted_answer']:
        from utils.prompt_utils import check_answer
        result['final_correct'] = check_answer(result['predicted_answer'], ground_truth)
        
        # Also fix first_answer
        if result['first_answer']:
            result['first_correct'] = check_answer(result['first_answer'], ground_truth)
        
        # Recompute case_type
        result['case_type'] = workflow._determine_case_type(
            result['first_correct'],
            result['final_correct'],
            result['total_iterations']
        )
    
    return result


# For testing
if __name__ == "__main__":
    print("Plan-and-Reflection Workflow - Quick Test")
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
        
        print(f"\nRunning Plan-and-Reflection workflow...")
        print(f"Question: {test_question}")
        print(f"Ground Truth: {test_ground_truth}\n")
        
        result = run_plan_and_reflection_workflow(
            question=test_question,
            ground_truth=test_ground_truth,
            model=model,
            tokenizer=tokenizer,
            max_iterations=2,
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

