"""
Ground-truth free Plan-and-Reflection workflow.

This variant keeps the same multi-agent structure but removes any dependency
on ground-truth answers. Correctness flags are heuristic (based on reflection
verdicts), and the execution phase falls back to solving the full problem when
no sub-problems are parsed.
"""

from typing import Dict, List

from .plan_and_reflection import (
    PlanAndReflectionWorkflow,
    apply_chat_template_if_enabled,
)


class PlanAndReflectionWorkflowNoGT(PlanAndReflectionWorkflow):
    """Plan-and-Reflection workflow that does not rely on ground truth."""

    def run(self, question: str) -> Dict:
        """Execute the workflow without ground-truth supervision."""
        if self.detailed:
            print(f"\n{'='*80}")
            print("PLAN-AND-REFLECTION WORKFLOW (NO GROUND TRUTH)")
            print(f"{'='*80}")
            print(f"Question: {question[:100]}...")
            print(f"{'='*80}\n")

        iterations = []
        final_answer = None
        final_correct = False

        for iteration in range(1, self.max_iterations + 1):
            if self.detailed:
                print(f"\n{'='*80}")
                print(f"ITERATION {iteration}/{self.max_iterations}")
                print(f"{'='*80}\n")

            plan_result = self._plan_phase(question, iterations)
            execute_result = self._execute_phase(question, plan_result)
            reflect_result = self._reflect_phase(
                question,
                plan_result,
                execute_result,
                ground_truth=""
            )
            integrate_result = self._integrate_phase_no_gt(
                question,
                plan_result,
                execute_result,
                reflect_result
            )

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
                    print(f"\n↻ Re-planning needed: {reflect_result['reasoning']}")

        first_answer = iterations[0]['answer'] if iterations else None
        first_correct = iterations[0]['is_correct'] if iterations else False
        final_verdict = "CORRECT" if final_correct else "INCORRECT"

        return {
            'question': question,
            'predicted_answer': final_answer,
            'final_correct': final_correct,
            'final_verdict': final_verdict,
            'first_answer': first_answer,
            'first_correct': first_correct,
            'total_iterations': len(iterations),
            'iterations': iterations,
            'case_type': self._determine_case_type(
                first_correct,
                final_correct,
                len(iterations)
            )
        }

    def _execute_phase(self, question: str, plan_result: Dict) -> Dict:
        """Phase 2: Execution with fallback when no sub-problems are parsed."""
        if self.detailed:
            print(f"\n{'─'*80}")
            print("PHASE 2: EXECUTION")
            print(f"{'─'*80}")

        subproblems = plan_result['subproblems']
        if not subproblems:
            subproblems = [f"Solve the original problem end-to-end: {question}"]

        solutions = []
        for i, subproblem in enumerate(subproblems, 1):
            if self.detailed:
                print(f"\n[Sub-problem {i}/{len(subproblems)}]")
                print(f"  {subproblem[:100]}...")

            prompt = self._build_executor_prompt(question, subproblem, i)
            solution_response = self._generate(prompt)
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

    def _integrate_phase_no_gt(
        self,
        question: str,
        plan_result: Dict,
        execute_result: Dict,
        reflect_result: Dict
    ) -> Dict:
        """
        Integration without ground truth. Uses reflection verdict as a
        self-consistency proxy for correctness.
        """
        if self.detailed:
            print(f"\n{'─'*80}")
            print("PHASE 4: INTEGRATION (NO GT)")
            print(f"{'─'*80}")

        prompt = self._build_integrator_prompt(
            question,
            plan_result,
            execute_result
        )

        integration_response = self._generate(prompt)
        final_answer = self.extract_answer(integration_response)
        is_correct = reflect_result.get('verdict', '').upper() == 'ACCEPT'

        if self.detailed:
            print(f"\n[Final Answer]: {final_answer}")
            print(f"[Reasoning]: {integration_response[:200]}...")

        return {
            'prompt': prompt,
            'response': integration_response,
            'final_answer': final_answer,
            'is_correct': is_correct
        }


def run_plan_and_reflection_workflow_no_gt(
    question: str,
    model,
    tokenizer,
    max_iterations: int = 3,
    max_subproblems: int = 5,
    detailed: bool = False,
    dataset_name: str = "",
    apply_chat_template: bool = False
) -> Dict:
    """
    Convenience runner for the ground-truth free workflow.
    """
    workflow = PlanAndReflectionWorkflowNoGT(
        model=model,
        tokenizer=tokenizer,
        max_iterations=max_iterations,
        max_subproblems=max_subproblems,
        detailed=detailed,
        dataset_name=dataset_name,
        apply_chat_template=apply_chat_template
    )
    return workflow.run(question)







