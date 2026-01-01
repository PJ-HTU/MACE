"""
SI Operator (Specialization Improvement Operator)

Corresponds to the SI operator in the MACE paper – Algorithm 2, line 5.

Core idea:
- Select the instance with the smallest coefficient of variation (CV)
- Generate a specialized heuristic tailored to this instance
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class SIOperator:
    """
    SI (Specialization Improvement) Operator

    Paper Equation (5): i* = arg min CV_i
    where CV_i = std({f_i(h) : h ∈ H}) / mean({f_i(h) : h ∈ H})

    The operator selects the instance with the smallest CV
    (indicating insufficient algorithmic coverage)
    and generates a specialized heuristic for it.
    """

    def __init__(
        self,
        problem: str,
        heuristic_dir: str,
        task_description_file: str,
        output_dir: str,
        test_data_dir: str,
        api_key: str,
        model: str
    ):
        """
        Args:
            problem: Problem type (tsp, jssp, cvrp, psp)
            heuristic_dir: Directory containing heuristic code
            task_description_file: Path to the task description file
            output_dir: Output directory
            test_data_dir: Test data directory (used to read instance files)
            api_key: LLM API key
            model: LLM model name
        """
        self.problem = problem
        self.heuristic_dir = heuristic_dir
        self.task_description_file = task_description_file
        self.output_dir = output_dir
        self.test_data_dir = test_data_dir
        self.api_key = api_key
        self.model = model

        self.task_description = self._load_task_description()

    def _load_task_description(self) -> str:
        """Load the task description file."""
        if self.task_description_file and os.path.exists(self.task_description_file):
            with open(self.task_description_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def generate(self, population: List[Dict]) -> Tuple[str, str]:
        """
        Execute the full SI operator workflow.

        Args:
            population: Current population
                Format: [{'name': ..., 'performance_vector': [...], 'avg_performance': ...}, ...]

        Returns:
            (file_path, code): Path to the generated heuristic file and the code itself
        """
        print("\n" + "=" * 80)
        print("[SI Operator] Specialization Improvement started")
        print("=" * 80)

        instance_idx, cv_value, instance_info = self._select_underserved_instance(population)
        print(f"Selected target instance: #{instance_idx}")
        print(f"Coefficient of variation (CV): {cv_value:.4f} (minimum)")
        print(f"Instance information: {instance_info}")

        instance_analysis = self._analyze_instance_performance(population, instance_idx)
        si_prompt = self._create_si_prompt(instance_idx, instance_info, instance_analysis)
        llm_response = self._call_llm(si_prompt)
        extracted_code = self._extract_code_from_response(llm_response)

        if not extracted_code:
            print("Failed to extract code from LLM response")
            print("\nFull response:")
            print(llm_response)
            return None, None

        try:
            file_path = self._save_generated_heuristic(extracted_code)

            response_file = file_path.replace('.py', '_full_response.txt')
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt used:\n{'-'*80}\n{si_prompt}\n\n")
                f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")

            print(f"Code saved to: {file_path}")

        except Exception as e:
            print(f"Saving failed: {str(e)}")
            return None, extracted_code

        print("\n" + "=" * 80)
        print("[SI Operator] Workflow completed")
        print("=" * 80)
        print(f"Generated heuristic file: {file_path}")
        print(f"Full response record: {response_file}")
        print(f"Specialization target: Instance #{instance_idx} (CV={cv_value:.4f})")
        print("=" * 80 + "\n")

        return file_path, extracted_code

    def _select_underserved_instance(self, population: List[Dict]) -> Tuple[int, float, str]:
        """
        Select the instance with the minimum coefficient of variation.

        CV_i = std({f_i(h) : h ∈ H}) / mean({f_i(h) : h ∈ H})

        Args:
            population: Current population

        Returns:
            (instance_idx, cv_value, instance_info)
        """
        if not population:
            raise ValueError("Population is empty; cannot select instance")

        m = len(population[0]['performance_vector'])
        cv_scores = {}

        for instance_idx in range(m):
            performances = [h['performance_vector'][instance_idx] for h in population]

            mean_perf = np.mean(performances)
            std_perf = np.std(performances)

            if mean_perf > 0:
                cv = std_perf / mean_perf
            else:
                cv = float('inf')

            cv_scores[instance_idx] = cv

        target_instance_idx = min(cv_scores, key=cv_scores.get)
        target_cv = cv_scores[target_instance_idx]
        instance_info = self._get_instance_info(target_instance_idx)

        return target_instance_idx, target_cv, instance_info

    def _get_instance_info(self, instance_idx: int) -> str:
        """
        Retrieve descriptive information for an instance.

        Args:
            instance_idx: Instance index

        Returns:
            Instance description
        """
        try:
            if os.path.exists(self.test_data_dir):
                files = sorted([f for f in os.listdir(self.test_data_dir) if not f.startswith('.')])
                if instance_idx < len(files):
                    return f"File: {files[instance_idx]}"
        except Exception as e:
            print(f"Warning: failed to read instance information: {e}")

        return f"Instance index: {instance_idx}"

    def _analyze_instance_performance(self, population: List[Dict], instance_idx: int) -> str:
        """
        Analyze heuristic performance on the target instance.

        Args:
            population: Current population
            instance_idx: Target instance index

        Returns:
            Performance analysis text
        """
        performances = []

        for h in population:
            performances.append({
                'name': os.path.basename(h['name']),
                'performance': h['performance_vector'][instance_idx]
            })

        performances.sort(key=lambda x: x['performance'])

        analysis = f"Performance analysis on target instance #{instance_idx}:\n"
        analysis += f"  Best performance: {performances[0]['performance']:.2f} ({performances[0]['name']})\n"
        analysis += f"  Worst performance: {performances[-1]['performance']:.2f} ({performances[-1]['name']})\n"
        analysis += f"  Mean performance: {np.mean([p['performance'] for p in performances]):.2f}\n"
        analysis += f"  Standard deviation: {np.std([p['performance'] for p in performances]):.2f}\n"

        return analysis

    def _create_si_prompt(
        self,
        instance_idx: int,
        instance_info: str,
        instance_analysis: str
    ) -> str:
        """Create the SI prompt for specialized heuristic generation."""
        prompt_template = PromptTemplate(
            input_variables=[
                "task_description",
                "instance_idx",
                "instance_info",
                "instance_analysis"
            ],
            template="""# Problem Context

{task_description}

# Task: Design a Specialized Heuristic for a Specific Instance

You are an expert in designing heuristic algorithms for combinatorial optimization problems.

## Situation

Current analysis shows that **Instance #{instance_idx}** has **low algorithmic differentiation** (low coefficient of variation), meaning:
- All existing heuristics perform similarly on this instance
- This instance is under-served by the current portfolio
- There is potential for a specialized heuristic to significantly improve performance

## Target Instance Information

{instance_info}

## Current Performance on This Instance

{instance_analysis}

## Your Task

Design a NEW specialized heuristic that is explicitly optimized for the structural characteristics of this specific instance type.

IMPORTANT CODE FORMAT REQUIREMENTS:

1. The function name must follow the pattern: <strategy_name>_<random_4_chars>
2. The code must be complete and executable
3. The heuristic should be highly specialized for this instance type

Response format:

***python_code:
[Your complete Python code here]
***
"""
        )

        return prompt_template.format(
            task_description=self.task_description,
            instance_idx=instance_idx,
            instance_info=instance_info,
            instance_analysis=instance_analysis
        )

    def _call_llm(self, prompt: str) -> str:
        """Invoke the LLM to generate heuristic code."""
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert algorithm designer for combinatorial optimization problems."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"API call failed: {str(e)}")
            raise

    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from the LLM response."""
        from src.run_hyper_heuristic.helper_function import extract_code_from_response
        return extract_code_from_response(response)

    def _save_generated_heuristic(self, code: str) -> str:
        """Save the generated heuristic code."""
        from src.run_hyper_heuristic.helper_function import save_generated_heuristic
        return save_generated_heuristic(code, output_dir=self.output_dir)
