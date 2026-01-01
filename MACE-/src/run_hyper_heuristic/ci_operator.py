"""
CI Operator (Complementary Improvement Operator)
Corresponds to CI operator in MACE paper - Algorithm 2, line 3

Integrates:
- Prompt generation logic from prompt_cs.py
- Complete workflow from complete_cs_workflow.py
"""

import os
from typing import Dict, List, Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class CIOperator:
    """
    CI (Complementary Improvement) Operator
    
    Paper Equation (3): C(h_a, h_b) = min(W_ab, W_ba) / m
    where W_ab = |{i : f_i(h_a) < f_i(h_b)}|
    
    Selects heuristic pairs with complementary advantages and generates new heuristics
    that integrate their strategies.
    """
    
    def __init__(
        self,
        problem: str,
        heuristic_dir: str,
        task_description_file: str,
        output_dir: str,
        api_key: str,
        model: str
    ):
        """
        Initialize CI Operator.
        
        Args:
            problem: Problem type (tsp, jssp, cvrp, psp)
            heuristic_dir: Heuristic code directory
            task_description_file: Task description file path
            output_dir: Output directory
            api_key: LLM API key
            model: LLM model name
        """
        self.problem = problem
        self.heuristic_dir = heuristic_dir
        self.task_description_file = task_description_file
        self.output_dir = output_dir
        self.api_key = api_key
        self.model = model
        
        self.task_description = self._load_task_description()
    
    def _load_task_description(self) -> str:
        """Load task description file."""
        if self.task_description_file and os.path.exists(self.task_description_file):
            with open(self.task_description_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def generate(self, results_dict: Dict) -> Tuple[str, str]:
        """
        Execute complete CI operator workflow.
        
        Args:
            results_dict: Performance results of all heuristics
                Format: {heuristic_name: [score_1, score_2, ...]}
        
        Returns:
            (file_path, code): Generated heuristic file path and code
        """
        print("\n" + "=" * 80)
        print("🔍 [CI Operator] Complementary Improvement Started")
        print("=" * 80)
        
        # Step 1: Select complementary pair (Paper Equation 3)
        h1_name, h2_name = self._select_complementary_pair(results_dict)
        print(f"✓ Selected complementary pair: {h1_name} and {h2_name}")
        
        # Step 2: Load heuristic code
        h1_code = self._load_heuristic_code(h1_name)
        h2_code = self._load_heuristic_code(h2_name)
        
        # Step 3: Generate CI prompt
        ci_prompt = self._create_ci_prompt(h1_name, h1_code, h2_name, h2_code)
        
        # Step 4: Call LLM to generate new heuristic
        llm_response = self._call_llm(ci_prompt)
        
        # Step 5: Extract code
        extracted_code = self._extract_code_from_response(llm_response)
        
        if not extracted_code:
            print("✗ Failed to extract code from response")
            print("\nFull response:")
            print(llm_response)
            return None, None
        
        # Step 6: Save code
        try:
            file_path = self._save_generated_heuristic(extracted_code)
            
            response_file = file_path.replace('.py', '_full_response.txt')
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt used:\n{'-'*80}\n{ci_prompt}\n\n")
                f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")
            
            print(f"✓ Code saved to: {file_path}")
            
        except Exception as e:
            print(f"✗ Save failed: {str(e)}")
            return None, extracted_code
        
        print("\n" + "=" * 80)
        print("✅ [CI Operator] Workflow Completed!")
        print("=" * 80)
        print(f"📁 Generated heuristic code: {file_path}")
        print(f"📄 Full response log: {response_file}")
        print(f"🔬 Based on complementary pair: {h1_name} + {h2_name}")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    def _select_complementary_pair(self, results_dict: Dict) -> Tuple[str, str]:
        """
        Select complementary pair - Paper Equation (3).
        
        C(h_a, h_b) = min(W_ab, W_ba) / m
        where W_ab = |{i : f_i(h_a) < f_i(h_b)}|
        """
        from src.run_hyper_heuristic.helper_function import select_complementary_pair
        return select_complementary_pair(results_dict)
    
    def _load_heuristic_code(self, heuristic_name: str) -> str:
        """Load heuristic code."""
        from src.run_hyper_heuristic.helper_function import load_heuristic_code
        return load_heuristic_code(self.problem, heuristic_name, self.heuristic_dir)
    
    def _create_ci_prompt(
        self,
        h1_name: str,
        h1_code: str,
        h2_name: str,
        h2_code: str
    ) -> str:
        """Create CI prompt."""
        prompt_template = PromptTemplate(
            input_variables=[
                "task_description",
                "heuristic1_name",
                "heuristic1_code",
                "heuristic2_name",
                "heuristic2_code"
            ],
            template="""# Problem Context

{task_description}

# Task: Design a New Complementary Heuristic Algorithm

You are an expert in designing heuristic algorithms for combinatorial optimization problems.

I have 2 existing heuristic algorithms that show complementary performance across different problem instances:

## Heuristic Algorithm 1: {heuristic1_name}
```python
{heuristic1_code}
```

## Heuristic Algorithm 2: {heuristic2_name}
```python
{heuristic2_code}
```

## Your Task

These two algorithms are effective for solving different instance distributions. Please design a NEW heuristic algorithm that:

1. **Synthesizes complementary strengths** - Integrate effective decision patterns from both algorithms
2. **Is distinct** from the two provided algorithms
3. **Improves instance-level coverage** - Handle cases where both algorithms underperform
4. **Explores new strategies** or combine their strengths in innovative ways

**IMPORTANT CODE FORMAT REQUIREMENTS:**

1. The function name must follow the pattern: `<strategy_name>_<random_4_chars>` (e.g., `hybrid_greedy_a3f7`)
2. Follow the exact code format shown in the examples above
3. Ensure your code is complete and executable

**Response Format:**

The response format is very important. Please respond in this format:

***python_code:
[Your complete Python code here]
***

**CRITICAL:** 
- Ensure there is no other content inside the ***
- Analysis and explanation outside *** are welcome
- The code must be complete and runnable

Please provide your new heuristic algorithm now:"""
        )
        
        final_prompt = prompt_template.format(
            task_description=self.task_description,
            heuristic1_name=h1_name,
            heuristic1_code=h1_code,
            heuristic2_name=h2_name,
            heuristic2_code=h2_code
        )
        
        return final_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate code."""
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
            
            llm_response = response.choices[0].message.content
            return llm_response
            
        except Exception as e:
            print(f"✗ API call failed: {str(e)}")
            raise
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response."""
        from src.run_hyper_heuristic.helper_function import extract_code_from_response
        return extract_code_from_response(response)
    
    def _save_generated_heuristic(self, code: str) -> str:
        """Save generated heuristic code."""
        from src.run_hyper_heuristic.helper_function import save_generated_heuristic
        return save_generated_heuristic(code, output_dir=self.output_dir)


def complete_cs_workflow(
    results_dict: Dict,
    api_key: str,
    problem: str,
    heuristic_dir: str,
    task_description_file: str,
    output_dir: str,
    model: str
) -> Tuple[str, str]:
    """
    CI workflow - Backward compatible convenience function.
    
    Args:
        results_dict: Performance results of all heuristics
        api_key: LLM API key
        problem: Problem type
        heuristic_dir: Heuristic directory
        task_description_file: Task description file path
        output_dir: Output directory
        model: LLM model name
    
    Returns:
        (file_path, code): Generated heuristic file path and code
    """
    operator = CIOperator(
        problem=problem,
        heuristic_dir=heuristic_dir,
        task_description_file=task_description_file,
        output_dir=output_dir,
        api_key=api_key,
        model=model
    )
    
    return operator.generate(results_dict)