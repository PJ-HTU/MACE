"""
EI Operator (Efficiency Improvement Operator)
Corresponds to EI operator in MACE paper - Algorithm 2, lines 7-9
"""

import os
from typing import Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class EIOperator:
    """
    EI (Efficiency Improvement) Operator
    
    Paper Algorithm 2, lines 7-9:
    if t(h') > T_max then
        h' ← EI(LLM, p_ei, h')
    end if
    
    When heuristic times out, reduce computational complexity while maintaining core strategy.
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
        Initialize EI Operator.
        
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
    
    def generate(self, heuristic_path: str) -> Tuple[str, str]:
        """
        Execute complete EI operator workflow.
        
        Args:
            heuristic_path: Heuristic file path to optimize
        
        Returns:
            (file_path, code): Generated optimized heuristic file path and code
        """
        print("\n" + "=" * 80)
        print("⚡ [EI Operator] Efficiency Improvement Started")
        print("=" * 80)
        
        h_name = os.path.basename(heuristic_path)
        h_code = self._load_heuristic_code(heuristic_path)
        print(f"✓ Target heuristic: {h_name}")
        print(f"  Reason: Timeout (t(h') > T_max)")
        
        ei_prompt = self._create_ei_prompt(h_name, h_code)
        
        llm_response = self._call_llm(ei_prompt)
        
        extracted_code = self._extract_code_from_response(llm_response)
        
        if not extracted_code:
            print("✗ Failed to extract code from response")
            print("\nFull response:")
            print(llm_response)
            return None, None
        
        try:
            file_path = self._save_generated_heuristic(extracted_code)
            
            response_file = file_path.replace('.py', '_full_response.txt')
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt used:\n{'-'*80}\n{ei_prompt}\n\n")
                f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")
            
            print(f"✓ Code saved to: {file_path}")
            
        except Exception as e:
            print(f"✗ Save failed: {str(e)}")
            return None, extracted_code
        
        print("\n" + "=" * 80)
        print("✅ [EI Operator] Workflow Completed!")
        print("=" * 80)
        print(f"📁 Generated optimized heuristic: {file_path}")
        print(f"📄 Full response log: {response_file}")
        print(f"🔬 Based on original heuristic: {h_name}")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    def _load_heuristic_code(self, heuristic_path: str) -> str:
        """Load heuristic code."""
        try:
            with open(heuristic_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"✗ Failed to read heuristic code: {e}")
            return ""
    
    def _create_ei_prompt(self, h_name: str, h_code: str) -> str:
        """
        Create EI prompt.
        Goal: Reduce computational complexity while maintaining core strategy.
        """
        prompt_template = PromptTemplate(
            input_variables=[
                "task_description",
                "h_name",
                "h_code"
            ],
            template="""# Problem Context

{task_description}

# Task: Improve Time Efficiency of Heuristic Algorithm

You are an expert in optimizing heuristic algorithms for combinatorial optimization problems.

## Current Heuristic Algorithm: {h_name}

The current algorithm has **EXCEEDED THE TIME BUDGET**. It runs too slowly and must be optimized.

### Current Implementation:
```python
{h_code}
```

## Your Task

**This algorithm is TOO SLOW and exceeds the time limit. You MUST make it MUCH FASTER.**

Please redesign this heuristic algorithm to be **significantly faster** while maintaining solution quality. Focus on:

1. **Reduce Time Complexity**
   - Optimize nested loops (reduce O(n²) to O(n log n) or O(n))
   - Use efficient data structures (heaps, hash tables, sorted containers)
   - Eliminate redundant computations
   - Cache frequently used values

2. **Simplify Operations**
   - Remove unnecessary calculations
   - Reduce the number of iterations
   - Use early stopping when possible
   - Simplify complex logic

3. **Optimize Implementation**
   - Use vectorized operations where possible
   - Avoid repeated function calls
   - Minimize memory allocations
   - Use built-in functions instead of custom implementations

**CRITICAL REQUIREMENTS:**

1. The optimized version must be **AT LEAST 2-3x FASTER** than the original
2. Maintain the **core strategic logic** - don't completely change the approach
3. Solution quality should remain **similar** (small degradation is acceptable)
4. The algorithm must complete **within the time budget**

**IMPORTANT CODE FORMAT REQUIREMENTS:**

1. The function name must follow the pattern: `<strategy_name>_<random_4_chars>` (e.g., `optimized_greedy_x9m2`)
2. Follow the exact code format shown in the example above
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
- Focus on SPEED OPTIMIZATION while preserving strategy

Please provide your optimized heuristic algorithm now:"""
        )
        
        final_prompt = prompt_template.format(
            task_description=self.task_description,
            h_name=h_name,
            h_code=h_code
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
                        "content": "You are an expert in optimizing algorithms for computational efficiency."
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


def improve_efficiency(
    file_path: str,
    api_key: str,
    problem: str,
    heuristic_dir: str,
    task_description_file: str,
    output_dir: str,
    model: str
) -> Tuple[str, str]:
    """
    EI workflow - Backward compatible convenience function.
    
    Args:
        file_path: Heuristic file path to optimize
        api_key: LLM API key
        problem: Problem type
        heuristic_dir: Heuristic directory
        task_description_file: Task description file path
        output_dir: Output directory
        model: LLM model name
    
    Returns:
        (file_path, code): Generated optimized heuristic file path and code
    """
    operator = EIOperator(
        problem=problem,
        heuristic_dir=heuristic_dir,
        task_description_file=task_description_file,
        output_dir=output_dir,
        api_key=api_key,
        model=model
    )
    
    return operator.generate(file_path)