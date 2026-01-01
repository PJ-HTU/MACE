"""
PI Operator (Performance Improvement Operator)
Corresponds to the PI operator in the MACE paper - Algorithm 2, line 4

Integrates:
- Prompt generation logic from prompt_ls.py
- Complete workflow from local_search_workflow.py
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class PIOperator:
    """
    PI (Performance Improvement) Operator
    
    Paper formula (4): h* = arg max σ_r(h)
    where σ_r(h) = std({r_i(h)})
    r_i(h) ∈ {1,...,n} represents the performance rank of h on instance i
    
    Selects heuristics with maximum ranking variance (unstable performance) and reduces variance through improvement
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
        """Load task description file"""
        if self.task_description_file and os.path.exists(self.task_description_file):
            with open(self.task_description_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def generate(self, population: List[Dict]) -> Tuple[str, str]:
        """
        Execute complete PI operator workflow
        
        Args:
            population: Current population
                Format: [{'name': ..., 'performance_vector': [...], 'avg_performance': ...}, ...]
        
        Returns:
            (file_path, code): Generated heuristic file path and code
        """
        print("\n" + "=" * 80)
        print("🔧 [PI Operator] Performance Improvement Started")
        print("=" * 80)
        
        h_name, ranking_variance = self._select_parent_with_max_variance(population)
        print(f"✓ Selected parent heuristic: {h_name}")
        print(f"  Ranking variance σ_r: {ranking_variance:.4f}")
        
        h_code = self._load_heuristic_code(h_name)
        
        pi_prompt = self._create_pi_prompt(h_name, h_code)
        
        llm_response = self._call_llm(pi_prompt)
        
        extracted_code = self._extract_code_from_response(llm_response)
        
        if not extracted_code:
            print("✗ Failed to extract code from response")
            print("\nComplete response:")
            print(llm_response)
            return None, None
        
        try:
            file_path = self._save_generated_heuristic(extracted_code)
            
            response_file = file_path.replace('.py', '_full_response.txt')
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt used:\n{'-'*80}\n{pi_prompt}\n\n")
                f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")
            
            print(f"✓ Code saved to: {file_path}")
            
        except Exception as e:
            print(f"✗ Save failed: {str(e)}")
            return None, extracted_code
        
        print("\n" + "=" * 80)
        print("✅ [PI Operator] Workflow completed!")
        print("=" * 80)
        print(f"📁 Generated heuristic code: {file_path}")
        print(f"📄 Full response log: {response_file}")
        print(f"🔬 Based on parent: {h_name} (σ_r={ranking_variance:.4f})")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    def _select_parent_with_max_variance(self, population: List[Dict]) -> Tuple[str, float]:
        """
        Select heuristic with maximum ranking variance - Paper formula (4)
        
        σ_r(h) = std({r_i(h)})
        where r_i(h) is the rank of h on instance i
        
        Args:
            population: Current population
        
        Returns:
            (h_name, ranking_variance): Selected heuristic name and its ranking variance
        """
        if not population:
            raise ValueError("Population is empty, cannot select parent")
        
        n = len(population)
        m = len(population[0]['performance_vector'])
        
        variances = {}
        
        for h in population:
            ranks = []
            
            for instance_idx in range(m):
                performances = [
                    (p['name'], p['performance_vector'][instance_idx])
                    for p in population
                ]
                
                performances.sort(key=lambda x: x[1])
                
                for rank, (name, _) in enumerate(performances, 1):
                    if name == h['name']:
                        ranks.append(rank)
                        break
            
            ranking_variance = np.std(ranks)
            variances[h['name']] = ranking_variance
        
        h_name = max(variances, key=variances.get)
        ranking_variance = variances[h_name]
        
        return h_name, ranking_variance
    
    def _load_heuristic_code(self, heuristic_name: str) -> str:
        """Load heuristic code"""
        from src.run_hyper_heuristic.helper_function import load_heuristic_code
        return load_heuristic_code(self.problem, heuristic_name, self.heuristic_dir)
    
    def _create_pi_prompt(self, h_name: str, h_code: str) -> str:
        """
        Create PI prompt
        Goal: Improve heuristic, reduce performance variance, maintain overall strategy
        """
        prompt_template = PromptTemplate(
            input_variables=[
                "task_description",
                "h_name",
                "h_code"
            ],
            template="""# Problem Context

{task_description}

# Task: Improve Heuristic Algorithm Performance

You are an expert in designing heuristic algorithms for combinatorial optimization problems.

I have an existing heuristic algorithm that shows **unstable performance** across different problem instances (high ranking variance). Your task is to create an **improved version** that:

1. **Reduces performance variability** - The algorithm should be more consistent across instances
2. **Preserves the overall strategy** - Keep the core algorithmic approach
3. **Enhances solution quality** - Improve average performance where possible

## Current Algorithm: {h_name}

```python
{h_code}
```

## Your Task

Please analyze this algorithm and create an **improved version** that addresses its performance instability. Consider:

- **Why might this algorithm be unstable?** (e.g., parameter sensitivity, edge cases, instance-specific weaknesses)
- **How can you make it more robust?** (e.g., adaptive parameters, better tie-breaking, hybrid strategies)
- **Can you refine the decision logic?** (e.g., better selection criteria, improved evaluation functions)

**IMPORTANT CODE FORMAT REQUIREMENTS:**

1. The function name must follow the pattern: `<strategy_name>_<random_4_chars>` (e.g., `improved_greedy_x7k2`)
2. Follow the exact code format shown in the example above
3. Ensure your code is complete and executable
4. The improved version should be MORE STABLE than the original

**Response Format:**

The response format is very important. Please respond in this format:

***python_code:
[Your complete Python code here]
***

**CRITICAL:** 
- Ensure there is no other content inside the ***
- Analysis and explanation outside *** are welcome
- The code must be complete and runnable
- Focus on reducing performance variance while maintaining quality

Please provide your improved heuristic algorithm now:"""
        )
        
        final_prompt = prompt_template.format(
            task_description=self.task_description,
            h_name=h_name,
            h_code=h_code
        )
        
        return final_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate code"""
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
        """Extract code from LLM response"""
        from src.run_hyper_heuristic.helper_function import extract_code_from_response
        return extract_code_from_response(response)
    
    def _save_generated_heuristic(self, code: str) -> str:
        """Save generated heuristic code"""
        from src.run_hyper_heuristic.helper_function import save_generated_heuristic
        return save_generated_heuristic(code, output_dir=self.output_dir)


def local_search_workflow(
    results_dict: Dict,
    api_key: str,
    problem: str,
    heuristic_dir: str,
    task_description_file: str,
    output_dir: str,
    model: str
) -> Tuple[str, str]:
    """
    PI workflow - backward compatible convenience function
    
    Args:
        results_dict: Performance results of all heuristics (used to construct population)
        api_key: LLM API key
        problem: Problem type
        heuristic_dir: Heuristic directory
        task_description_file: Task description file path
        output_dir: Output directory
        model: LLM model name
    
    Returns:
        (file_path, code): Generated heuristic file path and code
    """
    population = []
    for h_name, scores in results_dict.items():
        population.append({
            'name': h_name,
            'performance_vector': scores,
            'avg_performance': np.mean(scores)
        })
    
    operator = PIOperator(
        problem=problem,
        heuristic_dir=heuristic_dir,
        task_description_file=task_description_file,
        output_dir=output_dir,
        api_key=api_key,
        model=model
    )
    
    return operator.generate(population)