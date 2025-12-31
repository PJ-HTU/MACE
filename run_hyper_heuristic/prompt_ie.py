from openai import OpenAI
import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import search_file
import json
from src.run_hyper_heuristic.helper_function import select_complementary_pair, load_heuristic_code, extract_code_from_response, save_generated_heuristic, select_parent_for_local_search
from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Tuple, Any
import re
import sys

class IEPromptFormatter:
    
    def __init__(self, problem, task_description_file):
        
        self.problem = problem
        self.task_description = self._load_task_description(task_description_file)
    
    def _load_task_description(self, file_path) -> str:
        
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content

    def create_ie_prompt(
        self, 
        h_name: str, 
        h_code: str
    ) -> str:
        
        prompt_template = PromptTemplate(
            input_variables=[
                "task_description",
                "h_name",
                "h_code"
            ],
            template="""# Problem Context

{task_description}

# Task: Improve Time Efficiency of the Heuristic Algorithm

You are an expert in optimizing heuristic algorithms for the Port Scheduling Problem (PSP).

## Current Heuristic Algorithm: {h_name}

The current algorithm has **TIME EFFICIENCY ISSUES**. It runs too slowly and may exceed time limits.

### Current Implementation:
{h_code}

## Your Task

**This algorithm's time efficiency is NOT GOOD. You need to IMPROVE it.**

Please redesign this heuristic algorithm to be **MUCH FASTER** while maintaining solution quality. Focus on:

1. **Reduce time complexity** - optimize nested loops, use efficient data structures
2. **Simplify operations** - remove unnecessary calculations
3. **Optimize the implementation** - make the code run faster

**IMPORTANT CODE FORMAT REQUIREMENTS:**

1. The function name must follow the pattern: `<strategy_name>_<random_4_chars>` (e.g., `optimized_greedy_b8k2`)
2. Follow the exact code format shown in the example above
3. Ensure your code is complete and executable
4. The improved version must be FASTER than the original

**Response Format:**

The response format is very important. Please respond in this format:

***python_code:
from src.problems.tsp.components import *

def your_new_function_name(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[CompleteVesselAssignmentOperator, dict]:
    \"\"\" Description for this heuristic algorithm.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - necessary_key (value type): description
            - necessary_key (value type): description
            ...
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - necessary_key (value type): description
            - necessary_key (value type): description
            ...
        **kwargs: Description of hyper parameters if used.

    Returns:
        CompleteVesselAssignmentOperator: Description of the operator returned.
        dict: Description of updated algorithm data.
    \"\"\"
    
    # Your implementation here
    pass
***

**CRITICAL:** Ensure there is no other content inside the ***, and analysis outside *** is welcome.

Please provide your new heuristic algorithm now:"""
        )
        
        # 生成最终提示词
        final_prompt = prompt_template.format(
            task_description=self.task_description,
            h_name=h_name,
            h_code=h_code
        )
        
        return final_prompt

def generate_complementary_ie_heuristic(
    file_path,
    problem,
    heuristic_dir,
    task_description_file,
    save_prompt: bool = True,
    save_code: bool = True
) -> tuple[str, str]:
     
   
    file_name_with_ext = os.path.basename(file_path)
    h_name = os.path.splitext(file_name_with_ext)[0]
    
    with open(task_description_file, 'r', encoding='utf-8') as f:
        task_description = f.read()
 
    h_code = load_heuristic_code(problem, h_name, heuristic_dir)

    formatter = IEPromptFormatter(
        problem=problem,
        task_description_file=task_description_file
    )
    ls_prompt = formatter.create_ie_prompt(
        h_name= h_name,
        h_code= h_code,
    )
    
    # 4. 保存提示词
    if save_prompt:
        prompt_file = f"ls_prompt_{h_name}.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(ls_prompt)
    
    return ls_prompt, h_name


def call_deepseek_reasoner(prompt: str, api_key, model,
                           base_url: str = "https://openrouter.ai/api/v1/chat/completions") -> str:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    try:
        # 调用API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert algorithm designer for optimization problems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # 提取响应
        llm_response = response.choices[0].message.content
        
        return llm_response
        
    except Exception as e:
        print(f"✗ API调用失败: {str(e)}")
        raise