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

class LSPromptFormatter:
    
    def __init__(self, problem, task_description_file):
        
        self.problem = problem
        self.task_description = self._load_task_description(task_description_file)
    
    def _load_task_description(self, file_path) -> str:
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content

    def create_ls_prompt(
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

# Task: Design a New Complementary Heuristic Algorithm

You are an expert in designing heuristic algorithms for the Port Scheduling Problem (PSP).

I have one algorithm with its code as follows:

Algorithm: {h_name}

Code:
```python
{h_code}
```

## Your Task

Please assist me in creating an improved version of the algorithm provided.

**IMPORTANT CODE FORMAT REQUIREMENTS:**

1. The function name must follow the pattern: `<strategy_name>_<random_4_chars>` (e.g., `hybrid_greedy_a3f7`)
2. Follow the exact code format shown in the examples above
3. Ensure your code is complete and executable

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

def generate_complementary_ls_heuristic(
    results_dict: Dict,
    problem: str = "psp",
    heuristic_dir: str = "basic_heuristics",
    task_description_file: str = None,
    save_prompt: bool = True,
    save_code: bool = True
) -> tuple[str, str]:
    """
    完整工作流：从results_dict生成互补启发式
    
    Args:
        results_dict: 所有启发式的评估结果
        problem: 问题类型
        heuristic_dir: 启发式目录
        task_description_file: 任务描述文件路径
        save_prompt: 是否保存生成的prompt
        save_code: 是否保存生成的代码
        
    Returns:
        (prompt, code): 生成的提示词和代码
    """
    
    # Step 1: Initialize LS Engine and select parent   
    h_name = select_parent_for_local_search(results_dict)
    
    with open(task_description_file, 'r', encoding='utf-8') as f:
        task_description = f.read()
 
    h_code = load_heuristic_code(problem, h_name, heuristic_dir)

    formatter = LSPromptFormatter(
        problem=problem,
        task_description_file=task_description_file
    )
    ls_prompt = formatter.create_ls_prompt(
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