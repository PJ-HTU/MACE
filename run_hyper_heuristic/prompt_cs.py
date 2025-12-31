from openai import OpenAI
import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import search_file
import json
from src.run_hyper_heuristic.helper_function import select_complementary_pair, load_heuristic_code, extract_code_from_response, save_generated_heuristic
from langchain_core.prompts import PromptTemplate
from typing import Dict
import re
import sys

class CSPromptFormatter:
    """
    Complementary-aware Search 提示词格式化器 (PSP格式)
    """
    
    def __init__(self, problem, task_description_file):
        """
        Args:
            problem: 问题类型 (默认 'psp')
            task_description_file: 任务描述文件路径 (可选)
        """
        self.problem = problem
        self.task_description = self._load_task_description(task_description_file)
    
    def _load_task_description(self, file_path) -> str:
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content

    def create_cs_prompt(
        self, 
        heuristic1_name: str, 
        heuristic1_code: str,
        heuristic2_name: str, 
        heuristic2_code: str
    ) -> str:
        
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

You are an expert in designing heuristic algorithms for the Port Scheduling Problem (PSP).

I have 2 existing heuristic algorithms that show complementary performance across different problem instances:

## Heuristic Algorithm 1: {heuristic1_name}
{heuristic1_code}

## Heuristic Algorithm 2: {heuristic2_name}
{heuristic2_code}

## Your Task

These two algorithms are effective for solving different instance distributions. Please design a NEW heuristic algorithm that is DIFFERENT from both of them. Your algorithm should:

1. **Be distinct** from the two provided algorithms
2. **Explore new strategies** or combine their strengths in innovative ways
3. **Consider complementarity** - think about what problem characteristics each algorithm handles well, and design something that could handle different scenarios

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
            heuristic1_name=heuristic1_name,
            heuristic1_code=heuristic1_code,
            heuristic2_name=heuristic2_name,
            heuristic2_code=heuristic2_code
        )
        
        return final_prompt

def generate_complementary_heuristic(
    results_dict,
    problem,
    heuristic_dir,
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
    # 1. 选择互补对

    
    # 导入互补对选择函数
    import sys
    sys.path.append('.')
    
    # 简单的互补对选择
    h1_name, h2_name= select_complementary_pair(results_dict)
    print(f"✓ 选择的互补对: {h1_name} 和 {h2_name}")
    
    h1_code = load_heuristic_code(problem, h1_name, heuristic_dir)
    h2_code = load_heuristic_code(problem, h2_name, heuristic_dir)
    
    formatter = CSPromptFormatter(
        problem=problem,
        task_description_file=task_description_file
    )
    cs_prompt = formatter.create_cs_prompt(
        heuristic1_name=h1_name,
        heuristic1_code=h1_code,
        heuristic2_name=h2_name,
        heuristic2_code=h2_code
    )
    
    # 4. 保存提示词
    if save_prompt:
        prompt_file = f"cs_prompt_{h1_name}_vs_{h2_name}.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(cs_prompt)
       
    return cs_prompt, (h1_name, h2_name)


def call_deepseek_reasoner(prompt: str, api_key, model,
                           base_url: str = "https://openrouter.ai/api/v1") -> str:
    
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
        print(f"✗ API调用失败这里错误: {str(e)}")
        raise