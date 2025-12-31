"""
CI Operator (Complementary Improvement Operator)
对应MACE论文的CI算子 - Algorithm 2, line 3

整合了:
- prompt_cs.py 的提示词生成逻辑
- complete_cs_workflow.py 的完整工作流
"""

import os
from typing import Dict, List, Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class CIOperator:
    """
    CI (Complementary Improvement) 算子
    
    论文公式(3): C(h_a, h_b) = min(W_ab, W_ba) / m
    其中 W_ab = |{i : f_i(h_a) < f_i(h_b)}|
    
    选择具有互补优势的启发式对,生成融合策略的新启发式
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
            problem: 问题类型 (tsp, jssp, cvrp, psp)
            heuristic_dir: 启发式代码目录
            task_description_file: 任务描述文件路径
            output_dir: 输出目录
            api_key: LLM API密钥
            model: LLM模型名称
        """
        self.problem = problem
        self.heuristic_dir = heuristic_dir
        self.task_description_file = task_description_file
        self.output_dir = output_dir
        self.api_key = api_key
        self.model = model
        
        # 加载任务描述
        self.task_description = self._load_task_description()
    
    def _load_task_description(self) -> str:
        """加载任务描述文件"""
        if self.task_description_file and os.path.exists(self.task_description_file):
            with open(self.task_description_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def generate(self, results_dict: Dict) -> Tuple[str, str]:
        """
        执行CI算子完整工作流
        
        Args:
            results_dict: 所有启发式的性能结果
                格式: {heuristic_name: [score_1, score_2, ...]}
        
        Returns:
            (file_path, code): 生成的启发式文件路径和代码
        """
        print("\n" + "=" * 80)
        print("🔍 [CI算子] Complementary Improvement 开始")
        print("=" * 80)
        
        # Step 1: 选择互补对 (论文公式3)
        h1_name, h2_name = self._select_complementary_pair(results_dict)
        print(f"✓ 选择的互补对: {h1_name} 和 {h2_name}")
        
        # Step 2: 加载启发式代码
        h1_code = self._load_heuristic_code(h1_name)
        h2_code = self._load_heuristic_code(h2_name)
        
        # Step 3: 生成CI提示词
        ci_prompt = self._create_ci_prompt(h1_name, h1_code, h2_name, h2_code)
        
        # Step 4: 调用LLM生成新启发式
        llm_response = self._call_llm(ci_prompt)
        
        # Step 5: 提取代码
        extracted_code = self._extract_code_from_response(llm_response)
        
        if not extracted_code:
            print("✗ 未能从响应中提取代码")
            print("\n完整响应:")
            print(llm_response)
            return None, None
        
        # Step 6: 保存代码
        try:
            file_path = self._save_generated_heuristic(extracted_code)
            
            # 保存完整响应（包括思考过程）
            response_file = file_path.replace('.py', '_full_response.txt')
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt used:\n{'-'*80}\n{ci_prompt}\n\n")
                f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")
            
            print(f"✓ 代码已保存到: {file_path}")
            
        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")
            return None, extracted_code
        
        # 完成
        print("\n" + "=" * 80)
        print("✅ [CI算子] 工作流完成!")
        print("=" * 80)
        print(f"📁 生成的启发式代码: {file_path}")
        print(f"📄 完整响应记录: {response_file}")
        print(f"🔬 基于互补对: {h1_name} + {h2_name}")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    def _select_complementary_pair(self, results_dict: Dict) -> Tuple[str, str]:
        """
        选择互补对 - 论文公式(3)
        
        C(h_a, h_b) = min(W_ab, W_ba) / m
        其中 W_ab = |{i : f_i(h_a) < f_i(h_b)}|
        """
        # 导入辅助函数
        from src.run_hyper_heuristic.helper_function import select_complementary_pair
        return select_complementary_pair(results_dict)
    
    def _load_heuristic_code(self, heuristic_name: str) -> str:
        """加载启发式代码"""
        from src.run_hyper_heuristic.helper_function import load_heuristic_code
        return load_heuristic_code(self.problem, heuristic_name, self.heuristic_dir)
    
    def _create_ci_prompt(
        self,
        h1_name: str,
        h1_code: str,
        h2_name: str,
        h2_code: str
    ) -> str:
        """
        创建CI提示词
        """
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
        
        # 生成最终提示词
        final_prompt = prompt_template.format(
            task_description=self.task_description,
            heuristic1_name=h1_name,
            heuristic1_code=h1_code,
            heuristic2_name=h2_name,
            heuristic2_code=h2_code
        )
        
        return final_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM生成代码"""
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"  # OpenAI SDK会自动添加/chat/completions
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
            print(f"✗ API调用失败: {str(e)}")
            raise
    
    def _extract_code_from_response(self, response: str) -> str:
        """从LLM响应中提取代码"""
        from src.run_hyper_heuristic.helper_function import extract_code_from_response
        return extract_code_from_response(response)
    
    def _save_generated_heuristic(self, code: str) -> str:
        """保存生成的启发式代码"""
        from src.run_hyper_heuristic.helper_function import save_generated_heuristic
        return save_generated_heuristic(code, output_dir=self.output_dir)


# 便捷函数 - 保持与原来complete_cs_workflow的兼容性
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
    CI工作流 - 向后兼容的便捷函数
    
    Args:
        results_dict: 所有启发式的性能结果
        api_key: LLM API密钥
        problem: 问题类型
        heuristic_dir: 启发式目录
        task_description_file: 任务描述文件路径
        output_dir: 输出目录
        model: LLM模型名称
    
    Returns:
        (file_path, code): 生成的启发式文件路径和代码
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