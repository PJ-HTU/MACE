"""
EI Operator (Efficiency Improvement Operator)
对应MACE论文的EI算子 - Algorithm 2, lines 7-9

整合了:
- prompt_ie.py 的提示词生成逻辑
- improve_efficiency.py 的完整工作流
"""

import os
from typing import Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class EIOperator:
    """
    EI (Efficiency Improvement) 算子
    
    论文Algorithm 2, lines 7-9:
    if t(h') > T_max then
        h' ← EI(LLM, p_ei, h')
    end if
    
    当启发式超时时,减少计算复杂度,保持核心策略
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
    
    def generate(self, heuristic_path: str) -> Tuple[str, str]:
        """
        执行EI算子完整工作流
        
        Args:
            heuristic_path: 需要优化的启发式文件路径
        
        Returns:
            (file_path, code): 生成的优化后启发式文件路径和代码
        """
        print("\n" + "=" * 80)
        print("⚡ [EI算子] Efficiency Improvement 开始")
        print("=" * 80)
        
        # Step 1: 加载启发式代码
        h_name = os.path.basename(heuristic_path)
        h_code = self._load_heuristic_code(heuristic_path)
        print(f"✓ 目标启发式: {h_name}")
        print(f"  原因: 超时 (t(h') > T_max)")
        
        # Step 2: 生成EI提示词
        ei_prompt = self._create_ei_prompt(h_name, h_code)
        
        # Step 3: 调用LLM生成优化后的启发式
        llm_response = self._call_llm(ei_prompt)
        
        # Step 4: 提取代码
        extracted_code = self._extract_code_from_response(llm_response)
        
        if not extracted_code:
            print("✗ 未能从响应中提取代码")
            print("\n完整响应:")
            print(llm_response)
            return None, None
        
        # Step 5: 保存代码
        try:
            file_path = self._save_generated_heuristic(extracted_code)
            
            # 保存完整响应（包括思考过程）
            response_file = file_path.replace('.py', '_full_response.txt')
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt used:\n{'-'*80}\n{ei_prompt}\n\n")
                f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")
            
            print(f"✓ 代码已保存到: {file_path}")
            
        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")
            return None, extracted_code
        
        # 完成
        print("\n" + "=" * 80)
        print("✅ [EI算子] 工作流完成!")
        print("=" * 80)
        print(f"📁 生成的优化启发式: {file_path}")
        print(f"📄 完整响应记录: {response_file}")
        print(f"🔬 基于原始启发式: {h_name}")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    def _load_heuristic_code(self, heuristic_path: str) -> str:
        """加载启发式代码"""
        try:
            with open(heuristic_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"✗ 读取启发式代码失败: {e}")
            return ""
    
    def _create_ei_prompt(self, h_name: str, h_code: str) -> str:
        """
        创建EI提示词
        目标: 减少计算复杂度,保持核心策略
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
        
        # 生成最终提示词
        final_prompt = prompt_template.format(
            task_description=self.task_description,
            h_name=h_name,
            h_code=h_code
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


# 便捷函数 - 保持与原来improve_efficiency的兼容性
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
    EI工作流 - 向后兼容的便捷函数
    
    Args:
        file_path: 需要优化的启发式文件路径
        api_key: LLM API密钥
        problem: 问题类型
        heuristic_dir: 启发式目录
        task_description_file: 任务描述文件路径
        output_dir: 输出目录
        model: LLM模型名称
    
    Returns:
        (file_path, code): 生成的优化后启发式文件路径和代码
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