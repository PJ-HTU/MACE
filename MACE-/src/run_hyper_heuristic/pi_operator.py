"""
PI Operator (Performance Improvement Operator)
对应MACE论文的PI算子 - Algorithm 2, line 4

整合了:
- prompt_ls.py 的提示词生成逻辑
- local_search_workflow.py 的完整工作流
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class PIOperator:
    """
    PI (Performance Improvement) 算子
    
    论文公式(4): h* = arg max σ_r(h)
    其中 σ_r(h) = std({r_i(h)})
    r_i(h) ∈ {1,...,n} 表示h在实例i上的性能排名
    
    选择排名方差最大的启发式(性能不稳定),通过改进减少方差
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
    
    def generate(self, population: List[Dict]) -> Tuple[str, str]:
        """
        执行PI算子完整工作流
        
        Args:
            population: 当前种群
                格式: [{'name': ..., 'performance_vector': [...], 'avg_performance': ...}, ...]
        
        Returns:
            (file_path, code): 生成的启发式文件路径和代码
        """
        print("\n" + "=" * 80)
        print("🔧 [PI算子] Performance Improvement 开始")
        print("=" * 80)
        
        # Step 1: 选择排名方差最大的启发式 (论文公式4)
        h_name, ranking_variance = self._select_parent_with_max_variance(population)
        print(f"✓ 选择的父代启发式: {h_name}")
        print(f"  排名方差 σ_r: {ranking_variance:.4f}")
        
        # Step 2: 加载启发式代码
        h_code = self._load_heuristic_code(h_name)
        
        # Step 3: 生成PI提示词
        pi_prompt = self._create_pi_prompt(h_name, h_code)
        
        # Step 4: 调用LLM生成改进的启发式
        llm_response = self._call_llm(pi_prompt)
        
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
                f.write(f"Prompt used:\n{'-'*80}\n{pi_prompt}\n\n")
                f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")
            
            print(f"✓ 代码已保存到: {file_path}")
            
        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")
            return None, extracted_code
        
        # 完成
        print("\n" + "=" * 80)
        print("✅ [PI算子] 工作流完成!")
        print("=" * 80)
        print(f"📁 生成的启发式代码: {file_path}")
        print(f"📄 完整响应记录: {response_file}")
        print(f"🔬 基于父代: {h_name} (σ_r={ranking_variance:.4f})")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    def _select_parent_with_max_variance(self, population: List[Dict]) -> Tuple[str, float]:
        """
        选择排名方差最大的启发式 - 论文公式(4)
        
        σ_r(h) = std({r_i(h)})
        其中 r_i(h) 是h在实例i上的排名
        
        Args:
            population: 当前种群
        
        Returns:
            (h_name, ranking_variance): 选中的启发式名称和其排名方差
        """
        if not population:
            raise ValueError("种群为空,无法选择父代")
        
        n = len(population)  # 种群大小
        m = len(population[0]['performance_vector'])  # 实例数量
        
        # 计算每个启发式的排名方差
        variances = {}
        
        for h in population:
            # 计算该启发式在每个实例上的排名
            ranks = []
            
            for instance_idx in range(m):
                # 获取所有启发式在该实例上的性能
                performances = [
                    (p['name'], p['performance_vector'][instance_idx])
                    for p in population
                ]
                
                # 排序 (性能越小越好,排名越高)
                performances.sort(key=lambda x: x[1])
                
                # 找到当前启发式的排名 (1-indexed)
                for rank, (name, _) in enumerate(performances, 1):
                    if name == h['name']:
                        ranks.append(rank)
                        break
            
            # 计算排名的标准差
            ranking_variance = np.std(ranks)
            variances[h['name']] = ranking_variance
        
        # 选择排名方差最大的启发式
        h_name = max(variances, key=variances.get)
        ranking_variance = variances[h_name]
        
        return h_name, ranking_variance
    
    def _load_heuristic_code(self, heuristic_name: str) -> str:
        """加载启发式代码"""
        from src.run_hyper_heuristic.helper_function import load_heuristic_code
        return load_heuristic_code(self.problem, heuristic_name, self.heuristic_dir)
    
    def _create_pi_prompt(self, h_name: str, h_code: str) -> str:
        """
        创建PI提示词
        目标: 改进启发式,减少性能方差,保持整体策略
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


# 便捷函数 - 保持与原来local_search_workflow的兼容性
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
    PI工作流 - 向后兼容的便捷函数
    
    Args:
        results_dict: 所有启发式的性能结果 (用于构建population)
        api_key: LLM API密钥
        problem: 问题类型
        heuristic_dir: 启发式目录
        task_description_file: 任务描述文件路径
        output_dir: 输出目录
        model: LLM模型名称
    
    Returns:
        (file_path, code): 生成的启发式文件路径和代码
    """
    # 将results_dict转换为population格式
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