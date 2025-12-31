"""
SI Operator (Specialization Improvement Operator)
对应MACE论文的SI算子 - Algorithm 2, line 5

核心逻辑:
- 选择变异系数(CV)最小的实例
- 针对该实例生成专业化的启发式
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class SIOperator:
    """
    SI (Specialization Improvement) 算子
    
    论文公式(5): i* = arg min CV_i
    其中 CV_i = std({f_i(h) : h ∈ H}) / mean({f_i(h) : h ∈ H})
    
    选择变异系数最小的实例(算法覆盖不足),生成专业化启发式
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
            problem: 问题类型 (tsp, jssp, cvrp, psp)
            heuristic_dir: 启发式代码目录
            task_description_file: 任务描述文件路径
            output_dir: 输出目录
            test_data_dir: 测试数据目录 (用于读取实例文件)
            api_key: LLM API密钥
            model: LLM模型名称
        """
        self.problem = problem
        self.heuristic_dir = heuristic_dir
        self.task_description_file = task_description_file
        self.output_dir = output_dir
        self.test_data_dir = test_data_dir
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
        执行SI算子完整工作流
        
        Args:
            population: 当前种群
                格式: [{'name': ..., 'performance_vector': [...], 'avg_performance': ...}, ...]
        
        Returns:
            (file_path, code): 生成的启发式文件路径和代码
        """
        print("\n" + "=" * 80)
        print("🎯 [SI算子] Specialization Improvement 开始")
        print("=" * 80)
        
        # Step 1: 选择CV最小的实例 (论文公式5)
        instance_idx, cv_value, instance_info = self._select_underserved_instance(population)
        print(f"✓ 选择的目标实例: 实例 #{instance_idx}")
        print(f"  变异系数 CV: {cv_value:.4f} (最小 - 覆盖不足)")
        print(f"  实例信息: {instance_info}")
        
        # Step 2: 分析该实例上各启发式的表现
        instance_analysis = self._analyze_instance_performance(population, instance_idx)
        
        # Step 3: 生成SI提示词
        si_prompt = self._create_si_prompt(instance_idx, instance_info, instance_analysis)
        
        # Step 4: 调用LLM生成专业化启发式
        llm_response = self._call_llm(si_prompt)
        
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
                f.write(f"Prompt used:\n{'-'*80}\n{si_prompt}\n\n")
                f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")
            
            print(f"✓ 代码已保存到: {file_path}")
            
        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")
            return None, extracted_code
        
        # 完成
        print("\n" + "=" * 80)
        print("✅ [SI算子] 工作流完成!")
        print("=" * 80)
        print(f"📁 生成的启发式代码: {file_path}")
        print(f"📄 完整响应记录: {response_file}")
        print(f"🔬 专业化目标: 实例 #{instance_idx} (CV={cv_value:.4f})")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    def _select_underserved_instance(self, population: List[Dict]) -> Tuple[int, float, str]:
        """
        选择变异系数最小的实例 - 论文公式(5)
        
        CV_i = std({f_i(h) : h ∈ H}) / mean({f_i(h) : h ∈ H})
        
        Args:
            population: 当前种群
        
        Returns:
            (instance_idx, cv_value, instance_info): 实例索引、CV值、实例信息
        """
        if not population:
            raise ValueError("种群为空,无法选择实例")
        
        m = len(population[0]['performance_vector'])  # 实例数量
        
        # 计算每个实例的变异系数
        cv_scores = {}
        
        for instance_idx in range(m):
            # 获取所有启发式在该实例上的性能
            performances = [h['performance_vector'][instance_idx] for h in population]
            
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            
            # 计算变异系数 CV = std / mean
            if mean_perf > 0:
                cv = std_perf / mean_perf
            else:
                cv = float('inf')  # 避免除零
            
            cv_scores[instance_idx] = cv
        
        # 选择CV最小的实例 (算法覆盖不足,性能差异小)
        target_instance_idx = min(cv_scores, key=cv_scores.get)
        target_cv = cv_scores[target_instance_idx]
        
        # 获取实例信息
        instance_info = self._get_instance_info(target_instance_idx)
        
        return target_instance_idx, target_cv, instance_info
    
    def _get_instance_info(self, instance_idx: int) -> str:
        """
        获取实例的描述信息
        
        Args:
            instance_idx: 实例索引
        
        Returns:
            instance_info: 实例描述
        """
        # 尝试读取实例文件名
        try:
            # 假设test_data目录下的文件按顺序对应实例索引
            if os.path.exists(self.test_data_dir):
                files = sorted([f for f in os.listdir(self.test_data_dir) 
                              if not f.startswith('.')])
                if instance_idx < len(files):
                    instance_file = files[instance_idx]
                    return f"文件: {instance_file}"
        except Exception as e:
            print(f"⚠️ 无法读取实例信息: {e}")
        
        return f"实例索引: {instance_idx}"
    
    def _analyze_instance_performance(self, population: List[Dict], instance_idx: int) -> str:
        """
        分析各启发式在目标实例上的表现
        
        Args:
            population: 当前种群
            instance_idx: 目标实例索引
        
        Returns:
            analysis: 性能分析文本
        """
        performances = []
        
        for h in population:
            perf = h['performance_vector'][instance_idx]
            performances.append({
                'name': os.path.basename(h['name']),
                'performance': perf
            })
        
        # 按性能排序
        performances.sort(key=lambda x: x['performance'])
        
        # 生成分析文本
        analysis = f"目标实例 #{instance_idx} 上的性能分析:\n"
        analysis += f"  最佳性能: {performances[0]['performance']:.2f} ({performances[0]['name']})\n"
        analysis += f"  最差性能: {performances[-1]['performance']:.2f} ({performances[-1]['name']})\n"
        analysis += f"  平均性能: {np.mean([p['performance'] for p in performances]):.2f}\n"
        analysis += f"  标准差: {np.std([p['performance'] for p in performances]):.2f}\n"
        
        return analysis
    
    def _create_si_prompt(
        self,
        instance_idx: int,
        instance_info: str,
        instance_analysis: str
    ) -> str:
        """
        创建SI提示词
        目标: 为特定实例生成专业化启发式
        """
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
- This instance is **under-served** by the current portfolio
- There is potential for a specialized heuristic to significantly improve performance

## Target Instance Information

{instance_info}

## Current Performance on This Instance

{instance_analysis}

## Your Task

Design a **NEW specialized heuristic** that is explicitly optimized for the structural characteristics of this specific instance type. Consider:

1. **Instance-Specific Patterns**: What makes this instance unique?
   - Size characteristics
   - Data distribution
   - Structural properties
   - Constraint patterns

2. **Why Existing Heuristics Underperform**: Analyze potential weaknesses
   - Are they too general?
   - Do they miss specific patterns?
   - Are there unexploited structures?

3. **Specialization Strategies**: Design targeted approaches
   - Exploit specific instance features
   - Use domain knowledge for this instance type
   - Implement custom decision rules
   - Optimize for this specific scale/structure

**IMPORTANT CODE FORMAT REQUIREMENTS:**

1. The function name must follow the pattern: `<strategy_name>_<random_4_chars>` (e.g., `specialized_greedy_k9m4`)
2. Follow the exact code format shown in the examples
3. Ensure your code is complete and executable
4. The heuristic should be **highly specialized** for this instance type

**Response Format:**

The response format is very important. Please respond in this format:

***python_code:
[Your complete Python code here]
***

**CRITICAL:** 
- Ensure there is no other content inside the ***
- Analysis and explanation outside *** are welcome
- The code must be complete and runnable
- Focus on instance-specific optimization

Please provide your specialized heuristic algorithm now:"""
        )
        
        # 生成最终提示词
        final_prompt = prompt_template.format(
            task_description=self.task_description,
            instance_idx=instance_idx,
            instance_info=instance_info,
            instance_analysis=instance_analysis
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