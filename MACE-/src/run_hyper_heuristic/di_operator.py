"""
DI Operator (Diversity Improvement Operator)
对应MACE论文的DI算子 - Algorithm 2, line 6

两阶段实现:
Stage 1: 分析现有Portfolio的策略模式
Stage 2: 基于分析结果生成完全不同的新算法

核心逻辑:
- 输入整个Portfolio H
- 总结现有算法的共同模式和策略
- 识别未被探索的策略空间
- 生成根本性不同的新算法
"""

import os
from typing import Dict, List, Tuple
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


class DIOperator:
    """
    DI (Diversity Improvement) 算子
    
    论文描述: 
    - Takes existing portfolio H as input
    - Summarizes patterns in H
    - Prompts LLM to explore fundamentally different strategies
    - Mitigates premature convergence
    - Expands search space
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
        执行DI算子完整工作流 - 两阶段设计
        
        Args:
            population: 当前种群 (Portfolio H)
                格式: [{'name': ..., 'file': ..., 'code': ..., 'performance_vector': [...], 'avg_performance': ...}, ...]
        
        Returns:
            (file_path, code): 生成的启发式文件路径和代码
        """
        print("\n" + "=" * 80)
        print("🌈 [DI算子] Diversity Improvement 开始 (两阶段)")
        print("=" * 80)
        print(f"当前Portfolio大小: {len(population)}")
        
        # ============================================================
        # STAGE 1: 分析Portfolio的策略模式
        # ============================================================
        print("\n" + "-" * 80)
        print("📊 Stage 1: 分析Portfolio策略模式")
        print("-" * 80)
        
        # Step 1.1: 加载所有启发式代码
        portfolio_codes = self._load_portfolio_codes(population)
        print(f"✓ 已加载 {len(portfolio_codes)} 个启发式代码")
        
        # Step 1.2: 调用LLM进行深度分析
        analysis_prompt = self._create_analysis_prompt(portfolio_codes)
        print("✓ 正在调用LLM进行策略分析...")
        
        analysis_summary = self._call_llm_for_analysis(analysis_prompt)
        
        # Step 1.3: 提取分析结果
        extracted_summary = self._extract_summary_from_response(analysis_summary)
        
        if not extracted_summary:
            print("⚠️  警告: 未能提取有效的分析摘要，使用完整响应")
            extracted_summary = analysis_summary
        
        
        # ============================================================
        # STAGE 2: 基于分析生成多样化的新算法
        # ============================================================
        print("\n" + "-" * 80)
        print("🎨 Stage 2: 生成多样化新算法")
        print("-" * 80)
        
        # Step 2.1: 创建生成提示词
        generation_prompt = self._create_generation_prompt(extracted_summary)
        
        # Step 2.2: 调用LLM生成新算法
        print("✓ 正在调用LLM生成新算法...")
        llm_response = self._call_llm_for_generation(generation_prompt)
        
        # Step 2.3: 提取代码
        extracted_code = self._extract_code_from_response(llm_response)
        
        if not extracted_code:
            print("✗ 未能从响应中提取代码")
            print("\n完整响应:")
            print(llm_response[:1000])
            return None, None
        
        # Step 2.4: 保存代码
        try:
            file_path = self._save_generated_heuristic(extracted_code)
            
            # ✅ 保存完整记录 - 使用动态文件名
            base_name = os.path.basename(file_path).replace('.py', '')
            record_file = os.path.join(self.output_dir, f"{base_name}_full_response.txt")
            
            with open(record_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DI算子生成记录 - 两阶段流程\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("STAGE 1: 策略分析\n")
                f.write("-" * 80 + "\n")
                f.write("分析提示词:\n")
                f.write(analysis_prompt + "\n\n")
                f.write("分析结果:\n")
                f.write(analysis_summary + "\n\n")
                
                f.write("=" * 80 + "\n\n")
                f.write("STAGE 2: 算法生成\n")
                f.write("-" * 80 + "\n")
                f.write("生成提示词:\n")
                f.write(generation_prompt + "\n\n")
                f.write("生成响应:\n")
                f.write(llm_response + "\n")
            
            print(f"✓ 代码已保存到: {file_path}")
            print(f"✓ 完整记录保存到: {record_file}")
            
        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")
            return None, extracted_code
        
        # 完成
        print("\n" + "=" * 80)
        print("✅ [DI算子] 两阶段工作流完成!")
        print("=" * 80)
        print(f"📁 生成的多样化启发式: {file_path}")
        print(f"📊 基于Portfolio: {len(population)} 个算法的策略分析")
        print(f"📄 完整记录: {record_file}")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    # ========================================================================
    # STAGE 1: 策略分析相关方法
    # ========================================================================
    
    def _load_portfolio_codes(self, population: List[Dict]) -> List[Dict]:
        """
        加载Portfolio中所有启发式的代码
        
        Returns:
            [{'name': ..., 'code': ..., 'performance': ...}, ...]
        """
        portfolio_codes = []
        
        for h in population:
            # 如果已经有code字段
            if 'code' in h and h['code']:
                code = h['code']
            # 否则从文件加载
            elif 'file' in h:
                code = self._load_heuristic_code_from_file(h['file'])
            elif 'name' in h:
                code = self._load_heuristic_code_from_name(h['name'])
            else:
                print(f"⚠️  警告: 无法加载启发式代码: {h}")
                continue
            
            portfolio_codes.append({
                'name': h.get('name', 'unknown'),
                'code': code,
                'avg_performance': h.get('avg_performance', 'N/A')
            })
        
        return portfolio_codes
    
    def _load_heuristic_code_from_file(self, filepath: str) -> str:
        """从文件路径加载代码"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️  加载文件失败 {filepath}: {e}")
            return ""
    
    def _load_heuristic_code_from_name(self, heuristic_name: str) -> str:
        """从启发式名称加载代码"""
        from src.run_hyper_heuristic.helper_function import load_heuristic_code
        return load_heuristic_code(self.problem, heuristic_name, self.heuristic_dir)
    
    def _create_analysis_prompt(self, portfolio_codes: List[Dict]) -> str:
        """
        创建Stage 1的分析提示词
        
        目标: 让LLM深度分析Portfolio的策略模式
        """
        # 构建Portfolio代码展示
        portfolio_display = ""
        for i, h in enumerate(portfolio_codes, 1):
            # 限制每个代码的长度，避免超出context window
            code_preview = h['code'][:800] + "\n..." if len(h['code']) > 800 else h['code']
            
            portfolio_display += f"\n{'='*60}\n"
            portfolio_display += f"算法 #{i}: {h['name']}\n"
            portfolio_display += f"平均性能: {h['avg_performance']}\n"
            portfolio_display += f"{'='*60}\n"
            portfolio_display += f"```python\n{code_preview}\n```\n"
        
        prompt_template = PromptTemplate(
            input_variables=[
                "task_description",
                "portfolio_size",
                "portfolio_display"
            ],
            template="""# Problem Context

{task_description}

# Task: Deep Analysis of Heuristic Portfolio Strategies

You are an expert in combinatorial optimization and algorithm design.

## Current Heuristic Portfolio

We have {portfolio_size} heuristic algorithms in the current portfolio:

{portfolio_display}

## Your Task: Strategy Analysis

Please perform a **comprehensive analysis** of these algorithms and identify opportunities for diversity improvement.

### Analysis Framework:

1. **Core Strategies Inventory**
   - What are the fundamental algorithmic strategies used? (e.g., greedy, random, nearest neighbor, insertion-based, local search, construction + improvement, etc.)
   - List each distinct strategy type found

2. **Common Patterns**
   - What do most algorithms have in common?
   - Are there dominant approaches? (e.g., >70% use the same strategy)
   - What are the shared structural patterns?

3. **Strategy Differentiation**
   - How do algorithms differ from each other?
   - What are the key variations? (e.g., different selection criteria, different improvement operators)
   - Which strategies are used by only 1-2 algorithms?

4. **Unexplored Strategy Space**
   - What major strategies are **NOT represented** in this portfolio?
   - What alternative approaches could be considered?
   - Examples of unexplored strategies might include:
     * Meta-heuristics: Simulated annealing, tabu search, genetic algorithms
     * Population-based: Ant colony, particle swarm
     * Advanced constructive: Savings algorithm variants, cluster-first-route-second
     * Hybrid approaches: Multi-start, iterated local search
     * Problem-specific insights: Geometric properties, mathematical properties

5. **Diversity Assessment**
   - On a scale of 1-10, how diverse is this portfolio? (1=all very similar, 10=very diverse)
   - What is the main weakness in terms of diversity?

## Response Format

Please structure your analysis in the following format:

***strategy_analysis:
1. CORE STRATEGIES USED:
- [List each strategy with count, e.g., "Greedy nearest neighbor: 7/10"]

2. COMMON PATTERNS:
- [Describe dominant patterns]

3. DIFFERENTIATION POINTS:
- [How algorithms differ]

4. UNEXPLORED STRATEGIES:
- [List strategies NOT present in portfolio]
- [Prioritize the most promising ones]

5. DIVERSITY SCORE: X/10
- MAIN WEAKNESS: [Brief description]

6. RECOMMENDATION FOR NEW ALGORITHM:
- [Suggest 2-3 specific strategies that would maximize diversity]
***

Be specific and actionable in your recommendations.
"""
        )
        
        return prompt_template.format(
            task_description=self.task_description,
            portfolio_size=len(portfolio_codes),
            portfolio_display=portfolio_display
        )
    
    def _call_llm_for_analysis(self, prompt: str) -> str:
        """调用LLM进行策略分析"""
        return self._call_llm(prompt, temperature=0.3)  # 低温度确保分析准确
    
    def _extract_summary_from_response(self, response: str) -> str:
        """
        从LLM响应中提取策略分析摘要
        
        查找 ***strategy_analysis: ... *** 标记
        """
        import re
        
        # 尝试提取 ***strategy_analysis: ... ***
        pattern = r'\*\*\*strategy_analysis:(.*?)\*\*\*'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # 如果没找到标记，返回整个响应
        return response
    
    # ========================================================================
    # STAGE 2: 算法生成相关方法
    # ========================================================================
    
    def _create_generation_prompt(self, strategy_summary: str) -> str:
        """
        创建Stage 2的生成提示词
        
        基于Stage 1的分析结果，生成新算法
        """
        prompt_template = PromptTemplate(
            input_variables=[
                "task_description",
                "strategy_summary"
            ],
            template="""# Problem Context

{task_description}

# Task: Design a Fundamentally Different Heuristic Algorithm

You are an expert in designing heuristic algorithms for combinatorial optimization problems.

## Portfolio Strategy Analysis Summary

Based on a deep analysis of the current heuristic portfolio, we have identified the following:

{strategy_summary}

## Your Task: Design a Diverse Algorithm

Based on the analysis above, especially the **UNEXPLORED STRATEGIES** and **RECOMMENDATION** sections, please design a **completely new** heuristic algorithm that:

### Requirements:

1. **Fundamental Difference**
   - Must use strategies from the "UNEXPLORED" list
   - Must NOT replicate the "COMMON PATTERNS" identified
   - Should be structurally different from existing algorithms

2. **Correctness**
   - Must follow the standard function signature
   - Must produce valid solutions
   - Must handle edge cases properly

3. **Quality**
   - Should leverage problem-specific insights
   - Should be efficient and practical
   - Should explore a genuinely different part of the solution space

### Design Principles:

- If most algorithms use "greedy nearest neighbor", avoid that
- If no algorithms use "random + local search", consider that
- If no algorithms use problem-specific heuristics (e.g., angle-based, density-based), explore those
- Think outside the box while remaining practical

### Example Unexplored Strategies (if applicable):

- **Random + Multi-start**: Random construction + repeated local improvements
- **Cluster-based**: Group nodes by proximity, then optimize within/between clusters
- **Angle-based selection**: Use geometric properties for node selection
- **Savings-based**: Calculate insertion savings like Clarke-Wright
- **Simulated annealing style**: Accept worse moves probabilistically
- **Beam search style**: Maintain multiple partial solutions
- **Problem-specific**: Leverage domain knowledge creatively

**IMPORTANT CODE FORMAT REQUIREMENTS:**

1. The function name must follow the pattern: `<strategy_name>_<random_4_chars>` (e.g., `diverse_hybrid_x3k9`)
2. Follow the exact code format shown in the examples
3. Ensure your code is complete and executable

## Response Format

Please respond with ONLY the Python code implementing your new diverse heuristic:

***python_code:
[Your complete implementation here]
***

**CRITICAL**: 
- Do NOT explain your reasoning in the code section
- Do NOT use markdown code blocks (```)
- Just put clean Python code between the *** markers
- Add comments in the code to explain key logic
- The function name MUST follow the pattern: <strategy_name>_<random_4_chars>

Please provide your new diverse heuristic algorithm now:"""
        )
        
        return prompt_template.format(
            task_description=self.task_description,
            strategy_summary=strategy_summary
        )
    
    def _call_llm_for_generation(self, prompt: str) -> str:
        """调用LLM生成新算法"""
        return self._call_llm(prompt, temperature=0.7)  # 高温度鼓励创造性
    
    # ========================================================================
    # 通用辅助方法
    # ========================================================================
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """调用LLM API"""
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in combinatorial optimization and algorithm design. You provide detailed, accurate, and actionable analysis and code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature
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


# 便捷函数 - 保持接口一致性
def diversity_improvement(
    population: List[Dict],
    api_key: str,
    problem: str,
    heuristic_dir: str,
    task_description_file: str,
    output_dir: str,
    model: str
) -> Tuple[str, str]:
    """
    DI工作流 - 向后兼容的便捷函数
    
    Args:
        population: 当前启发式种群
        api_key: LLM API密钥
        problem: 问题类型
        heuristic_dir: 启发式目录
        task_description_file: 任务描述文件路径
        output_dir: 输出目录
        model: LLM模型名称
    
    Returns:
        (file_path, code): 生成的启发式文件路径和代码
    """
    operator = DIOperator(
        problem=problem,
        heuristic_dir=heuristic_dir,
        task_description_file=task_description_file,
        output_dir=output_dir,
        api_key=api_key,
        model=model
    )
    
    return operator.generate(population)