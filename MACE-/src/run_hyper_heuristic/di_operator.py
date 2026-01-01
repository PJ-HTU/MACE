"""
DI Operator (Diversity Improvement Operator) - FIXED VERSION
修复版DI算子 - 精简Stage 2的prompt

核心修复:
1. Stage 1: 分析Portfolio → 提取**核心策略关键词** (不是完整分析)
2. Stage 2: 只用简短的关键词生成 → prompt大幅缩减

修复前问题:
- Stage 1返回长篇分析 (1000+ tokens)
- Stage 2把完整分析塞入prompt → 导致prompt过长

修复后:
- Stage 1只提取: "使用的策略: X, Y, Z" + "未探索: A, B, C"
- Stage 2 prompt精简到 ~300 tokens
"""

import os
import re
from typing import Dict, List, Tuple
from openai import OpenAI


class DIOperator:
    """
    DI (Diversity Improvement) 算子 - 修复版
    
    关键改进: 精简Stage 2的prompt输入
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
        执行DI算子 - 两阶段(修复版)
        
        修复: Stage 1 → Stage 2只传递精简摘要,不是完整分析
        """
        print("\n" + "=" * 80)
        print("🌈 [DI算子] Diversity Improvement")
        print("=" * 80)
        print(f"Portfolio大小: {len(population)}")
        
        # ============================================================
        # STAGE 1: 分析Portfolio → 提取**精简摘要**
        # ============================================================
        print("\n📊 Stage 1: 分析策略模式...")
        
        # 加载代码
        portfolio_codes = self._load_portfolio_codes(population)
        
        # LLM分析
        analysis_prompt = self._create_analysis_prompt(portfolio_codes)
        full_analysis = self._call_llm(analysis_prompt, temperature=0.7)
        
        # ✅ 关键修复: 只提取精简摘要,不传递完整分析!
        compact_summary = self._extract_compact_summary(full_analysis)

        # ============================================================
        # STAGE 2: 生成算法 (使用精简摘要)
        # ============================================================
        print("\n🎨 Stage 2: 生成多样化算法...")
        
        # ✅ 使用精简摘要,而非完整分析
        generation_prompt = self._create_simplified_generation_prompt(compact_summary)
        
        llm_response = self._call_llm(generation_prompt, temperature=0.8)
        
        # 提取代码
        extracted_code = self._extract_code_from_response(llm_response)
        
        if not extracted_code:
            print("✗ 未能提取代码")
            return None, None
        
        # 保存
        try:
            file_path = self._save_generated_heuristic(extracted_code)
            
            # 保存记录
            base_name = os.path.basename(file_path).replace('.py', '')
            record_file = os.path.join(self.output_dir, f"{base_name}_record.txt")
            
            with open(record_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("DI算子 - 修复版记录\n")
                f.write("="*80 + "\n\n")
                
                f.write("STAGE 1 完整分析:\n")
                f.write("-"*80 + "\n")
                f.write(full_analysis + "\n\n")
                
                f.write("提取的精简摘要:\n")
                f.write("-"*80 + "\n")
                f.write(compact_summary + "\n\n")
                
                f.write("="*80 + "\n")
                f.write("STAGE 2 生成Prompt:\n")
                f.write("-"*80 + "\n")
                f.write(generation_prompt + "\n\n")
                
                f.write("生成响应:\n")
                f.write("-"*80 + "\n")
                f.write(llm_response)
            
            print(f"✓ 代码: {file_path}")
            print(f"✓ 记录: {record_file}")
            
        except Exception as e:
            print(f"✗ 保存失败: {e}")
            return None, extracted_code
        
        print("\n" + "=" * 80)
        print("✅ [DI算子] 完成!")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    # ========================================================================
    # STAGE 1: 分析相关方法
    # ========================================================================
    
    def _load_portfolio_codes(self, population: List[Dict]) -> List[Dict]:
        """加载Portfolio代码"""
        portfolio_codes = []
        
        for h in population:
            # 从不同来源加载代码
            if 'code' in h and h['code']:
                code = h['code']
            elif 'file' in h:
                code = self._load_heuristic_code_from_file(h['file'])
            elif 'name' in h:
                code = self._load_heuristic_code_from_name(h['name'])
            else:
                print(f"⚠️  无法加载: {h}")
                continue
            
            portfolio_codes.append({
                'name': h.get('name', 'unknown'),
                'code': code,
                'avg_performance': h.get('avg_performance', 'N/A')
            })
        
        return portfolio_codes
    
    def _create_analysis_prompt(self, portfolio_codes: List[Dict]) -> str:
        """
        创建Stage 1分析prompt
        
        要求LLM输出**结构化**的精简摘要
        """
        # 限制每个代码预览长度
        portfolio_display = ""
        for i, h in enumerate(portfolio_codes, 1):
            code_preview = h['code'][:600] + "\n..." if len(h['code']) > 600 else h['code']
            
            portfolio_display += f"\n{'='*60}\n"
            portfolio_display += f"算法 #{i}: {h['name']}\n"
            portfolio_display += f"{'='*60}\n"
            portfolio_display += f"```python\n{code_preview}\n```\n"
        
        prompt = f"""# Task: Analyze Heuristic Portfolio Strategies

{self.task_description}

## Portfolio ({len(portfolio_codes)} algorithms):

{portfolio_display}

## Your Task:

Analyze these algorithms and provide a **STRUCTURED SUMMARY** in this EXACT format:

***summary_start
COMMON_STRATEGIES: [list the main strategies used by most algorithms, e.g., "greedy", "nearest_neighbor", "insertion"]
UNEXPLORED_STRATEGIES: [list strategies that NO algorithm uses, e.g., "clustering", "savings", "angle-based", "local_search"]
RECOMMENDATION: [1 sentence: which unexplored strategy would add most diversity]
***summary_end

**CRITICAL**: 
- Output MUST be wrapped in ***summary_start and ***summary_end
- Keep it CONCISE - no long explanations
- Focus on high-level strategy types, not implementation details

Provide the structured summary now:"""
        
        return prompt
    
    def _extract_compact_summary(self, full_analysis: str) -> str:
        """
        从完整分析中提取精简摘要
        
        ✅ 核心修复点: 只提取结构化的关键信息
        """
        # 尝试提取 ***summary_start ... ***summary_end
        pattern = r'\*\*\*summary_start(.*?)\*\*\*summary_end'
        matches = re.findall(pattern, full_analysis, re.DOTALL)
        
        if matches:
            summary = matches[0].strip()
            return summary
        
        # 如果没有标记,尝试智能提取前300字符
        print("⚠️  未找到结构化摘要,使用智能提取")
        
        # 查找关键段落
        lines = full_analysis.split('\n')
        key_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in 
                   ['common', 'strategies', 'unexplored', 'recommendation', 
                    'greedy', 'nearest', 'insertion', 'cluster', 'savings']):
                key_lines.append(line)
        
        if key_lines:
            summary = '\n'.join(key_lines[:10])  # 最多10行
            return summary[:500]  # 最多500字符
        
        # 兜底: 返回前300字符
        return full_analysis[:300]
    
    # ========================================================================
    # STAGE 2: 生成相关方法
    # ========================================================================
    
    def _create_simplified_generation_prompt(self, compact_summary: str) -> str:
        """
        创建精简的Stage 2 prompt
        
        ✅ 核心修复: 只用精简摘要,不塞完整分析
        """
        prompt = f"""# Task: Design a Diverse Heuristic

{self.task_description}

## Portfolio Analysis Summary:

{compact_summary}

## Your Task:

Based on the UNEXPLORED_STRATEGIES above, design a **COMPLETELY DIFFERENT** heuristic algorithm.

### Requirements:
1. Must use strategies from UNEXPLORED list
2. Must NOT use strategies from COMMON list
3. Must be correct and executable

### Output Format:
***python_code:
[Your complete code]
***

**Function name pattern**: `<strategy>_<4random_chars>`

Generate now:"""
        
        return prompt
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _load_heuristic_code_from_file(self, filepath: str) -> str:
        """从文件加载代码"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️  加载失败 {filepath}: {e}")
            return ""
    
    def _load_heuristic_code_from_name(self, heuristic_name: str) -> str:
        """从名称加载代码"""
        from src.run_hyper_heuristic.helper_function import load_heuristic_code
        return load_heuristic_code(self.problem, heuristic_name, self.heuristic_dir)
    
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
                        "content": "You are an expert in combinatorial optimization. Provide concise, structured analysis and diverse algorithm designs."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"✗ API调用失败: {e}")
            raise
    
    def _extract_code_from_response(self, response: str) -> str:
        """从响应提取代码"""
        # 尝试 ***python_code: ... ***
        pattern = r'\*\*\*python_code:(.*?)\*\*\*'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # 尝试 ```python ... ```
        pattern = r'```python(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return response.strip()
    
    def _save_generated_heuristic(self, code: str) -> str:
        """保存代码"""
        from src.run_hyper_heuristic.helper_function import save_generated_heuristic
        return save_generated_heuristic(code, output_dir=self.output_dir)


# 便捷函数
def diversity_improvement(
    population: List[Dict],
    api_key: str,
    problem: str,
    heuristic_dir: str,
    task_description_file: str,
    output_dir: str,
    model: str
) -> Tuple[str, str]:
    """DI算子便捷函数"""
    operator = DIOperator(
        problem=problem,
        heuristic_dir=heuristic_dir,
        task_description_file=task_description_file,
        output_dir=output_dir,
        api_key=api_key,
        model=model
    )
    
    return operator.generate(population)