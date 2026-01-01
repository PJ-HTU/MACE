"""
DI Operator (Diversity Improvement Operator)
"""

import os
import re
from typing import Dict, List, Tuple
from openai import OpenAI


class DIOperator:
    
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
        
        self.task_description = self._load_task_description()
    
    def _load_task_description(self) -> str:
        """Load task description file."""
        if self.task_description_file and os.path.exists(self.task_description_file):
            with open(self.task_description_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def generate(self, population: List[Dict]) -> Tuple[str, str]:
        """
        Execute DI operator - Two-stage (fixed version).
        
        Fix: Stage 1 → Stage 2 passes compact summary only, not full analysis
        """
        print("\n" + "=" * 80)
        print("🌈 [DI Operator] Diversity Improvement")
        print("=" * 80)
        print(f"Portfolio size: {len(population)}")
        
        # Stage 1: Analyze Portfolio → Extract compact summary
        print("\n📊 Stage 1: Analyzing strategy patterns...")
        
        portfolio_codes = self._load_portfolio_codes(population)
        
        analysis_prompt = self._create_analysis_prompt(portfolio_codes)
        full_analysis = self._call_llm(analysis_prompt, temperature=0.7)
        
        # Key fix: Extract only compact summary, not full analysis
        compact_summary = self._extract_compact_summary(full_analysis)

        # Stage 2: Generate algorithm (using compact summary)
        print("\n🎨 Stage 2: Generating diverse algorithm...")
        
        generation_prompt = self._create_simplified_generation_prompt(compact_summary)
        
        llm_response = self._call_llm(generation_prompt, temperature=0.8)
        
        extracted_code = self._extract_code_from_response(llm_response)
        
        if not extracted_code:
            print("✗ Failed to extract code")
            return None, None
        
        try:
            file_path = self._save_generated_heuristic(extracted_code)
            
            base_name = os.path.basename(file_path).replace('.py', '')
            record_file = os.path.join(self.output_dir, f"{base_name}_record.txt")
            
            with open(record_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("DI Operator - Fixed Version Record\n")
                f.write("="*80 + "\n\n")
                
                f.write("STAGE 1 Full Analysis:\n")
                f.write("-"*80 + "\n")
                f.write(full_analysis + "\n\n")
                
                f.write("Extracted Compact Summary:\n")
                f.write("-"*80 + "\n")
                f.write(compact_summary + "\n\n")
                
                f.write("="*80 + "\n")
                f.write("STAGE 2 Generation Prompt:\n")
                f.write("-"*80 + "\n")
                f.write(generation_prompt + "\n\n")
                
                f.write("Generated Response:\n")
                f.write("-"*80 + "\n")
                f.write(llm_response)
            
            print(f"✓ Code: {file_path}")
            print(f"✓ Record: {record_file}")
            
        except Exception as e:
            print(f"✗ Save failed: {e}")
            return None, extracted_code
        
        print("\n" + "=" * 80)
        print("✅ [DI Operator] Completed!")
        print("=" * 80 + "\n")
        
        return file_path, extracted_code
    
    # Stage 1: Analysis methods
    
    def _load_portfolio_codes(self, population: List[Dict]) -> List[Dict]:
        """Load portfolio codes."""
        portfolio_codes = []
        
        for h in population:
            if 'code' in h and h['code']:
                code = h['code']
            elif 'file' in h:
                code = self._load_heuristic_code_from_file(h['file'])
            elif 'name' in h:
                code = self._load_heuristic_code_from_name(h['name'])
            else:
                print(f"⚠️  Cannot load: {h}")
                continue
            
            portfolio_codes.append({
                'name': h.get('name', 'unknown'),
                'code': code,
                'avg_performance': h.get('avg_performance', 'N/A')
            })
        
        return portfolio_codes
    
    def _create_analysis_prompt(self, portfolio_codes: List[Dict]) -> str:
        """
        Create Stage 1 analysis prompt.
        
        Requires LLM to output structured compact summary.
        """
        portfolio_display = ""
        for i, h in enumerate(portfolio_codes, 1):
            code_preview = h['code'][:600] + "\n..." if len(h['code']) > 600 else h['code']
            
            portfolio_display += f"\n{'='*60}\n"
            portfolio_display += f"Algorithm #{i}: {h['name']}\n"
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
        Extract compact summary from full analysis.
        
        Core fix: Extract only structured key information.
        """
        pattern = r'\*\*\*summary_start(.*?)\*\*\*summary_end'
        matches = re.findall(pattern, full_analysis, re.DOTALL)
        
        if matches:
            summary = matches[0].strip()
            return summary
        
        print("⚠️  Structured summary not found, using smart extraction")
        
        lines = full_analysis.split('\n')
        key_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in 
                   ['common', 'strategies', 'unexplored', 'recommendation', 
                    'greedy', 'nearest', 'insertion', 'cluster', 'savings']):
                key_lines.append(line)
        
        if key_lines:
            summary = '\n'.join(key_lines[:10])
            return summary[:500]
        
        return full_analysis[:300]
    
    # Stage 2: Generation methods
    
    def _create_simplified_generation_prompt(self, compact_summary: str) -> str:
        """
        Create simplified Stage 2 prompt.
        
        Core fix: Use only compact summary, not full analysis.
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
    
    # Helper methods
    
    def _load_heuristic_code_from_file(self, filepath: str) -> str:
        """Load code from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️  Load failed {filepath}: {e}")
            return ""
    
    def _load_heuristic_code_from_name(self, heuristic_name: str) -> str:
        """Load code from name."""
        from src.run_hyper_heuristic.helper_function import load_heuristic_code
        return load_heuristic_code(self.problem, heuristic_name, self.heuristic_dir)
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call LLM API."""
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
            print(f"✗ API call failed: {e}")
            raise
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from response."""
        pattern = r'\*\*\*python_code:(.*?)\*\*\*'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        pattern = r'```python(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return response.strip()
    
    def _save_generated_heuristic(self, code: str) -> str:
        """Save generated code."""
        from src.run_hyper_heuristic.helper_function import save_generated_heuristic
        return save_generated_heuristic(code, output_dir=self.output_dir)


def diversity_improvement(
    population: List[Dict],
    api_key: str,
    problem: str,
    heuristic_dir: str,
    task_description_file: str,
    output_dir: str,
    model: str
) -> Tuple[str, str]:
    """DI operator convenience function."""
    operator = DIOperator(
        problem=problem,
        heuristic_dir=heuristic_dir,
        task_description_file=task_description_file,
        output_dir=output_dir,
        api_key=api_key,
        model=model
    )
    
    return operator.generate(population)