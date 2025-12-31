"""
smoke_test.py - 终极修复版本 + 多问题类型支持

处理所有可能的代码格式问题:
- ```python 标记
- *** 标记
- 嵌套的代码块

新增: 支持多问题类型 (tsp, jssp, cvrp, psp)
"""

from openai import OpenAI
import os
import traceback
from typing import Tuple, Optional
import re
from src.run_hyper_heuristic.run_hyper_heuristic import run_hyper_heuristic
from src.run_hyper_heuristic.helper_function import extract_code_from_response, save_generated_heuristic


class SmokeTestRunner:
    
    def __init__(
        self,
        problem: str = None,  # ✅ 新增: 问题类型参数，默认tsp保持向后兼容
        max_fix_attempts: int = 3,
        api_key: str = None
    ):
        self.problem = problem  # ✅ 保存问题类型
        self.max_fix_attempts = max_fix_attempts
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    
    def _clean_code_aggressive(self, code: str) -> str:
        """
        🆕 超强代码清理 - 处理所有已知格式问题
        """
        original_code = code
        
        # 1. 移除 ***python_code: 和 *** 标记
        code = re.sub(r'\*\*\*python_code:\s*\n?', '', code)
        code = re.sub(r'\n?\*\*\*\s*$', '', code)
        code = re.sub(r'^\*\*\*\s*\n?', '', code)
        
        # 2. 移除 ```python 和 ``` 标记
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'```$', '', code)
        
        # 3. 移除可能的嵌套情况 (***python_code: 里面嵌套 ```python)
        # 先移除外层***，再移除内层```
        if '***' in code or '```' in code:
            # 多次清理直到没有这些标记
            max_iterations = 5
            for _ in range(max_iterations):
                old_code = code
                code = re.sub(r'\*\*\*python_code:\s*\n?', '', code)
                code = re.sub(r'\n?\*\*\*\s*', '', code)
                code = re.sub(r'```python\s*\n?', '', code)
                code = re.sub(r'```\s*', '', code)
                if code == old_code:
                    break
        
        # 4. 清理首尾空白
        code = code.strip()
        
        # 5. 验证清理效果
        if '```' in code or '***' in code:
            print(f"⚠️  警告: 清理后仍包含标记!")
            print(f"原始长度: {len(original_code)}, 清理后长度: {len(code)}")
            # 显示前200字符用于调试
            print(f"清理后前200字符:\n{code[:200]}")
        
        return code
    
    def run_smoke_test(self, heuristic_file) -> Tuple[bool, Optional[str]]:
        """运行烟雾测试"""
        try:
            # 提取启发式名称
            heuristic_name = os.path.splitext(os.path.basename(heuristic_file))[0]
            
            print(f"\n{'='*80}")
            print(f"🧪 运行Smoke Test: {heuristic_name}")
            print(f"📋 问题类型: {self.problem}")  # ✅ 显示问题类型
            print(f"{'='*80}")
            
            # 🆕 在测试前先清理文件
            with open(heuristic_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 检查是否包含任何标记
            needs_cleaning = False
            if '```' in code:
                print(f"⚠️  检测到Markdown标记 (```)")
                needs_cleaning = True
            if '***' in code:
                print(f"⚠️  检测到自定义标记 (***)")
                needs_cleaning = True
            
            if needs_cleaning:
                print(f"正在清理代码...")
                cleaned_code = self._clean_code_aggressive(code)
                # 保存清理后的代码
                with open(heuristic_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_code)
                print(f"✓ 代码已清理并保存")
            
            # ✅ 测试参数 - 使用动态问题类型
            problem = self.problem  # ✅ 使用实例的问题类型
            test_data = "smoke_data"
            llm_config_file = os.path.join("output", "llm_config", "azure_gpt_4o.json")
            heuristic_dir = "basic_heuristics"
            iterations_scale_factor = 1.0
            result_dir = "result"
            
            # 运行测试
            validation_result = run_hyper_heuristic(
                problem=problem,  # ✅ 传入正确的问题类型
                heuristic=heuristic_name,
                llm_config_file=llm_config_file,
                heuristic_dir=heuristic_dir,
                test_data=test_data,
                iterations_scale_factor=iterations_scale_factor,
                result_dir=result_dir
            )
            
            # 验证结果
            validation_symbol = True
            for i in validation_result:
                try:
                    if i > 0:
                        pass
                except:
                    validation_symbol = False
            
            if validation_symbol:
                print(f"✅ Smoke Test 通过!")
                return True, None
            else:
                error_msg = f"验证失败: 结果包含无效值"
                print(f"❌ Smoke Test 失败:\n{error_msg}")
                return False, error_msg
                
        except SyntaxError as e:
            error_msg = f"SyntaxError: {str(e)}"
            print(f"❌ Smoke Test 失败:\n{error_msg}")
            print(f"\n堆栈追踪:\n{traceback.format_exc()}")
            return False, error_msg
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"❌ Smoke Test 失败:\n{error_msg}")
            print(f"\n堆栈追踪:\n{traceback.format_exc()}")
            return False, error_msg
    
    def fix_code_with_llm(
        self,
        original_code: str,
        error_message: str,
        attempt: int
    ) -> Optional[str]:
        """使用LLM修复代码"""
        print(f"\n{'='*80}")
        print(f"🔧 尝试修复 (第 {attempt}/{self.max_fix_attempts} 次)")
        print(f"{'='*80}")
        
        # 🆕 更严格的提示词
        fix_prompt = f"""You are an expert Python debugger. Fix this code.

Original Code:
{original_code}

Error:
{error_message}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY executable Python code
2. NO markdown formatting (no ```, no ***, no markdown blocks)
3. NO explanations, NO comments outside the code
4. Start directly with "from" or "import" or "def"
5. The response must be valid Python that can be executed immediately

Fixed code:"""
        
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            print("正在调用LLM修复代码...")
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Python code fixer. Respond ONLY with executable Python code. No markdown, no explanations, no formatting. Just pure Python code."
                    },
                    {
                        "role": "user",
                        "content": fix_prompt
                    }
                ],
                temperature=0.2  # 更低的温度以获得更确定的输出
            )
            
            llm_response = response.choices[0].message.content
            
            # 🆕 超强清理流程
            # 1. 先尝试标准提取
            fixed_code = extract_code_from_response(llm_response)
            
            # 2. 如果失败,使用超强清理
            if not fixed_code:
                fixed_code = self._clean_code_aggressive(llm_response)
            else:
                # 即使提取成功,也要再次清理
                fixed_code = self._clean_code_aggressive(fixed_code)
            
            if fixed_code:
                # 3. 最终验证
                if '```' in fixed_code or '***' in fixed_code:
                    print("⚠️  警告: 代码中仍包含标记,执行最终清理...")
                    fixed_code = self._clean_code_aggressive(fixed_code)
                
                # 4. 检查是否是有效的Python代码开头
                valid_starts = ['from ', 'import ', 'def ', 'class ', '#']
                if not any(fixed_code.lstrip().startswith(s) for s in valid_starts):
                    print(f"⚠️  警告: 代码开头不符合预期")
                    print(f"代码前50字符: {fixed_code[:50]}")
                
                print("✓ LLM已生成修复代码")
                return fixed_code
            else:
                print("✗ 无法从LLM响应中提取代码")
                print(f"响应前200字符: {llm_response[:200]}")
                return None
                
        except Exception as e:
            print(f"✗ LLM修复失败: {str(e)}")
            traceback.print_exc()
            return None
    
    def test_and_fix_cycle(
        self,
        heuristic_file: str,
        save_fixed: bool = True
    ) -> Tuple[bool, str]:
        """完整的测试-修复循环"""
        current_file = heuristic_file
        
        # 第一次测试
        success, error_msg = self.run_smoke_test(current_file)
        
        if success:
            return True, current_file
        
        # 进入修复循环
        for attempt in range(1, self.max_fix_attempts + 1):
            # 读取当前代码
            with open(current_file, 'r', encoding='utf-8') as f:
                current_code = f.read()
            
            # 🆕 先超强清理当前代码
            current_code = self._clean_code_aggressive(current_code)
            
            # 让LLM修复
            fixed_code = self.fix_code_with_llm(current_code, error_msg, attempt)
            
            if not fixed_code:
                print(f"❌ 修复失败,放弃")
                return False, current_file
            
            # 保存修复后的代码
            if save_fixed:
                fixed_file = save_generated_heuristic(
                    fixed_code,
                    output_dir=os.path.dirname(current_file)
                )
                print(f"✓ 代码已保存到: {fixed_file}")
                current_file = fixed_file
            else:
                # 覆盖原文件
                with open(current_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_code)
                print(f"✓ 代码已保存到: {current_file}")
            
            # 再次测试
            success, error_msg = self.run_smoke_test(current_file)
            
            if success:
                print(f"\n{'='*80}")
                print(f"🎉 修复成功! (尝试 {attempt} 次后通过)")
                print(f"{'='*80}")
                return True, current_file
        
        # 所有尝试都失败
        print(f"\n{'='*80}")
        print(f"❌ 放弃: 经过 {self.max_fix_attempts} 次修复尝试仍未通过")
        print(f"{'='*80}")
        return False, current_file


def standalone_smoke_test(
    heuristic_file: str, 
    problem: str = None,  # ✅ 新增: 问题类型参数，默认tsp保持向后兼容
    api_key: str = None
) -> bool:
    """
    独立的烟雾测试函数
    
    Args:
        heuristic_file: 启发式文件路径
        problem: 问题类型 (tsp, jssp, cvrp, psp)，默认 "tsp"
        api_key: API密钥
    
    Returns:
        bool: 测试是否通过
    
    Example:
        # TSP问题
        success = standalone_smoke_test("heuristic.py", problem="tsp")
        
        # JSSP问题
        success = standalone_smoke_test("heuristic.py", problem="jssp")
    """
    tester = SmokeTestRunner(
        problem=problem,  # ✅ 传入问题类型
        max_fix_attempts=3,
        api_key=api_key
    )
    
    success, _ = tester.test_and_fix_cycle(heuristic_file)
    return success