from openai import OpenAI
import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import search_file
import json
from src.run_hyper_heuristic.run_hyper_heuristic import run_hyper_heuristic, evaluate_all_heuristics
from src.run_hyper_heuristic.helper_function import select_complementary_pair, load_heuristic_code, extract_code_from_response, save_generated_heuristic
from langchain_core.prompts import PromptTemplate
import re
import sys
import traceback
from typing import Tuple, Optional, Dict

class SmokeTestRunner:
    
    def __init__(
        self,
        max_fix_attempts: int = 3,
        api_key: str = None

    ):
        self.max_fix_attempts = max_fix_attempts
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        
    def run_smoke_test(self, heuristic_file) -> Tuple[bool, Optional[str]]:
       
        try:
            # 动态导入问题环境
            import importlib.util
            
            # 加载环境类
            from src.problems.tsp.env import Env
            from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
            problem = 'tsp'
            # 提取启发式名称
            heuristic_name = os.path.splitext(os.path.basename(heuristic_file))[0]
            test_data = "smoke_data" 
            print(f"\n{'='*60}")
            print(f"🧪 运行Smoke Test: {heuristic_name}")
            print(f"{'='*60}")
            
            llm_config_file = os.path.join("output", "llm_config", "azure_gpt_4o.json")  
            heuristic_dir = "basic_heuristics" 
            iterations_scale_factor = 1.0 
            result_dir = "result"  
            
            validation_result = run_hyper_heuristic(
                    problem=problem,
                    heuristic=heuristic_name,
                    llm_config_file=llm_config_file,
                    heuristic_dir=heuristic_dir,
                    test_data=test_data,
                    iterations_scale_factor=iterations_scale_factor,
                    result_dir=result_dir
                )
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
                error_msg = "验证失败:生成的解决方案无效"
                print(f"❌ Smoke Test 失败: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n\n堆栈追踪:\n{traceback.format_exc()}"
            print(f"❌ Smoke Test 失败:")
            print(error_msg)
            return False, error_msg
    
    def fix_code_with_llm(
        self,
        original_code: str,
        error_message: str,
        attempt: int
    ) -> Optional[str]:
        """
        使用LLM修复代码
        
        Args:
            original_code: 原始代码
            error_message: 错误信息
            attempt: 第几次尝试
            
        Returns:
            修复后的代码,如果失败返回None
        """
        print(f"\n{'='*60}")
        print(f"🔧 尝试修复 (第 {attempt}/{self.max_fix_attempts} 次)")
        print(f"{'='*60}")
        
        # 构建修复prompt
        fix_prompt = f"""You are debugging a Python heuristic algorithm for the Port Scheduling Problem.

The following code has an error:

```python
{original_code}
```

**Error Message:**
```
{error_message}
```

Please fix the code. 

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

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            print("正在调用DeepSeek修复代码...")
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1",  # 使用chat模型更快
                messages=[
                    {"role": "system", "content": "You are an expert Python debugger."},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.3  # 降低温度以获得更确定的修复
            )
            
            llm_response = response.choices[0].message.content
            
            # 提取修复后的代码
            fixed_code = extract_code_from_response(llm_response)
            
            if fixed_code:
                print("✓ LLM已生成修复代码")
                return fixed_code
            else:
                print("✗ 无法从LLM响应中提取代码")
                return None
                
        except Exception as e:
            print(f"✗ LLM修复失败: {str(e)}")
            return None
    
    def test_and_fix_cycle(
        self,
        heuristic_file: str,
        save_fixed: bool = True
    ) -> Tuple[bool, str]:
        """
        完整的测试-修复循环
        
        Args:
            heuristic_file: 启发式文件路径
            save_fixed: 是否保存修复后的代码
            
        Returns:
            (success, final_file_path): 是否成功和最终文件路径
        """
        
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
                current_file = fixed_file
            else:
                # 临时保存
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.py',
                    delete=False,
                    encoding='utf-8'
                )
                temp_file.write(fixed_code)
                temp_file.close()
                current_file = temp_file.name
            
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


def standalone_smoke_test(heuristic_file: str, api_key: str = None) -> bool:

    tester = SmokeTestRunner(
        max_fix_attempts=3,
        api_key=api_key
    )
    
    success, _ = tester.test_and_fix_cycle(heuristic_file)
    return success