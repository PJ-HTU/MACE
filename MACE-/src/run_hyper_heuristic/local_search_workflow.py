import os
import re
import sys
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import search_file
from src.run_hyper_heuristic.run_hyper_heuristic import run_hyper_heuristic, evaluate_all_heuristics
from src.run_hyper_heuristic.helper_function import select_complementary_pair, load_heuristic_code, extract_code_from_response, save_generated_heuristic, select_parent_for_local_search
from src.run_hyper_heuristic.prompt_cs import CSPromptFormatter, generate_complementary_heuristic, call_deepseek_reasoner
from src.run_hyper_heuristic.prompt_ls import LSPromptFormatter, generate_complementary_ls_heuristic
from src.run_hyper_heuristic.smoke_test import SmokeTestRunner,standalone_smoke_test
import json
from typing import Dict, List, Tuple, Any

def local_search_workflow(
    results_dict,
    api_key,
    problem,
    heuristic_dir,
    task_description_file,
    output_dir,
    model
) -> Tuple[str, str]:
    print("\n" + "=" * 80)
    print("✅ ls工作流开始!")
     
    # Step 1: 生成LS提示词（包含任务描述）
    ls_prompt, h_name = generate_complementary_ls_heuristic(
        results_dict=results_dict,
        problem=problem,
        heuristic_dir=heuristic_dir,
        task_description_file=task_description_file,
        save_prompt=True
    )
    
    # Save prompt for debugging
    prompt_file = f"{output_dir}/ls_prompt_latest.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(ls_prompt)
    
    # Step 2: 调用DeepSeek API
    llm_response = call_deepseek_reasoner(
        prompt=ls_prompt,
        api_key=api_key,
        model=model
    )
    
    # Step 3: 提取代码
    extracted_code = extract_code_from_response(llm_response)
    
    if not extracted_code:
        print("✗ 未能从响应中提取代码")
        print("\n完整响应:")
        print(llm_response)
        return None, None

    # Step 4: 保存代码
    try:
        file_path = save_generated_heuristic(extracted_code, output_dir=output_dir)
        
        # 保存完整响应（包括思考过程）
        response_file = file_path.replace('.py', '_full_response.txt')
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(f"Prompt used:\n{'-'*80}\n{ls_prompt}\n\n")
            f.write(f"LLM Response:\n{'-'*80}\n{llm_response}")
        print(f"✓ 完整响应已保存到: {response_file}")
        
    except Exception as e:
        print(f"✗ 保存失败: {str(e)}")
        return None, extracted_code
    
    # 完成
    print("\n" + "=" * 80)
    print("✅ ls工作流完成!")
    
    return file_path, extracted_code
