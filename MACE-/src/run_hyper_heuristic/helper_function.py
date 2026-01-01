from openai import OpenAI
import os
import re
import sys
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import search_file
from src.run_hyper_heuristic.run_hyper_heuristic import run_hyper_heuristic, evaluate_all_heuristics
import json
import numpy as np
from typing import Dict, List, Tuple, Any

def select_complementary_pair(results_dict: Dict) -> Tuple[str, str]:
    """
    从结果字典中选择互补性强且未被选择过的两个启发式（加权随机选择）
    
    Args:
        results_dict: 字典，key是启发式名称，value是该启发式在各实例上的结果列表
    Returns:
        (h1, h2): 互补性强的两个启发式名称
    """
    heuristic_names = list(results_dict.keys())
    n = len(heuristic_names)
    
    # 存储所有候选对及其距离
    candidates = []
    
    # 遍历所有启发式对
    for i in range(n):
        for j in range(i + 1, n):
            h1 = heuristic_names[i]
            h2 = heuristic_names[j]
            
            # 创建标准化的对（字典序排序，确保 (A,B) 和 (B,A) 被视为相同）
            pair = tuple(sorted([h1, h2]))

            feability = True
            penalty_value = 1e10-1
            for h_each_result in results_dict[h1]:
                if h_each_result > penalty_value:
                    feability = False
            for h_each_result in results_dict[h2]:
                if h_each_result > penalty_value:
                    feability = False
            if feability:
                # 计算曼哈顿距离
                perf1 = np.array(results_dict[h1])
                perf2 = np.array(results_dict[h2])
                distance = np.sum(np.abs(perf1 - perf2))
                
                candidates.append((distance, h1, h2, pair))
    
    
    # 按距离降序排序
    candidates.sort(reverse=True, key=lambda x: x[0])
    
    # 提取距离和对应的索引
    distances = np.array([c[0] for c in candidates])
    
    # 基于距离的排名（距离越大，排名越前）
    ranks = np.arange(1, len(candidates) + 1)
    
    # 计算选择概率（距离大的对有更高概率被选中）
    weights = 1.0 / ranks
    probabilities = weights / np.sum(weights)
    
    # 加权随机选择
    selected_idx = np.random.choice(len(candidates), p=probabilities)
    distance, h1, h2, pair = candidates[selected_idx]
    
    return h1, h2

def select_parent_for_local_search(results_dict: dict) -> Tuple[str, List[float]]:
    heuristics = []
    avg_scores = []
    
    # Calculate average scores
    for h_name, h_scores in results_dict.items():

        feability = True
        penalty_value = 1e10-1
        for h_each_result in h_scores:
            if h_each_result > penalty_value:
                feability = False
        if feability:
            heuristics.append(h_name)
            avg_scores.append(np.mean(h_scores))
    
    # Rank by performance (lower score is better)
    avg_scores = np.array(avg_scores)
    ranks = np.argsort(avg_scores) + 1  # 1 = best, n = worst
    
    # Calculate selection probabilities (inverse of rank)
    weights = 1.0 / ranks
    probabilities = weights / np.sum(weights)
    
    # Weighted random selection
    selected_idx = np.random.choice(len(heuristics), p=probabilities)
    selected_name = heuristics[selected_idx]
    selected_scores = results_dict[selected_name]
    
    return selected_name

def load_heuristic_code(problem: str, heuristic_name: str, heuristic_dir: str = "basic_heuristics") -> str:

    heuristic_file = os.path.join(
        "src", "problems", problem, "heuristics", 
        heuristic_dir, f"{heuristic_name}.py"
    )
    
    if not os.path.exists(heuristic_file):
        raise FileNotFoundError(f"找不到启发式文件: {heuristic_file}")
    
    with open(heuristic_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    return code


def extract_code_from_response(llm_response: str) -> str:

    pattern = r'\*\*\*python_code:(.*?)\*\*\*'
    match = re.search(pattern, llm_response, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
        return code
    else:
        code_block_pattern = r'```python(.*?)```'
        match = re.search(code_block_pattern, llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def save_generated_heuristic(code: str, output_dir: str = "generated_heuristics"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 从代码中提取函数名
    func_name_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    match = re.search(func_name_pattern, code)
    
    if match:
        func_name = match.group(1)
        file_path = os.path.join(output_dir, f"{func_name}.py")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"✓ 代码已保存到: {file_path}")
        return file_path
    else:
        raise ValueError("无法从代码中提取函数名")