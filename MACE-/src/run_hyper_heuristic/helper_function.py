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
    Select two heuristics with strong complementarity using weighted random selection.
    
    Args:
        results_dict: Dictionary with heuristic names as keys and result lists as values
        
    Returns:
        (h1, h2): Two heuristic names with strong complementarity
    """
    heuristic_names = list(results_dict.keys())
    n = len(heuristic_names)
    
    candidates = []
    
    for i in range(n):
        for j in range(i + 1, n):
            h1 = heuristic_names[i]
            h2 = heuristic_names[j]
            
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
                perf1 = np.array(results_dict[h1])
                perf2 = np.array(results_dict[h2])
                distance = np.sum(np.abs(perf1 - perf2))
                
                candidates.append((distance, h1, h2, pair))
    
    candidates.sort(reverse=True, key=lambda x: x[0])
    
    distances = np.array([c[0] for c in candidates])
    ranks = np.arange(1, len(candidates) + 1)
    
    weights = 1.0 / ranks
    probabilities = weights / np.sum(weights)
    
    selected_idx = np.random.choice(len(candidates), p=probabilities)
    distance, h1, h2, pair = candidates[selected_idx]
    
    return h1, h2


def select_parent_for_local_search(results_dict: dict) -> Tuple[str, List[float]]:
    """
    Select parent heuristic for local search using weighted random selection.
    
    Args:
        results_dict: Dictionary with heuristic names and their performance scores
        
    Returns:
        selected_name: Name of selected heuristic
    """
    heuristics = []
    avg_scores = []
    
    for h_name, h_scores in results_dict.items():
        feability = True
        penalty_value = 1e10-1
        for h_each_result in h_scores:
            if h_each_result > penalty_value:
                feability = False
        if feability:
            heuristics.append(h_name)
            avg_scores.append(np.mean(h_scores))
    
    avg_scores = np.array(avg_scores)
    ranks = np.argsort(avg_scores) + 1
    
    weights = 1.0 / ranks
    probabilities = weights / np.sum(weights)
    
    selected_idx = np.random.choice(len(heuristics), p=probabilities)
    selected_name = heuristics[selected_idx]
    
    return selected_name


def load_heuristic_code(problem: str, heuristic_name: str, heuristic_dir: str = "basic_heuristics") -> str:
    """
    Load heuristic code from file.
    
    Args:
        problem: Problem type
        heuristic_name: Heuristic name
        heuristic_dir: Heuristic directory
        
    Returns:
        code: Heuristic code content
    """
    heuristic_file = os.path.join(
        "src", "problems", problem, "heuristics", 
        heuristic_dir, f"{heuristic_name}.py"
    )
    
    if not os.path.exists(heuristic_file):
        raise FileNotFoundError(f"Heuristic file not found: {heuristic_file}")
    
    with open(heuristic_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    return code


def extract_code_from_response(llm_response: str) -> str:
    """
    Extract code from LLM response.
    
    Args:
        llm_response: LLM response text
        
    Returns:
        code: Extracted code, or None if not found
    """
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
    """
    Save generated heuristic code to file.
    
    Args:
        code: Code content to save
        output_dir: Output directory
        
    Returns:
        file_path: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    func_name_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    match = re.search(func_name_pattern, code)
    
    if match:
        func_name = match.group(1)
        file_path = os.path.join(output_dir, f"{func_name}.py")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"✓ Code saved to: {file_path}")
        return file_path
    else:
        raise ValueError("Cannot extract function name from code")