"""
Tool Library Builder - Generates problem tool library through LLM.

This module implements the Tool Library Builder component in MACE Stage One,
generating domain-specific helper functions via LLM to encapsulate complex domain logic.
"""

import os
import importlib
import traceback
import re
from typing import List, Dict, Tuple, Optional, Set
from src.util.util import search_file, load_function, extract


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    Clean string to make it a valid filename.
    
    Args:
        name: Original string
        max_length: Maximum length
        
    Returns:
        Sanitized filename string
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = sanitized.strip('_.')
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


class ToolLibraryBuilder:
    """Tool library builder - Uses LLM to generate helper functions for CO problems."""
    
    def __init__(self, llm_client, problem: str):
        """
        Initialize Tool Library Builder.
        
        Args:
            llm_client: LLM client for model interaction
            problem: Problem type, e.g., 'tsp', 'jssp', 'cvrp'
        """
        self.llm_client = llm_client
        self.problem = problem
        self.output_dir = llm_client.output_dir
        
        task_description_file = os.path.join("src", "problems", problem, "task_description.txt")
        if os.path.exists(task_description_file):
            with open(task_description_file, encoding='utf-8') as f:
                self.task_description = f.read()
        else:
            self.task_description = f"Combinatorial optimization problem: {problem}"
        
        self.state_info = self._load_state_info()
        self.action_info = self._load_action_info()
    
    def _load_state_info(self) -> str:
        """Load generated state space information."""
        state_file = os.path.join("output", self.problem, "generate_problem_state", "problem_state.py")
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                return f.read()
        return "State space not yet generated"
    
    def _load_action_info(self) -> str:
        """Load generated action space information."""
        action_file = os.path.join("output", self.problem, "generate_action_space", "action_space.py")
        if os.path.exists(action_file):
            with open(action_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        comp_file = os.path.join("src", "problems", self.problem, "components.py")
        if os.path.exists(comp_file):
            with open(comp_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        return "Action space not yet generated"
    
    def generate_tool_library(self, smoke_test: bool = True, max_try_times: int = 5) -> str:
        """
        Generate complete tool library.
        
        Args:
            smoke_test: Whether to perform smoke testing on generated code
            max_try_times: Maximum retry attempts
            
        Returns:
            str: Path to generated tool library code file
        """
        print("=" * 60)
        print("Starting Tool Library Generation")
        print("=" * 60)
        
        print("\n[Step 1/4] Analyzing problem and identifying tool categories...")
        tool_categories = self._identify_tool_categories()
        self.llm_client.dump("tool_categories")
        
        print(f"  Identified {len(tool_categories)} tool categories:")
        for i, category in enumerate(tool_categories, 1):
            print(f"    {i}. {category}")
        
        print("\n[Step 2/4] Generating tools for each category...")
        all_tools_code = []
        
        for i, category in enumerate(tool_categories, 1):
            print(f"  [{i}/{len(tool_categories)}] Generating {category} tools...")
            tools_code = self._generate_tools_for_category(category)
            all_tools_code.append(tools_code)
            
            safe_category_name = sanitize_filename(category, max_length=50)
            self.llm_client.dump(f"tools_{safe_category_name}")
        
        print("\n[Step 3/4] Assembling tool library...")
        tool_library_code = self._assemble_tool_library(all_tools_code)
        
        tool_library_file = os.path.join(self.output_dir, "tool_library.py")
        with open(tool_library_file, "w", encoding='utf-8') as fp:
            fp.write(tool_library_code)
        print(f"Tool library temporarily saved to: {tool_library_file}")
        
        if smoke_test:
            print("\n=== Smoke Test Started ===")
            error_message = self._smoke_test(tool_library_code)
            
            current_try = 0
            while error_message and current_try < max_try_times:
                print(f"[Repair attempt {current_try + 1}/{max_try_times}]")
                self.llm_client.dump(f"tool_library_revision_try_{current_try}")
                
                tool_library_code = self._repair_tool_library(error_message, tool_library_code)
                error_message = self._smoke_test(tool_library_code)
                current_try += 1
            
            if error_message:
                print("❌ Smoke test failed: Maximum retry attempts exceeded")
                self.llm_client.dump("tool_library_abandoned")
                return None
            else:
                print("✅ Smoke test passed!")
        
        final_file = os.path.join(self.output_dir, "tool_library.py")
        with open(final_file, "w", encoding='utf-8') as fp:
            fp.write(tool_library_code)
        
        self._generate_tool_documentation(final_file)
        
        print(f"\n{'=' * 60}")
        print(f"✅ Tool library generation completed!")
        print(f"Saved to: {final_file}")
        print(f"{'=' * 60}\n")
        
        return final_file
    
    def _identify_tool_categories(self) -> List[str]:
        """Identify tool categories to generate."""
        prompt_dict = {
            "problem": self.problem,
            "task_description": self.task_description,
            "state_info": self.state_info[:2000],
            "action_info": self.action_info[:2000]
        }
        
        self.llm_client.load("identify_tool_categories", prompt_dict)
        response = self.llm_client.chat()
        categories_text = extract(response, "tool_categories")
        
        raw_categories = [line.strip().strip('-').strip() 
                         for line in categories_text.split('\n') 
                         if line.strip() and not line.strip().startswith('#')]
        
        categories = []
        for cat in raw_categories:
            if ':' in cat:
                cat_name = cat.split(':')[0].strip()
            else:
                cat_name = cat
            
            cat_name = sanitize_filename(cat_name, max_length=50)
            categories.append(cat_name)
        
        return categories[:8]
    
    def _generate_tools_for_category(self, category: str) -> str:
        """Generate tool functions for specific category."""
        prompt_dict = {
            "problem": self.problem,
            "task_description": self.task_description,
            "category": category,
            "state_info": self.state_info[:1500],
            "action_info": self.action_info[:1500]
        }
        
        self.llm_client.load("generate_category_tools", prompt_dict)
        response = self.llm_client.chat()
        code = extract(response, "python_code")
        
        return code
    
    def _assemble_tool_library(self, all_tools_code: List[str]) -> str:
        """Assemble all tool functions into complete tool_library.py."""
        header = f"""# This file is auto-generated by ToolLibraryBuilder for {self.problem} problem.
# It contains helper functions (tools) that encapsulate domain-specific logic.

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
"""
        
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            header += f"from src.problems.{self.problem}.components import Solution\n"
        else:
            header += "from src.problems.base.components import BaseSolution as Solution\n"
        
        header += "\n# ========== Tool Library ==========\n\n"
        
        full_code = header + "\n\n".join(all_tools_code)
        
        return full_code
    
    def _smoke_test(self, tool_library_code: str) -> str:
        """Perform smoke test on generated tool library."""
        try:
            smoke_data_dir = search_file("smoke_data", problem=self.problem)
            exclude_folders = [".ipynb_checkpoints", "__pycache__"]
            smoke_files = [
                f for f in os.listdir(smoke_data_dir) 
                if f != "previous_operations.txt" 
                and os.path.isfile(os.path.join(smoke_data_dir, f))
                and f not in exclude_folders
            ]
            
            if not smoke_files:
                return "No valid test data files found"
            
            smoke_data = os.path.join(smoke_data_dir, smoke_files[0])
            
            module = importlib.import_module(f"src.problems.{self.problem}.env")
            Env = getattr(module, "Env")
            
            if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
                comp_module = importlib.import_module(f"src.problems.{self.problem}.components")
            else:
                comp_module = importlib.import_module("src.problems.base.components")
            
            Solution = getattr(comp_module, "Solution", getattr(comp_module, "BaseSolution"))
            
            namespace = {
                'Solution': Solution,
                'np': __import__('numpy'),
                'List': List,
                'Tuple': Tuple,
                'Dict': Dict,
                'Optional': Optional,
                'Set': Set
            }
            exec(tool_library_code, namespace)
            
            env = Env(data_name=smoke_data)
            env.reset()
            
            print("  Testing tool function callability...")
            tool_count = 0
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_') and name not in ['List', 'Tuple', 'Dict', 'Optional', 'Set']:
                    tool_count += 1
                    print(f"    - Found tool: {name}")
            
            if tool_count == 0:
                return "No tool functions found"
            
            print(f"  ✓ Found {tool_count} tool functions")
            
            return None
            
        except Exception as e:
            error_message = traceback.format_exc()
            return f"Tool library code execution error:\n{error_message}"
    
    def _repair_tool_library(self, error_message: str, old_code: str) -> str:
        """Request LLM to repair tool library code."""
        prompt_dict = {
            "error_message": error_message,
            "old_code": old_code,
            "problem": self.problem
        }
        
        self.llm_client.load("repair_tool_library", prompt_dict)
        response = self.llm_client.chat()
        new_code = extract(response, "python_code")
        
        return new_code
    
    def _generate_tool_documentation(self, tool_file: str):
        """Generate documentation for tool library."""
        doc_file = tool_file.replace(".py", "_documentation.txt")
        
        with open(tool_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        import re
        functions = re.findall(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\):(.*?)(?=\ndef |$)', 
                              code, re.DOTALL)
        
        doc_content = f"# Tool Library Documentation for {self.problem}\n\n"
        doc_content += f"Total tools: {len(functions)}\n\n"
        doc_content += "=" * 60 + "\n\n"
        
        for func_name, params, body in functions:
            doc_content += f"## {func_name}\n"
            doc_content += f"Parameters: {params}\n"
            
            docstring_match = re.search(r'"""(.*?)"""', body, re.DOTALL)
            if docstring_match:
                doc_content += f"Description:\n{docstring_match.group(1).strip()}\n"
            
            doc_content += "\n" + "-" * 60 + "\n\n"
        
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        print(f"Tool documentation generated: {doc_file}")