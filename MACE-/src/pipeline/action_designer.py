"""
Action Designer - Generates problem action space through LLM.

This module implements the Action Designer component in MACE Stage One,
generating Constructive Actions and Improvement Actions via LLM.
"""

import os
import importlib
import traceback
from typing import List, Dict, Tuple
from src.util.util import search_file, load_function, extract


class ActionDesigner:
    """Action space designer - Uses LLM to generate action operators for CO problems."""
    
    def __init__(self, llm_client, problem: str):
        """
        Initialize Action Designer.
        
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
    
    def generate_action_space(self, smoke_test: bool = True, max_try_times: int = 5) -> str:
        """
        Generate complete action space including constructive and improvement actions.
        
        Args:
            smoke_test: Whether to perform smoke testing on generated code
            max_try_times: Maximum retry attempts
            
        Returns:
            str: Path to generated action space code file
        """
        print("=" * 60)
        print("Starting Action Space Generation")
        print("=" * 60)
        
        print("\n[Step 1/3] Generating Constructive Actions...")
        constructive_actions_code = self._generate_constructive_actions()
        self.llm_client.dump("constructive_actions")
        
        print("\n[Step 2/3] Generating Improvement Actions...")
        improvement_actions_code = self._generate_improvement_actions()
        self.llm_client.dump("improvement_actions")
        
        print("\n[Step 3/3] Assembling and saving action space...")
        action_space_code = self._assemble_action_space(
            constructive_actions_code, 
            improvement_actions_code
        )
        
        action_space_file = os.path.join(self.output_dir, "action_space.py")
        with open(action_space_file, "w", encoding='utf-8') as fp:
            fp.write(action_space_code)
        print(f"Action space temporarily saved to: {action_space_file}")
        
        if smoke_test:
            print("\n=== Smoke Test Started ===")
            error_message = self._smoke_test(action_space_code)
            
            current_try = 0
            while error_message and current_try < max_try_times:
                print(f"[Repair attempt {current_try + 1}/{max_try_times}]")
                self.llm_client.dump(f"action_space_revision_try_{current_try}")
                
                action_space_code = self._repair_action_space(error_message, action_space_code)
                error_message = self._smoke_test(action_space_code)
                current_try += 1
            
            if error_message:
                print("❌ Smoke test failed: Maximum retry attempts exceeded")
                self.llm_client.dump("action_space_abandoned")
                return None
            else:
                print("✅ Smoke test passed!")
        
        final_file = os.path.join(self.output_dir, "action_space.py")
        with open(final_file, "w", encoding='utf-8') as fp:
            fp.write(action_space_code)
        
        print(f"\n{'=' * 60}")
        print(f"✅ Action space generation completed!")
        print(f"Saved to: {final_file}")
        print(f"{'=' * 60}\n")
        
        return final_file
    
    def _generate_constructive_actions(self) -> str:
        """Generate constructive actions code."""
        prompt_dict = {
            "problem": self.problem,
            "task_description": self.task_description
        }
        
        self.llm_client.load("design_constructive_actions", prompt_dict)
        response = self.llm_client.chat()
        code = extract(response, "python_code")
        
        return code
    
    def _generate_improvement_actions(self) -> str:
        """Generate improvement actions code."""
        prompt_dict = {
            "problem": self.problem,
            "task_description": self.task_description
        }
        
        self.llm_client.load("design_improvement_actions", prompt_dict)
        response = self.llm_client.chat()
        code = extract(response, "python_code")
        
        return code
    
    def _assemble_action_space(self, constructive_code: str, improvement_code: str) -> str:
        """
        Assemble constructive and improvement actions into complete action_space.py.
        
        Args:
            constructive_code: Constructive actions code
            improvement_code: Improvement actions code
            
        Returns:
            str: Complete action space code
        """
        header = f"""# This file is auto-generated by ActionDesigner for {self.problem} problem.
# It contains both Constructive Actions and Improvement Actions.

from src.problems.base.components import BaseOperator
"""
        
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            header += f"from src.problems.{self.problem}.components import Solution\n"
        else:
            header += "from src.problems.base.components import BaseSolution as Solution\n"
        
        header += "\n# ===== Constructive Actions =====\n"
        
        full_code = header + constructive_code + "\n\n# ===== Improvement Actions =====\n" + improvement_code
        
        return full_code
    
    def _smoke_test(self, action_space_code: str) -> str:
        """
        Perform smoke test on generated action space.
        
        Args:
            action_space_code: Action space code
            
        Returns:
            str: Error message if test fails, None if passes
        """
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
                'BaseOperator': getattr(comp_module, 'BaseOperator')
            }
            exec(action_space_code, namespace)
            
            env = Env(data_name=smoke_data)
            env.reset()
            
            print("  Testing constructive actions...")
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, namespace['BaseOperator']) and obj != namespace['BaseOperator']:
                    if 'Append' in name or 'Insert' in name:
                        print(f"    - Found action: {name}")
            
            print("  Testing improvement actions...")
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, namespace['BaseOperator']) and obj != namespace['BaseOperator']:
                    if 'Swap' in name or 'Opt' in name or 'Reverse' in name:
                        print(f"    - Found action: {name}")
            
            return None
            
        except Exception as e:
            error_message = traceback.format_exc()
            return f"Action space code execution error:\n{error_message}"
    
    def _repair_action_space(self, error_message: str, old_code: str) -> str:
        """
        Request LLM to repair action space code.
        
        Args:
            error_message: Error message
            old_code: Previous code
            
        Returns:
            str: Repaired code
        """
        prompt_dict = {
            "error_message": error_message,
            "old_code": old_code,
            "problem": self.problem
        }
        
        self.llm_client.load("repair_action_space", prompt_dict)
        response = self.llm_client.chat()
        new_code = extract(response, "python_code")
        
        return new_code