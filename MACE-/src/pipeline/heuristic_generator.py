import os
import json
import importlib
import traceback
from copy import deepcopy
from src.problems.base.components import BaseOperator
from src.util.util import (
    extract, extract_function_with_short_docstring, filter_dict_to_str, 
    find_key_value, load_function, parse_paper_to_dict, replace_strings_in_dict, 
    sanitize_function_name, load_framework_description, search_file
)
from src.util.llm_client.base_llm_client import BaseLLMClient

reference_data = None


class HeuristicGenerator:
    """Heuristic generator for creating heuristic algorithms."""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        problem: str
    ) -> None:
        self.llm_client = llm_client
        self.problem = problem
        self.output_dir = self.llm_client.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_from_llm(self, reference_data: str=None, smoke_test: bool=False) -> list[str]:
        """Generate heuristic algorithms directly from LLM."""
        heuristic_files = []

        prompt_dict = self.llm_client.load_background(self.problem, "background_with_code", reference_data)
        
        self.llm_client.load("generate_from_llm", prompt_dict)
        response = self.llm_client.chat()
        heuristics = extract(response, "heuristic", sep="\n")
        self.llm_client.dump("heuristic_from_llm")

        for heuristic in heuristics:
            self.llm_client.load_chat("heuristic_from_llm")
            heuristic_name = heuristic.split(":")[0]
            description = heuristic[len(heuristic_name) + 1: ]
            env_summarize = prompt_dict["env_summarize"]
            
            more_prompt_dict = {"problem_state_introduction": prompt_dict['problem_state_introduction']}
            heuristic_files.append(self.generate(heuristic_name, description, env_summarize, smoke_test, more_prompt_dict=more_prompt_dict))

        return heuristic_files

    def generate(self, heuristic_name: str, description: str, env_summarize: str="All data are possible", 
                 smoke_test: bool=False, more_prompt_dict=None, reminder=True) -> str:
        """Generate code file for a single heuristic algorithm."""
        special_remind_file = os.path.join("src", "problems", self.problem, "prompt", "special_remind.txt")
        special_remind = "None"
        if os.path.exists(special_remind_file):
            special_remind = open(special_remind_file, encoding='utf-8').read()

        function_name = sanitize_function_name(heuristic_name, description)
        prompt_dict = {
            "problem": self.problem, 
            "heuristic_name": heuristic_name, 
            "description": description, 
            "function_name": function_name, 
            "special_remind": special_remind, 
            "env_summarize": env_summarize
        }
        if more_prompt_dict:
            prompt_dict.update(more_prompt_dict)
        
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            prompt_dict["components_file"] = f"src.problems.{self.problem}.components"
        else:
            prompt_dict["components_file"] = f"src.problems.base.mdp_components"
        
        if reminder:
            self.llm_client.load("implement_code_with_reminder", prompt_dict)
        else:
            self.llm_client.load("implement_code_without_reminder", prompt_dict)
        
        response = self.llm_client.chat()
        code = extract(response, "python_code")

        if smoke_test:
            code = self.smoke_test(code, function_name)
            if not code:
                self.llm_client.dump(f"{function_name}_abandoned")
                return None
            else:
                print("=== Smoke test passed ===")

        self.llm_client.dump(f"{function_name}")

        output_heuristic_file = os.path.join(self.output_dir, function_name + ".py")
        print(f"Save {function_name} code to {output_heuristic_file}")
        with open(output_heuristic_file, "w", encoding='utf-8') as fp:
            fp.write(code)
        
        return output_heuristic_file

    def smoke_test(self, heuristic_code: str, function_name: str, max_try_times: int=5) -> str:
        """Perform smoke test on generated heuristic code."""
        print("=== Smoke test started ===")
        prompt_dict = {}
        
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            prompt_dict["components_file"] = f"src.problems.{self.problem}.components"
        else:
            prompt_dict["components_file"] = f"src.problems.base.mdp_components"

        smoke_data_dir = search_file("smoke_data", problem=self.problem)
        previous_operations = []
        if os.path.exists(os.path.join(smoke_data_dir, "previous_operations.txt")):
            previous_operations = open(os.path.join(smoke_data_dir, "previous_operations.txt"), encoding='utf-8').readlines()

        exclude_folders = [".ipynb_checkpoints", "__pycache__"]
        if self.problem != 'dposp':
            smoke_files = [f for f in os.listdir(smoke_data_dir) if f != "previous_operations.txt" and os.path.isfile(os.path.join(smoke_data_dir, f))]
        else:
            smoke_files = [
                f for f in os.listdir(smoke_data_dir) 
                if f != "previous_operations.txt" 
                and os.path.isdir(os.path.join(smoke_data_dir, f))
                and f not in exclude_folders
            ]
        
        if not smoke_files:
            print("Test failed: No valid test data files found")
            print("=== Smoke test ended ===")
            return "No valid test data files found", None, None
        
        smoke_data = os.path.join(smoke_data_dir, smoke_files[0])
        
        prompt_dict["function_name"] = function_name
        prompt_dict["previous_operations"] = "".join(previous_operations)

        module = importlib.import_module(f"src.problems.{self.problem}.env")
        globals()["Env"] = getattr(module, "Env")
        
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            module = importlib.import_module(f"src.problems.{self.problem}.components")
        else:
            module = importlib.import_module(f"src.problems.base.mdp_components")
        
        names_to_import = (name for name in dir(module) if not name.startswith('_'))
        for name in names_to_import:
            globals()[name] = getattr(module, name)
        
        env = Env(data_name=smoke_data)

        for _ in range(max_try_times):
            env.reset()
            prompt_dict["smoke_instance_problem_state"] = filter_dict_to_str(env.get_instance_problem_state(env.instance_data))
            
            for previous_operation in previous_operations:
                env.run_operator(eval(previous_operation.strip()))
            
            prompt_dict["smoke_solution"] = env.current_solution
            prompt_dict["smoke_solution_problem_state"] = filter_dict_to_str(env.get_solution_problem_state(env.instance_data, env.current_solution))
            
            try:
                heuristic = load_function(heuristic_code, function_name=function_name)
                operator = env.run_heuristic(heuristic)
            except Exception as e:
                operator = traceback.format_exc()
            
            if operator is None or isinstance(operator, BaseOperator):
                self.llm_client.load("smoke_test_expected_result.txt", prompt_dict)
                response = self.llm_client.chat()
                expected_result = extract(response, "expected_result")

                prompt_dict["output_result"] = str(operator)
                prompt_dict["updated_smoke_solution"] = env.current_solution
                prompt_dict["updated_smoke_solution_problem_state"] = filter_dict_to_str(env.get_solution_problem_state(env.instance_data, env.current_solution))

                prompt_dict["expected_result"] = expected_result
                self.llm_client.load("smoke_test_compare.txt", prompt_dict)
                response = self.llm_client.chat()
                response = extract(response, "python_code")
                
                if response is None:
                    self.llm_client.load("We can not implement and give up.")
                    return None
                elif "correct" in response:
                    self.llm_client.load(f"To ensure the stable of heuristics, we adjust the code to:\n{heuristic_code}")
                    return heuristic_code
                else:
                    heuristic_code = response
            else:
                prompt_dict["error_message"] = operator
                self.llm_client.load("smoke_test_crashed.txt", prompt_dict)
                response = self.llm_client.chat()
                heuristic_code = extract(response, "python_code")
                if heuristic_code is None:
                    self.llm_client.load("We can not implement and give up.")
                    return None
        
        self.llm_client.load("We can not implement and give up.")
        return None