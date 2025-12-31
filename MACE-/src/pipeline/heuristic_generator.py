import os  # 导入os模块，用于文件和目录操作
import json  # 导入json模块，用于JSON数据的处理
import importlib  # 导入importlib模块，用于动态导入模块
import traceback  # 导入traceback模块，用于捕获和输出异常信息
from copy import deepcopy  # 从copy模块导入deepcopy，用于深拷贝对象
from src.problems.base.components import BaseOperator  # 从基础组件模块导入BaseOperator类，作为操作的基类
from src.util.util import (  # 从工具模块导入多个工具函数，用于提取信息、加载函数、处理字典等
    extract, extract_function_with_short_docstring, filter_dict_to_str, 
    find_key_value, load_function, parse_paper_to_dict, replace_strings_in_dict, 
    sanitize_function_name, load_framework_description, search_file
)
from src.util.llm_client.base_llm_client import BaseLLMClient  # 从LLM客户端模块导入BaseLLMClient类，作为LLM客户端的基类
reference_data = None

class HeuristicGenerator:  # 定义启发式生成器类，用于生成启发式算法
    def __init__(  # 初始化方法，设置LLM客户端、问题名称和输出目录
        self,
        llm_client: BaseLLMClient,  # LLM客户端实例，用于与大语言模型交互
        problem: str  # 问题名称，指定要处理的组合优化问题
    ) -> None:
        self.llm_client = llm_client  # 保存LLM客户端实例
        self.problem = problem  # 保存问题名称
        self.output_dir = self.llm_client.output_dir  # 从LLM客户端获取输出目录
        os.makedirs(self.output_dir, exist_ok=True)  # 创建输出目录（若不存在）

    def generate_from_llm(self, reference_data: str=None, smoke_test: bool=False) -> list[str]:  # 从LLM直接生成启发式算法
        heuristic_files = []  # 存储生成的启发式算法文件路径

        # 加载背景信息（问题描述、解决方案框架等）
        prompt_dict = self.llm_client.load_background(self.problem, "background_with_code", reference_data)
        
        # 生成可用的启发式算法描述
        self.llm_client.load("generate_from_llm", prompt_dict)  # 加载生成启发式的提示模板
        response = self.llm_client.chat()  # 与LLM交互，获取响应
        heuristics = extract(response, "heuristic", sep="\n")  # 从响应中提取启发式算法列表
        self.llm_client.dump("heuristic_from_llm")  # 保存交互记录

        for heuristic in heuristics:  # 遍历每个启发式算法描述，生成对应的代码文件
            # 加载之前的对话记录，继续生成单个启发式的详细信息
            self.llm_client.load_chat("heuristic_from_llm")
            heuristic_name = heuristic.split(":")[0]  # 提取启发式名称
            description = heuristic[len(heuristic_name) + 1: ]  # 提取启发式描述
            env_summarize = prompt_dict["env_summarize"]  # 获取环境数据摘要
            # 调用generate方法生成启发式代码文件，并添加到列表中
            more_prompt_dict = {"problem_state_introduction": prompt_dict['problem_state_introduction']}
            
            heuristic_files.append(self.generate(heuristic_name, description, env_summarize, smoke_test,more_prompt_dict = more_prompt_dict))

        return heuristic_files  # 返回生成的启发式文件路径列表

    def generate(self, heuristic_name: str, description: str, env_summarize: str="All data are possible", smoke_test: bool=False, more_prompt_dict=None, reminder=True) -> str:  # 生成单个启发式算法的代码文件
        # 加载特殊提示信息（用于避免常见错误）
        special_remind_file = os.path.join("src", "problems", self.problem, "prompt", "special_remind.txt")
        special_remind = "None"
        if os.path.exists(special_remind_file):
            special_remind = open(special_remind_file, encoding='utf-8').read()

        # 生成函数名称（标准化处理，避免命名冲突）
        function_name = sanitize_function_name(heuristic_name, description)
        # 构建提示字典，包含生成代码所需的信息
        prompt_dict = {"problem": self.problem, "heuristic_name": heuristic_name, "description": description, "function_name": function_name, "special_remind": special_remind, "env_summarize": env_summarize}
        if more_prompt_dict:  # 合并额外的提示信息
            prompt_dict.update(more_prompt_dict)
        
        # 确定组件文件路径（优先使用当前问题的组件，否则用基础组件）
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            prompt_dict["components_file"] = f"src.problems.{self.problem}.components"
        else:
            prompt_dict["components_file"] = f"src.problems.base.mdp_components"
        # 根据是否需要提醒，加载对应的代码生成提示模板
        if reminder:
            self.llm_client.load("implement_code_with_reminder", prompt_dict)
        else:
            self.llm_client.load("implement_code_without_reminder", prompt_dict)
        response = self.llm_client.chat()  # 与LLM交互，获取生成的代码
        code = extract(response, "python_code")  # 提取代码内容

        # 若需要冒烟测试，验证并修正代码
        if smoke_test:
            code = self.smoke_test(code, function_name)
            if not code:  # 若测试失败，放弃生成
                self.llm_client.dump(f"{function_name}_abandoned")
                return None
            else:
                print("=== 冒烟测试（smoke test）通过 ===")

        self.llm_client.dump(f"{function_name}")  # 保存交互记录

        # 保存生成的代码到文件
        output_heuristic_file = os.path.join(self.output_dir, function_name + ".py")
        print(f"Save {function_name} code to {output_heuristic_file}")
        with open(output_heuristic_file, "w", encoding='utf-8') as fp:
            fp.write(code)
        return output_heuristic_file  # 返回生成的文件路径

    def smoke_test(self, heuristic_code: str, function_name: str, max_try_times: int=5) -> str:  # 对生成的启发式代码进行冒烟测试
        print("=== 冒烟测试（smoke test）开始 ===")
        prompt_dict = {}
        # 确定组件文件路径
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            prompt_dict["components_file"] = f"src.problems.{self.problem}.components"
        else:
            prompt_dict["components_file"] = f"src.problems.base.mdp_components"

        smoke_data_dir = search_file("smoke_data", problem=self.problem)
        # 读取之前的操作记录（如果存在）
        previous_operations = []
        if os.path.exists(os.path.join(smoke_data_dir, "previous_operations.txt")):
            previous_operations = open(os.path.join(smoke_data_dir, "previous_operations.txt"), encoding='utf-8').readlines()
        

        # 过滤非文件的项（避免 .ipynb_checkpoints 等目录）
        # 定义需要排除的隐藏文件夹
        exclude_folders = [".ipynb_checkpoints", "__pycache__"]
        if self.problem != 'dposp':
            smoke_files = [f for f in os.listdir(smoke_data_dir) if f != "previous_operations.txt" and os.path.isfile(os.path.join(smoke_data_dir, f))]
        else:
            # 对于dposp问题，查找案例文件夹（目录）并排除隐藏文件夹
            smoke_files = [
                f for f in os.listdir(smoke_data_dir) 
                if f != "previous_operations.txt" 
                and os.path.isdir(os.path.join(smoke_data_dir, f))
                and f not in exclude_folders
            ]
        
        if not smoke_files:
            print("测试失败：未找到有效的测试数据文件")
            print("=== 冒烟测试（smoke test）结束 ===")
            return "未找到有效的测试数据文件", None, None
        smoke_data = os.path.join(smoke_data_dir, smoke_files[0])
        
        prompt_dict["function_name"] = function_name
        prompt_dict["previous_operations"] = "".join(previous_operations)

        # 准备测试环境
        module = importlib.import_module(f"src.problems.{self.problem}.env")  # 导入环境模块
        globals()["Env"] = getattr(module, "Env")  # 将环境类加入全局变量
        # 导入组件模块（优先当前问题，否则用基础组件）
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            module = importlib.import_module(f"src.problems.{self.problem}.components")
        else:
            module = importlib.import_module(f"src.problems.base.mdp_components")
        # 将组件类加入全局变量
        names_to_import = (name for name in dir(module) if not name.startswith('_'))
        for name in names_to_import:
            globals()[name] = getattr(module, name)
        env = Env(data_name=smoke_data)  # 初始化环境

        
        # 最多尝试max_try_times次修正代码
        for _ in range(max_try_times):
            env.reset()  # 重置环境
            # 提取实例问题状态并过滤（简化显示）
            prompt_dict["smoke_instance_problem_state"] = filter_dict_to_str(env.get_instance_problem_state(env.instance_data))
            # 执行之前的操作，恢复到测试状态
            for previous_operation in previous_operations:
                env.run_operator(eval(previous_operation.strip()))
            prompt_dict["smoke_solution"] = env.current_solution  # 当前解决方案
            # 提取解决方案状态并过滤
            prompt_dict["smoke_solution_problem_state"] = filter_dict_to_str(env.get_solution_problem_state(env.instance_data, env.current_solution))
            try:
                # 加载启发式函数并运行一次
                heuristic = load_function(heuristic_code, function_name=function_name)
                operator = env.run_heuristic(heuristic)
            except Exception as e:  # 捕获运行时异常
                operator = traceback.format_exc()  # 保存异常信息
            # 若操作有效（为空或BaseOperator实例），进行结果对比
            if operator is None or isinstance(operator, BaseOperator):
                # 获取预期结果
                self.llm_client.load("smoke_test_expected_result.txt", prompt_dict)
                response = self.llm_client.chat()
                expected_result = extract(response, "expected_result")

                # 准备实际结果信息
                prompt_dict["output_result"] = str(operator)
                prompt_dict["updated_smoke_solution"] = env.current_solution
                prompt_dict["updated_smoke_solution_problem_state"] = filter_dict_to_str(env.get_solution_problem_state(env.instance_data, env.current_solution))

                # 对比实际结果与预期结果
                prompt_dict["expected_result"] = expected_result
                self.llm_client.load("smoke_test_compare.txt", prompt_dict)
                response = self.llm_client.chat()
                response = extract(response, "python_code")
                # 根据对比结果处理
                if response is None:  # 无法修正，放弃
                    self.llm_client.load("We can not implement and give up.")
                    return None
                elif "correct" in response:  # 结果正确，返回代码
                    self.llm_client.load(f"To ensure the stable of heuristics, we adjust the code to:\n{heuristic_code}")
                    return heuristic_code
                else:  # 结果不正确，更新代码重试
                    heuristic_code = response
            else:  # 代码运行崩溃，提示LLM修正
                prompt_dict["error_message"] = operator
                self.llm_client.load("smoke_test_crashed.txt", prompt_dict)
                response = self.llm_client.chat()
                heuristic_code = extract(response, "python_code")
                if heuristic_code is None:  # 无法修正，放弃
                    self.llm_client.load("We can not implement and give up.")
                    return None
        # 超过最大尝试次数，放弃
        self.llm_client.load("We can not implement and give up.")
        return None