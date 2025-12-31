import os  # 导入os模块，用于处理文件路径等操作系统相关功能
import time  # 导入time模块，用于时间控制
from src.problems.base.components import BaseOperator  # 从基础组件模块导入BaseOperator类，作为所有操作的基类
from src.problems.base.env import BaseEnv  # 从基础环境模块导入BaseEnv类，作为问题环境的基类
from src.util.util import load_function  # 从工具模块导入load_function函数，用于加载启发式函数


class SingleHyperHeuristic:  # 定义SingleHyperHeuristic类，用于单一启发式算法的执行控制
    def __init__(  # 类的初始化方法
        self,
        heuristic: str,  # 启发式算法的文件路径或代码字符串
        problem: str,  # 问题名称，用于定位相关资源
        iterations_scale_factor: float=1.0,  # 迭代缩放因子，用于计算最大步骤数
        penalty_value: float=1e10,  # 超时时返回的惩罚值，默认为1e10
    ) -> None:
        self.heuristic = load_function(heuristic, problem=problem)  # 加载启发式函数并存储在实例变量中
        self.iterations_scale_factor = iterations_scale_factor  # 保存迭代缩放因子
        self.penalty_value = penalty_value  # 保存惩罚值

    def run(self, env: BaseEnv, time_limit: float=None, **kwargs) -> bool:  # 定义run方法，用于在环境中运行启发式算法，返回布尔值表示是否得到有效完整解
        current_steps = 0  # 初始化当前步骤计数为0
        max_steps = int(env.construction_steps * self.iterations_scale_factor) *1
        # 计算最大步骤数，为环境构建步骤数乘以缩放因子
        
        start_time = time.time()  # 记录开始时间
        timeout_occurred = False  # 标记是否发生超时
        
        while current_steps <= max_steps and env.continue_run:  
            # 检查是否超过时间限制
            if time_limit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time > time_limit:
                    timeout_occurred = True
                    print(f"求解超时（{time_limit}秒），返回惩罚值")
                    break
            
            _ = env.run_heuristic(self.heuristic)
            # 在环境中运行选中的启发式函数，忽略返回结果
            current_steps += 1  # 当前步骤数加1
        
        # 如果发生超时，设置惩罚值
        if timeout_occurred:
            # 返回False表示未得到有效解
    
            return False
        # env.is_complete_solution and env.validation_solution
        return env.is_complete_solution and env.validation_solution