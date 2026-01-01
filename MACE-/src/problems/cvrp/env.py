import os
import tsplib95
import numpy as np
import pandas as pd
import networkx as nx
from src.problems.base.env import BaseEnv
from src.problems.cvrp.components import Solution


class Env(BaseEnv):
    """CVRP env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "cvrp")
        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "total_current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(set([node for route in self.current_solution.routes for node in route])) == self.instance_data["node_num"]

    # 加载CVRP问题的数据文件，解析并返回包含问题核心参数的字典
    def load_data(self, data_path: str) -> None:
        # 加载TSPLIB格式的CVRP数据文件（如.vrp文件），返回问题对象
        problem = tsplib95.load(data_path)
        
        # 获取 depot（仓库节点）的索引，TSPLIB中节点编号从1开始，这里转为0基索引
        depot = problem.depots[0] - 1
        
        # 判断距离计算方式是否为二维欧氏距离（EUC_2D）
        if problem.edge_weight_type == "EUC_2D":
            # 提取所有节点的坐标（键为节点编号1-based，值为(x,y)）
            node_coords = problem.node_coords
            # 计算节点总数（包括depot）
            node_num = len(node_coords)
            # 初始化距离矩阵（node_num x node_num）
            distance_matrix = np.zeros((node_num, node_num))
            # 遍历所有节点对，计算欧氏距离并填充矩阵
            for i in range(node_num):
                for j in range(node_num):
                    if i != j:  # 跳过自身到自身的距离（保持为0）
                        # 获取节点i+1和j+1的坐标（因为node_coords的键是1-based）
                        x1, y1 = node_coords[i + 1]
                        x2, y2 = node_coords[j + 1]
                        # 计算欧氏距离并赋值到矩阵
                        distance_matrix[i][j] = np.sqrt((x1 - x2) **2 + (y1 - y2)** 2)
        # 若距离类型不是EUC_2D，则直接从问题的图结构中提取距离矩阵
        else:
            # 将问题的图结构转换为numpy数组形式的距离矩阵
            distance_matrix = nx.to_numpy_array(problem.get_graph())
            # 节点总数为距离矩阵的维度
            node_num = len(distance_matrix)
        
        # 解析车辆车辆数量（从文件名或数据文件中提取）
        # 情况1：文件名格式如"xxx-kXX.vrp"，其中k后的数字为车辆数（如k10表示10辆车）

        if os.path.basename(data_path).split(".")[0].split("-")[-1][0] == "k":
            vehicle_num = int(os.path.basename(data_path).split(".")[0].split("-")[-1][1:])
        # 情况2：数据文件最后一行包含"VEHICLE : X"，其中X为车辆数
        elif open(data_path).readlines()[-1].strip().split(" : ")[0] == "VEHICLE":
            vehicle_num = int(open(data_path).readlines()[-1].strip().split(" : ")[-1])
        # 其他格式不支持，抛出异常
        else:
            raise NotImplementedError("Vehicle number error")
        
        # 提取车辆的最大容量限制
        capacity = problem.capacity
        # 提取各节点的需求（0基索引，depot的需求通常为0）
        demands = np.array(list(problem.demands.values()))
        
        # 返回包含所有解析出的问题参数的字典
        return {"node_num": node_num, "distance_matrix": distance_matrix, "depot": depot, "vehicle_num": vehicle_num, "capacity": capacity, "demands": demands}

    def init_solution(self) -> Solution:
        return Solution(routes=[[self.instance_data["depot"]] for _ in range(self.instance_data["vehicle_num"])], depot=self.instance_data["depot"])

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        if solution is None:
            solution = self.current_solution
        total_current_cost = 0
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            # The cost of the current solution for each vehicle.
            cost_for_vehicle = sum([self.instance_data["distance_matrix"][route[index]][route[index + 1]] for index in range(len(route) - 1)])
            if len(route) > 0:
                cost_for_vehicle += self.instance_data["distance_matrix"][route[-1]][route[0]]
            total_current_cost += cost_for_vehicle
        return total_current_cost

    def validation_solution(self, solution: Solution=None) -> bool:
        """
        Check the validation of this solution in following items:
            1. Node existence: Each node in each route must be within the valid range.
            2. Uniqueness: Each node (except for the depot) must only be visited once across all routes.
            3. Include depot: Each route must include at the depot.
            4. Capacity constraints: The load of each vehicle must not exceed its capacity.
        """
        if solution is None:
            solution = self.current_solution

        # Check node existence
        for route in solution.routes:
            for node in route:
                if not (0 <= node < self.instance_data["node_num"]):
                    return False

        # Check uniqueness
        all_nodes = [node for route in solution.routes for node in route if node != self.instance_data["depot"]] + [self.instance_data["depot"]]
        if len(all_nodes) != len(set(all_nodes)):
            return False

        for route in solution.routes:
            # Check include depot
            if self.instance_data["depot"] not in route:
                return False

            # Check vehicle load capacity constraints
            load = sum(self.instance_data["demands"][node] for node in route)
            if load > self.instance_data["capacity"]:
                return False

        return True

    # ==================== 辅助函数 ====================
    
    def get_unvisited_customers(self, solution: Solution = None) -> list:
        """
        获取未访问的客户节点列表(不含depot).
        
        Args:
            solution: 要查询的解,默认为当前解
            
        Returns:
            list: 未被任何路径访问的客户节点ID列表
            
        Example:
            unvisited = env.get_unvisited_customers()
            for customer in unvisited:
                # 处理未访问客户
        """
        if solution is None:
            solution = self.current_solution
            
        # 收集所有已访问的节点
        visited = set()
        for route in solution.routes:
            visited.update(route)
        
        # 找出未访问的客户(排除depot)
        all_customers = set(range(self.instance_data["node_num"])) - {self.instance_data["depot"]}
        unvisited = all_customers - visited
        
        return list(unvisited)
    
    def get_route_load(self, vehicle_id: int, solution: Solution = None) -> float:
        """
        计算指定车辆路径的总需求量.
        注意: depot的需求不计入总载重,只计算客户节点.
        
        Args:
            vehicle_id: 车辆ID
            solution: 要查询的解,默认为当前解
            
        Returns:
            float: 路径上所有客户节点的需求总和(不含depot)
            
        Example:
            load = env.get_route_load(vehicle_id=0)
            remaining = capacity - load
        """
        if solution is None:
            solution = self.current_solution
            
        route = solution.routes[vehicle_id]
        depot = self.instance_data["depot"]
        # 排除depot,只计算客户节点的需求
        load = sum(self.instance_data["demands"][node] for node in route if node != depot)
        return load
    
    def can_add_to_route(self, vehicle_id: int, customer: int, solution: Solution = None) -> bool:
        """
        检查是否可以将客户添加到指定车辆而不违反容量约束.
        
        Args:
            vehicle_id: 车辆ID
            customer: 客户节点ID
            solution: 要查询的解,默认为当前解
            
        Returns:
            bool: True表示可以添加
            
        Example:
            if env.can_add_to_route(vehicle_id, customer):
                # 可以添加
        """
        if solution is None:
            solution = self.current_solution
            
        current_load = self.get_route_load(vehicle_id, solution)
        customer_demand = self.instance_data["demands"][customer]
        
        return current_load + customer_demand <= self.instance_data["capacity"]
    
    def helper_function(self) -> dict:
        """
        返回所有辅助函数的字典,供启发式算法调用.
        
        Returns:
            dict: 函数名到函数引用的映射
            
        Example:
            helpers = env.helper_function()
            unvisited = helpers['get_unvisited_customers']()
        """
        return {
            # Core validation and state
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution,
            
            "get_unvisited_customers": self.get_unvisited_customers,
            "get_route_load": self.get_route_load,
            "can_add_to_route": self.can_add_to_route,
        }
