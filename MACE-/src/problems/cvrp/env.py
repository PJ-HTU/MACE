import os
import tsplib95
import numpy as np
import pandas as pd
import networkx as nx
from src.problems.base.env import BaseEnv
from src.problems.cvrp.components import Solution


class Env(BaseEnv):
    """CVRP environment that stores instance data, current solution, and problem state."""

    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "cvrp")
        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "total_current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(
            set(node for route in self.current_solution.routes for node in route)
        ) == self.instance_data["node_num"]

    def load_data(self, data_path: str) -> None:
        problem = tsplib95.load(data_path)

        depot = problem.depots[0] - 1

        if problem.edge_weight_type == "EUC_2D":
            node_coords = problem.node_coords
            node_num = len(node_coords)
            distance_matrix = np.zeros((node_num, node_num))
            for i in range(node_num):
                for j in range(node_num):
                    if i != j:
                        x1, y1 = node_coords[i + 1]
                        x2, y2 = node_coords[j + 1]
                        distance_matrix[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        else:
            distance_matrix = nx.to_numpy_array(problem.get_graph())
            node_num = len(distance_matrix)

        if os.path.basename(data_path).split(".")[0].split("-")[-1][0] == "k":
            vehicle_num = int(
                os.path.basename(data_path).split(".")[0].split("-")[-1][1:]
            )
        elif open(data_path).readlines()[-1].strip().split(" : ")[0] == "VEHICLE":
            vehicle_num = int(
                open(data_path).readlines()[-1].strip().split(" : ")[-1]
            )
        else:
            raise NotImplementedError("Vehicle number parsing failed")

        capacity = problem.capacity
        demands = np.array(list(problem.demands.values()))

        return {
            "node_num": node_num,
            "distance_matrix": distance_matrix,
            "depot": depot,
            "vehicle_num": vehicle_num,
            "capacity": capacity,
            "demands": demands,
        }

    def init_solution(self) -> Solution:
        return Solution(
            routes=[[self.instance_data["depot"]] for _ in range(self.instance_data["vehicle_num"])],
            depot=self.instance_data["depot"],
        )

    def get_key_value(self, solution: Solution = None) -> float:
        if solution is None:
            solution = self.current_solution

        total_current_cost = 0
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            cost_for_vehicle = sum(
                self.instance_data["distance_matrix"][route[i]][route[i + 1]]
                for i in range(len(route) - 1)
            )
            if len(route) > 0:
                cost_for_vehicle += self.instance_data["distance_matrix"][route[-1]][route[0]]
            total_current_cost += cost_for_vehicle

        return total_current_cost

    def validation_solution(self, solution: Solution = None) -> bool:
        if solution is None:
            solution = self.current_solution

        for route in solution.routes:
            for node in route:
                if not (0 <= node < self.instance_data["node_num"]):
                    return False

        all_nodes = (
            [node for route in solution.routes for node in route if node != self.instance_data["depot"]]
            + [self.instance_data["depot"]]
        )
        if len(all_nodes) != len(set(all_nodes)):
            return False

        for route in solution.routes:
            if self.instance_data["depot"] not in route:
                return False

            load = sum(self.instance_data["demands"][node] for node in route)
            if load > self.instance_data["capacity"]:
                return False

        return True

    def get_unvisited_customers(self, solution: Solution = None) -> list:
        """
        Return a list of unvisited customer nodes (excluding the depot).
        """
        if solution is None:
            solution = self.current_solution

        visited = set()
        for route in solution.routes:
            visited.update(route)

        all_customers = set(range(self.instance_data["node_num"])) - {self.instance_data["depot"]}
        return list(all_customers - visited)

    def get_route_load(self, vehicle_id: int, solution: Solution = None) -> float:
        """
        Return the total demand of the specified vehicle route (excluding the depot).
        """
        if solution is None:
            solution = self.current_solution

        route = solution.routes[vehicle_id]
        depot = self.instance_data["depot"]
        return sum(self.instance_data["demands"][node] for node in route if node != depot)

    def can_add_to_route(self, vehicle_id: int, customer: int, solution: Solution = None) -> bool:
        """
        Check whether a customer can be added to a vehicle route without violating capacity constraints.
        """
        if solution is None:
            solution = self.current_solution

        current_load = self.get_route_load(vehicle_id, solution)
        customer_demand = self.instance_data["demands"][customer]
        return current_load + customer_demand <= self.instance_data["capacity"]

    def helper_function(self) -> dict:
        """
        Return a dictionary mapping helper function names to callables.
        """
        return {
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution,
            "get_unvisited_customers": self.get_unvisited_customers,
            "get_route_load": self.get_route_load,
            "can_add_to_route": self.can_add_to_route,
        }
