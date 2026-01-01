import numpy as np
import networkx as nx
import tsplib95
from src.problems.base.env import BaseEnv
from src.problems.tsp.components import Solution

class Env(BaseEnv):
    """TSP environment that stores instance data, current solution, and problem state to support algorithms."""
    
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "tsp")
        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        """Check if current solution is complete (all nodes visited once)."""
        return len(set(self.current_solution.tour)) == self.instance_data["node_num"]

    def load_data(self, data_path: str) -> None:
        """Load TSP instance data from TSPLIB format file."""
        problem = tsplib95.load(data_path)
        distance_matrix = nx.to_numpy_array(problem.get_graph())
        node_num = len(distance_matrix)
        return {"node_num": node_num, "distance_matrix": distance_matrix}

    def init_solution(self) -> None:
        """Initialize an empty solution."""
        return Solution(tour=[])

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value (total tour cost) of the current solution."""
        if solution is None:
            solution = self.current_solution
        
        current_cost = sum([self.instance_data["distance_matrix"][solution.tour[index]][solution.tour[index + 1]] 
                           for index in range(len(solution.tour) - 1)])
        if len(solution.tour) > 0:
            current_cost += self.instance_data["distance_matrix"][solution.tour[-1]][solution.tour[0]]
        
        return current_cost

    def validation_solution(self, solution: Solution=None) -> bool:
        """Check solution validity: node existence, uniqueness, and connectivity."""
        node_set = set()
        if solution is None:
            solution = self.current_solution

        if solution is not None and solution.tour is not None:
            for index, node in enumerate(solution.tour):
                if not (0 <= node < self.instance_data["node_num"]):
                    return False

                if node in node_set:
                    return False
                node_set.add(node)

                if index < len(solution.tour) - 1:
                    next_node = solution.tour[index + 1]
                    if self.instance_data["distance_matrix"][node][next_node] == np.inf:
                        return False
        return True

    def get_unvisited_nodes(self, solution: Solution = None) -> list:
        """Get list of unvisited nodes."""
        if solution is None:
            solution = self.current_solution
        
        visited = set(solution.tour)
        unvisited = [node for node in range(self.instance_data["node_num"]) if node not in visited]
        return unvisited
    
    def get_insertion_cost(self, node: int, position: int, solution: Solution = None) -> float:
        """Calculate cost increment of inserting a node at specified position."""
        if solution is None:
            solution = self.current_solution
        
        tour = solution.tour
        distance_matrix = self.instance_data["distance_matrix"]
        
        if position == 0:
            if len(tour) == 0:
                return 0.0
            return distance_matrix[node][tour[0]]
        elif position == len(tour):
            if len(tour) == 0:
                return 0.0
            return distance_matrix[tour[-1]][node]
        else:
            prev_node = tour[position - 1]
            next_node = tour[position]
            return (distance_matrix[prev_node][node] + 
                   distance_matrix[node][next_node] - 
                   distance_matrix[prev_node][next_node])
    
    def get_min_distance_to_tour(self, node: int, solution: Solution = None) -> float:
        """Calculate minimum distance from node to any node in current tour."""
        if solution is None:
            solution = self.current_solution
        
        tour = solution.tour
        if len(tour) == 0:
            return float('inf')
        
        distance_matrix = self.instance_data["distance_matrix"]
        return min(distance_matrix[node][tour_node] for tour_node in tour)
    
    def find_closest_pair(self, nodes: list = None) -> tuple:
        """Find the closest node pair in given node list."""
        if nodes is None:
            nodes = list(range(self.instance_data["node_num"]))
        
        distance_matrix = self.instance_data["distance_matrix"]
        min_dist = float('inf')
        closest_pair = (0, 1)
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = distance_matrix[nodes[i]][nodes[j]]
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (nodes[i], nodes[j])
        
        return closest_pair[0], closest_pair[1], min_dist
    
    def helper_function(self) -> dict:
        """Return dictionary of all helper functions for heuristic algorithms."""
        return {
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution,
            "get_unvisited_nodes": self.get_unvisited_nodes,
            "get_insertion_cost": self.get_insertion_cost,
            "get_min_distance_to_tour": self.get_min_distance_to_tour,
            "find_closest_pair": self.find_closest_pair,
        }