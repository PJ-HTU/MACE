from src.problems.tsp.components import *
from typing import Union
import random

def cheapest_insertion_df42(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[Union[AppendOperator, InsertOperator], dict]:
    """
    Implements the Cheapest Insertion heuristic for TSP. This constructive heuristic starts with an empty tour by appending a random initial node. In subsequent steps, it evaluates possible insertions of unvisited nodes into the current partial tour and selects the (node, position) pair with the minimum insertion cost, using the get_insertion_cost helper. If the number of candidate evaluations exceeds the max_evaluations limit, it samples randomly to approximate the best insertion. This greedy choice per step builds a high-quality partial tour incrementally. The algorithm returns None if the tour is already complete (all nodes visited). It ensures validity by using only unvisited nodes and valid positions, assuming the input partial solution is valid (no duplicates, proper partial tour).

    The hyper-parameter max_evaluations limits computation: when the number of unvisited nodes times possible positions exceeds this, random sampling is used instead of exhaustive search, trading optimality for efficiency in large instances.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current partial tour solution.
            - get_unvisited_nodes (callable): Function to retrieve unvisited node IDs.
            - get_insertion_cost (callable): Function to compute the cost increase for inserting a node at a position.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no items are necessary or used.
        **kwargs: Hyper-parameters for the algorithm.
            - max_evaluations (int): Maximum number of cost evaluations per step. Default: 100. Used to limit time when many candidates exist by sampling.

    Returns:
        Union[AppendOperator, InsertOperator]: For an empty tour, an AppendOperator for a random starting node. For non-empty partial tours, an InsertOperator for the node and position minimizing insertion cost (or approximated via sampling). Returns None if the tour is complete (no unvisited nodes).
        dict: Empty dictionary, as this algorithm does not update or require persistent algorithm data.
    """
    # Extract necessary components from problem_state (never modify problem_state)
    current_solution = problem_state['current_solution']
    get_unvisited_nodes_func = problem_state['get_unvisited_nodes']
    get_insertion_cost_func = problem_state['get_insertion_cost']
    
    # Get unvisited nodes using the helper (uses current_solution if solution=None)
    unvisited = get_unvisited_nodes_func()
    
    # If no unvisited nodes, tour is complete: return None (no operator needed)
    if not unvisited:
        return None, {}
    
    # Hyper-parameter with default
    max_evaluations = kwargs.get('max_evaluations', 100)
    
    # Special case: empty tour (start with random append)
    if len(current_solution.tour) == 0:
        # Pick a random starting node from unvisited
        start_node = random.choice(unvisited)
        # Return AppendOperator to add it at the end (equivalent to inserting into empty)
        return AppendOperator(start_node), {}
    
    # Non-empty tour: find best insertion (node, position) minimizing cost
    num_visited = len(current_solution.tour)
    num_positions = num_visited + 1  # Positions: 0 to len(tour)
    total_candidates = len(unvisited) * num_positions
    
    best_node = None
    best_position = None
    min_cost = float('inf')
    
    if total_candidates <= max_evaluations:
        # Exhaustive evaluation: check all (node, position) pairs
        for node in unvisited:
            for position in range(num_positions):
                cost = get_insertion_cost_func(node, position)
                # Only update if cost is finite and smaller (handles any inf cases, though unlikely)
                if cost < min_cost and cost != float('inf'):
                    min_cost = cost
                    best_node = node
                    best_position = position
    else:
        # Sampling: generate all candidate pairs and sample max_evaluations randomly
        candidates = [(node, position) for node in unvisited for position in range(num_positions)]
        sampled_candidates = random.sample(candidates, max_evaluations)
        # Evaluate sampled pairs
        for node, position in sampled_candidates:
            cost = get_insertion_cost_func(node, position)
            if cost < min_cost and cost != float('inf'):
                min_cost = cost
                best_node = node
                best_position = position
    
    # If no valid best found (unlikely, but if all costs inf, e.g., disconnected graph), return None
    # Otherwise, return InsertOperator for the best insertion
    if best_node is None:
        return None, {}
    else:
        return InsertOperator(best_node, best_position), {}