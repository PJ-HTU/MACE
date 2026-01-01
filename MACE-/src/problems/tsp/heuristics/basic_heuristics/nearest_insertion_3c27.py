from src.problems.tsp.components import *
import random

def nearest_insertion_3c27(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BaseOperator, dict]:
    """Nearest Insertion heuristic for TSP: Builds the tour incrementally by first appending a random (or specified) initial node if the tour is empty, then repeatedly selecting the unvisited node closest to the current tour (using get_min_distance_to_tour) and inserting it at the position that minimizes the insertion cost (using get_insertion_cost).

    This is a constructive heuristic that advances one step at a time: initial append or subsequent insertion. It ensures the tour grows validly without duplicates, using helper functions to avoid direct modifications or recomputations. The algorithm terminates (returns None) when all nodes are visited (num_unvisited == 0).

    Hyper-parameters in kwargs (with defaults):
        - initial_node (int or None, default=None): Specifies the starting node for the initial append step. If None, a random node from 0 to node_num-1 is chosen. This parameter is only relevant when the current tour is empty; subsequent calls ignore it.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - node_num (int): Total number of nodes (used for random initial node selection if initial_node is None).
            - current_solution (Solution): Current partial tour (used to check if empty and for helper functions).
            - get_unvisited_nodes (callable): Returns list of unvisited node IDs (used to check completion and select candidates).
            - get_min_distance_to_tour (callable): Computes min distance from a node to the tour (used to select the closest unvisited node).
            - get_insertion_cost (callable): Computes cost increase for inserting a node at a position (used to find optimal insertion position).
        algorithm_data (dict): Not used in this algorithm (can be empty).

    Returns:
        BaseOperator: An AppendOperator instance for the initial step (appends the starting node to an empty tour) or an InsertOperator instance for subsequent steps (inserts the selected node at the optimal position). Returns None if the tour is already complete (all nodes visited).
        dict: Empty dictionary {} as no algorithm data is updated or required.

    The algorithm proceeds as follows:
    1. Retrieve unvisited nodes using get_unvisited_nodes().
    2. If no unvisited nodes (complete tour), return None, {}.
    3. If current tour is empty (initial step):
       - Select initial_node from kwargs (random if None).
       - Return AppendOperator(initial_node), {}.
    4. Otherwise (insertion step):
       - Among unvisited nodes, select the one with the smallest get_min_distance_to_tour value (closest to tour). If ties, select the one with the smallest node ID for determinism.
       - For the selected node, evaluate get_insertion_cost at all possible positions (0 to len(tour)).
       - Select the position with the minimum insertion cost (ties broken by smallest position).
       - Return InsertOperator(selected_node, selected_position), {}.
    Note: get_min_distance_to_tour returns inf for empty tour, but initial step is handled separately. All operations preserve solution validity (no duplicates, proper insertions). If unvisited list is empty, it returns None without error.
    """
    # Retrieve necessary items from problem_state (never modify problem_state)
    current_solution = problem_state['current_solution']
    node_num = problem_state['node_num']
    get_unvisited_nodes_func = problem_state['get_unvisited_nodes']
    get_min_distance_to_tour_func = problem_state['get_min_distance_to_tour']
    get_insertion_cost_func = problem_state['get_insertion_cost']

    # Get current unvisited nodes
    unvisited_nodes = get_unvisited_nodes_func(current_solution)

    # If no unvisited nodes, tour is complete; return None
    if not unvisited_nodes:
        return None, {}

    # Check if current tour is empty (initial step)
    if not current_solution.tour:
        # Get initial_node from kwargs, default to random
        initial_node = kwargs.get('initial_node', None)
        if initial_node is None:
            initial_node = random.randint(0, node_num - 1)
        # Ensure initial_node is unvisited (should be all, but validate)
        if initial_node in current_solution.tour:
            # Fallback to random unvisited if somehow invalid, but unlikely
            initial_node = random.choice(unvisited_nodes)
        # Create and return AppendOperator
        operator = AppendOperator(initial_node)
        return operator, {}
    else:
        # Insertion step: Select unvisited node closest to the tour
        # Compute min distance for each unvisited node
        min_dists = {}
        for node in unvisited_nodes:
            min_dist = get_min_distance_to_tour_func(node, current_solution)
            # Since tour is not empty, min_dist should be finite
            if min_dist == float('inf'):
                continue  # Skip if somehow inf, though shouldn't happen
            min_dists[node] = min_dist

        if not min_dists:
            # No valid candidates (edge case, e.g., all inf), return None
            return None, {}

        # Select node with smallest min_dist (tie-break by smallest node ID)
        selected_node = min(min_dists, key=lambda n: (min_dists[n], n))

        # Find insertion position that minimizes cost
        tour_length = len(current_solution.tour)
        insertion_costs = {}
        min_cost = float('inf')
        best_position = None

        # Evaluate positions from 0 to tour_length (inclusive for end)
        for position in range(tour_length + 1):
            cost = get_insertion_cost_func(selected_node, position, current_solution)
            insertion_costs[position] = cost
            if cost < min_cost:
                min_cost = cost
                best_position = position
            elif cost == min_cost:
                # Tie-break by smallest position
                if position < best_position:
                    best_position = position

        if best_position is None:
            # No valid position (edge case), return None
            return None, {}

        # Create and return InsertOperator
        operator = InsertOperator(selected_node, best_position)
        return operator, {}