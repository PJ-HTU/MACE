from src.problems.tsp.components import *
import random

def farthest_insertion_33fc(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BaseOperator, dict]:
    """Farthest Insertion heuristic for TSP. This is a constructive heuristic that builds the tour incrementally.
    It starts by appending a random (or specified) initial node to an empty tour. Then, in each subsequent step,
    it selects the unvisited node that is farthest from the current partial tour (maximum minimum distance to any tour node)
    and inserts it at the position that minimizes the insertion cost (least increase in tour length). This promotes
    covering distant nodes early while optimizing local placements. The process continues until all nodes are visited.
    Each call to this function advances the solution by one step (appending the initial node or inserting one node),
    returning the appropriate operator. If the tour is already complete, it returns None, {} to indicate no further action.

    Hyper-parameters in kwargs (with defaults):
    - initial_node (int or None, default=None): The node to append as the starting point if the tour is empty.
      If None, a random node from 0 to node_num-1 is selected. This is only used when num_visited == 0.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - node_num (int): The total number of nodes in the problem.
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes (used indirectly via helpers).
            - current_solution (Solution): Current solution instance (used indirectly via helpers).
            - get_unvisited_nodes (callable): Returns list of unvisited node IDs.
            - get_min_distance_to_tour (callable): Returns the min distance from a node to the current tour.
            - get_insertion_cost (callable): Returns the cost increase for inserting a node at a position.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not require or update any specific keys in algorithm_data;
            it operates solely based on the current problem_state and returns an empty dict.

    Returns:
        BaseOperator or None: The AppendOperator for the initial node (if tour is empty) or InsertOperator for the selected node at the best position (if partial tour).
            Returns None if the tour is already complete (num_unvisited == 0).
        dict: Empty dict {} as no persistent data is updated in algorithm_data by this heuristic.
    """
    # Extract necessary elements from problem_state (never modify problem_state)
    node_num = problem_state['node_num']
    current_solution = problem_state['current_solution']
    get_unvisited_nodes_func = problem_state['get_unvisited_nodes']
    get_min_distance_to_tour_func = problem_state['get_min_distance_to_tour']
    get_insertion_cost_func = problem_state['get_insertion_cost']
    
    # Get current unvisited nodes using the helper (uses current_solution if solution=None)
    unvisited = get_unvisited_nodes_func()
    num_unvisited = len(unvisited)
    
    if num_unvisited == 0:
        # Tour is complete; no further action needed
        return None, {}
    
    # Handle initial step: empty tour (num_visited == 0, which implies num_unvisited == node_num)
    current_tour = current_solution.tour
    if len(current_tour) == 0:
        # Select initial node
        initial_node = kwargs.get('initial_node', None)
        if initial_node is None:
            # Randomly select from all nodes (0 to node_num-1)
            selected_node = random.randint(0, node_num - 1)
        else:
            # Use provided initial_node (assume it's valid, 0 <= initial_node < node_num)
            selected_node = initial_node
        # For empty tour, append the selected node (insertion cost is 0.0 by helper)
        operator = AppendOperator(selected_node)
        return operator, {}
    
    # Partial tour: select the farthest unvisited node (max min_distance_to_tour)
    # Compute min_dist for each unvisited node using the helper
    node_distances = {}
    for node in unvisited:
        min_dist = get_min_distance_to_tour_func(node)
        # Note: for non-empty tour, min_dist should be finite; skip if inf (though unlikely)
        if min_dist != float('inf'):
            node_distances[node] = min_dist
        # If no valid distances (edge case, empty distances), but since tour non-empty and unvisited exist, should have values
    
    if not node_distances:
        # No valid unvisited nodes with finite distance (unlikely, but safety)
        return None, {}
    
    # Select node with maximum min_distance (farthest from tour)
    selected_node = max(node_distances, key=node_distances.get)
    
    # Find the best insertion position: minimize get_insertion_cost over possible positions (0 to len(tour))
    tour_length = len(current_tour)
    best_position = 0
    best_cost = float('inf')
    for position in range(tour_length + 1):
        insertion_cost = get_insertion_cost_func(selected_node, position)
        # For empty tour handled above; here tour non-empty, costs should be finite
        if insertion_cost < best_cost:
            best_cost = insertion_cost
            best_position = position
    
    # If no valid position found (unlikely), return None
    if best_cost == float('inf'):
        return None, {}
    
    # Create and return the InsertOperator for the selected node at best position
    operator = InsertOperator(selected_node, best_position)
    return operator, {}