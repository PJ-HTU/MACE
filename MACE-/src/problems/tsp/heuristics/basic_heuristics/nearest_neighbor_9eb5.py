import random
from src.problems.tsp.components import *
def nearest_neighbor_9eb5(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """Nearest Neighbor heuristic for TSP: This constructive heuristic builds the tour step-by-step by starting with an initial node (random or fixed if empty) and repeatedly appending the unvisited node closest to the current last node in the tour until all nodes are visited. It extends any partial tour provided, always appending to the end without insertions. This ensures greedy path extension from the current endpoint. The algorithm returns one AppendOperator per call for incremental construction in a hyper-heuristic framework. If the tour is already complete (no unvisited nodes), it returns None to indicate no further action is needed.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - node_num (int): The total number of nodes in the problem.
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes.
            - current_solution (Solution): Current solution instance representing the partial tour.
            - get_unvisited_nodes (callable): Function to retrieve the list of unvisited node IDs from the current solution.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no items are necessary or modified.
        **kwargs: Hyper-parameters for the algorithm.
            - random_start (bool): Whether to select a random starting node when the tour is empty. If False, selects the smallest node ID (e.g., 0). Default: True.

    Returns:
        AppendOperator: An instance of AppendOperator to append the next closest unvisited node to the end of the tour, or None if the tour is complete (no unvisited nodes left) or no valid action is possible.
        dict: Updated algorithm data, which is empty {} since this heuristic does not require persistent state across calls.
    """
    # Extract necessary items from problem_state (do not modify problem_state)
    node_num = problem_state['node_num']
    distance_matrix = problem_state['distance_matrix']
    current_solution = problem_state['current_solution']
    get_unvisited_nodes_func = problem_state['get_unvisited_nodes']  # Renamed to avoid conflict with built-in

    # Extract hyper-parameters with defaults (all hyper-parameters must have defaults)
    random_start = kwargs.get('random_start', True)

    # Get the current tour (do not modify current_solution)
    tour = current_solution.tour[: ]  # Copy to avoid any accidental modification

    # Get unvisited nodes using the helper function (works for partial or empty solutions)
    unvisited = get_unvisited_nodes_func(current_solution)

    # If no unvisited nodes, the tour is complete; return None to stop construction
    if not unvisited:
        return None, {}

    # Case 1: Empty tour (len(tour) == 0) - Select initial starting node
    if len(tour) == 0:
        # Ensure there are nodes to start with
        if node_num == 0:
            return None, {}
        # Select starting node based on hyper-parameter
        if random_start:
            # Randomly select from all nodes (which are all unvisited)
            start_node = random.choice(unvisited)
        else:
            # Select the smallest node ID (deterministic start, e.g., node 0)
            start_node = min(unvisited)
        # Return AppendOperator for the starting node; this begins the tour
        return AppendOperator(start_node), {}

    # Case 2: Partial tour (len(tour) > 0) - Extend by appending closest to the last node
    # Get the last node in the current tour
    last_node = tour[-1]

    # Find the unvisited node with the minimum distance to last_node
    # Iterate over unvisited nodes, checking distances (handle empty unvisited already checked above)
    # Use direct distance_matrix access for precision to the last node (not min to entire tour)
    min_dist = float('inf')
    next_node = None
    for node in unvisited:
        # Ensure node != last_node (though unvisited should exclude tour nodes, including last)
        if node == last_node:
            continue
        dist = distance_matrix[last_node][node]
        # Check for valid finite distance (assume distance_matrix has no inf for valid TSP)
        if dist < min_dist:
            min_dist = dist
            next_node = node

    # If no valid next_node found (e.g., all distances inf, unlikely in valid TSP), stop
    if next_node is None:
        return None, {}

    # Return AppendOperator to append the closest node; this ensures incremental valid extension
    # The resulting tour after application will have no duplicates (enforced by unvisited selection)
    # and will progress toward visiting all nodes
    return AppendOperator(next_node), {}