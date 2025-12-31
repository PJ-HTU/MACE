from src.problems.tsp.components import *
import random

def random_greedy_2da2(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BaseOperator, dict]:
    """Random Greedy heuristic for TSP: Builds the tour constructively by first appending a random starting node if the tour is empty. For subsequent steps, randomly selects an unvisited node and inserts it at the position that minimizes the insertion cost, adding greediness to the random selection for partial optimization. This introduces stochastic diversity while ensuring progressive construction. The algorithm terminates (returns None) when all nodes are visited, ensuring a complete and valid tour upon full execution. It handles partial solutions, starting from empty if needed, and uses helpers for efficiency without modifying any input data.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - node_num (int): The total number of nodes in the problem, used to select initial random node if tour is empty.
            - current_solution (Solution): Current partial tour solution, checked for emptiness and length to determine action (append vs. insert).
            - get_unvisited_nodes (callable): Returns list of unvisited node IDs, used to select random unvisited node for insertion.
            - get_insertion_cost (callable): Computes cost increase for inserting a node at a position, used to find the best insertion position minimizing cost.
        algorithm_data (dict): Not used in this algorithm as no persistent state is required beyond the current_solution in problem_state.
        **kwargs: Hyper-parameters with defaults.
            - random_seed (int or None, default=None): Seed for the random module to ensure reproducibility of random node selections. If provided, sets random.seed(random_seed) at the start of the function.

    Returns:
        BaseOperator or None: Returns an AppendOperator for the initial random node if the tour is empty, or an InsertOperator for the randomly selected unvisited node at its best insertion position otherwise. Returns None if the tour already visits all nodes (complete solution), indicating no further action needed.
        dict: Empty dict {} as this algorithm does not update or require persistent algorithm_data.
    """
    # Set random seed for reproducibility if provided in kwargs
    random_seed = kwargs.get('random_seed', None)
    if random_seed is not None:
        random.seed(random_seed)

    # Extract necessary items from problem_state (do not modify)
    node_num = problem_state['node_num']
    current_solution = problem_state['current_solution']
    get_unvisited_nodes = problem_state['get_unvisited_nodes']
    get_insertion_cost = problem_state['get_insertion_cost']

    # Check if tour is empty: append a random starting node
    if len(current_solution.tour) == 0:
        # Select a random node from 0 to node_num-1 as starting point
        start_node = random.randint(0, node_num - 1)
        # Return AppendOperator to add it; insertion cost for empty is 0.0, but use append as per heuristic
        return AppendOperator(start_node), {}

    # Get unvisited nodes; if none, tour is complete, return None
    unvisited_nodes = get_unvisited_nodes()
    if not unvisited_nodes:
        # All nodes visited: no operator needed, solution is complete and valid
        return None, {}

    # Randomly select one unvisited node
    selected_node = random.choice(unvisited_nodes)

    # Find the best insertion position: evaluate all possible positions (0 to len(tour))
    tour_length = len(current_solution.tour)
    min_cost = float('inf')
    best_position = None

    for position in range(tour_length + 1):
        # Get insertion cost; for empty tour this wouldn't be called, but handles prepend/append naturally
        insertion_cost = get_insertion_cost(selected_node, position)
        # Check if this is the minimum cost position (handles inf or valid floats)
        if insertion_cost < min_cost:
            min_cost = insertion_cost
            best_position = position

    # If a valid best position found (always should be if unvisited exist), return InsertOperator
    # Note: get_insertion_cost returns 0.0 or valid float, never inf here since tour non-empty
    if best_position is not None:
        return InsertOperator(selected_node, best_position), {}
    else:
        # Unreachable in normal cases, but return None if no valid insertion (e.g., invalid state)
        return None, {}