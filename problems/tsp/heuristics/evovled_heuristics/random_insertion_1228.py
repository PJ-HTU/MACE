from src.problems.tsp.components import *
from typing import Optional
import random

def random_insertion_1228(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[Optional[BaseOperator], dict]:
    """Random Insertion heuristic for TSP: Builds the tour by adding unvisited nodes randomly, using append for early stages (progress_ratio < 0.5) to reduce bias and insert at random positions later for diversity.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current solution instance representing the partial tour.
            - get_unvisited_nodes (callable): Function to retrieve the list of unvisited node IDs.
            - progress_ratio (float): Ratio of visited cities to total cities, used to decide between append and insert operations.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are optional but used for state:
            - rng (random.Random): Persistent random number generator instance for reproducible randomness across calls.
        **kwargs: Hyper-parameters for the algorithm.
            - random_seed (int, default=42): Seed for initializing the random number generator if 'rng' is not provided in algorithm_data.

    Returns:
        tuple[Optional[BaseOperator], dict]: 
            - operator (Optional[BaseOperator]): An AppendOperator or InsertOperator to add a random unvisited node; None if the tour is already complete (no unvisited nodes).
            - updated_algorithm_data (dict): Updated algorithm data; includes {'rng': random.Random} if the RNG was initialized in this call, otherwise an empty dict {}. This ensures the random state persists for future calls without modifying the input algorithm_data directly.
    """
    # Retrieve necessary components from problem_state without modification
    current_solution = problem_state['current_solution']
    get_unvisited_nodes = problem_state['get_unvisited_nodes']
    progress_ratio = problem_state['progress_ratio']
    
    # Step 1: Check if the tour is complete by getting unvisited nodes
    # If no unvisited nodes, return None to indicate no further action needed (tour is complete)
    unvisited_nodes = get_unvisited_nodes()
    if not unvisited_nodes:
        return None, {}
    
    # Step 2: Initialize or retrieve the random number generator for reproducibility
    # Hyper-parameter 'random_seed' defaults to 42; used only if 'rng' not in algorithm_data
    # This ensures consistent random selections across multiple calls to this heuristic
    if 'rng' not in algorithm_data:
        random_seed = kwargs.get('random_seed', 42)
        rng = random.Random(random_seed)
        updated_algorithm_data = {'rng': rng}
    else:
        rng = algorithm_data['rng']
        updated_algorithm_data = {}
    
    # Step 3: Select a random unvisited node using the RNG
    # This ensures the node is not already in the tour, maintaining validity
    selected_node = rng.choice(unvisited_nodes)
    
    # Step 4: Decide between append or insert based on progress to avoid early randomness bias
    # If tour is empty (length 0) or progress_ratio < 0.5, use append to build sequentially at start
    # Otherwise, insert at a random position (0 to current length inclusive) for diversity in later stages
    # Note: Insert at position == len(tour) effectively appends, but uses InsertOperator for consistency
    # This strategy ensures the solution remains valid as we only add unvisited nodes
    tour_length = len(current_solution.tour)
    if tour_length == 0 or progress_ratio < 0.5:
        operator = AppendOperator(selected_node)
    else:
        insert_position = rng.randint(0, tour_length)
        operator = InsertOperator(selected_node, insert_position)
    
    # Step 5: Return the operator and updated data
    # The operator will be applied externally; no validation needed here as construction guarantees validity (no duplicates, all nodes visited eventually)
    # If called repeatedly until unvisited is empty, the full tour will be complete and valid
    return operator, updated_algorithm_data