from src.problems.cvrp.components import *
import numpy as np

def cheapest_insertion_heuristic_c124(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Cheapest Insertion Heuristic for CVRP: A constructive heuristic that builds the solution incrementally by repeatedly selecting the unserved customer with the lowest insertion cost (minimal increase in total route distance) and inserting it into the most feasible position in an existing or new route, respecting capacity constraints. This is applied step-by-step in a hyper-heuristic framework, where each call adds one customer until all are served.

    The algorithm proceeds as follows:
    1. Retrieve the current solution and check for unserved customers using get_unvisited_customers.
    2. If no unserved customers, return None, {} to indicate completion.
    3. For each unserved customer k:
       - For each vehicle v (0 to vehicle_num-1):
         - If adding k to v's route violates capacity (via can_add_to_route), skip this vehicle.
         - For each possible insertion position pos (0 to len(route[v]) inclusive):
           - Compute the insertion delta cost considering depot connections:
             - pos == 0 (before first): dist(depot, k) + dist(k, first) - dist(depot, first) if route non-empty, else 2 * dist(depot, k).
             - pos == len(route) (after last): dist(last, k) + dist(k, depot) - dist(last, depot).
             - Otherwise (between route[pos-1] and route[pos]): dist(prev, k) + dist(k, next) - dist(prev, next).
           - Track the minimum delta for this k across all feasible (v, pos).
    4. Select the customer k with the overall smallest minimum delta.
    5. For that k, return the InsertOperator for the corresponding best (v, pos).
    6. If no feasible insertion found for any k (e.g., capacity exhausted but unserved remain, though assuming feasible instances), return None, {}.
    7. The process repeats in subsequent calls until all customers are served, ensuring a complete and valid solution (validated implicitly via helpers; full validation via validation_solution post-construction).

    Hyper-parameters in kwargs:
    - regret_threshold (float, default=0.1 * problem_state['avg_depot_distance']): A threshold for potential regret-based selection (e.g., difference between best and second-best insertion costs). In this basic implementation, it is not actively used for selection (always selects cheapest), but can be extended for regret-k variants. If provided, it scales with instance size.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current partial or complete solution instance to build upon.
            - distance_matrix (numpy.ndarray): 2D array of distances between all nodes (including depot).
            - vehicle_num (int): Number of available vehicles.
            - capacity (int): Capacity per vehicle.
            - depot (int): Index of the depot node.
            - demands (numpy.ndarray): 1D array of demands for all nodes (depot demand assumed 0).
            - get_unvisited_customers (callable): Function to get list of unvisited customer node IDs.
            - get_route_load (callable): Function to get current load of a vehicle's route.
            - can_add_to_route (callable): Function to check if a customer can be added to a vehicle without exceeding capacity.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no items are necessary (not used).
        **kwargs: Optional hyper-parameters, including 'regret_threshold' (float).

    Returns:
        tuple[InsertOperator, dict]: The InsertOperator instance to insert the selected unserved customer into the best feasible position of a route (new or existing), which minimizes the distance increase while respecting capacity. The second element is an empty dict as no algorithm data is updated in this implementation. Returns (None, {}) if no unserved customers or no feasible insertion possible.
    """
    # Retrieve necessary data from problem_state (never modify problem_state)
    solution = problem_state['current_solution']
    distance_matrix = problem_state['distance_matrix']
    vehicle_num = problem_state['vehicle_num']
    capacity = problem_state['capacity']
    depot = problem_state['depot']
    demands = problem_state['demands']
    get_unvisited_customers_func = problem_state['get_unvisited_customers']
    get_route_load_func = problem_state['get_route_load']
    can_add_to_route_func = problem_state['can_add_to_route']

    # Hyper-parameter defaults
    regret_threshold = kwargs.get('regret_threshold', 0.1 * problem_state.get('avg_depot_distance', 1.0))  # Default based on instance stat; not used in basic selection but available for extensions

    # Get unserved customers
    unvisited = get_unvisited_customers_func(solution)
    if not unvisited:
        return None, {}

    # Initialize tracking for best insertion across all unvisited customers
    best_k = None
    best_vehicle_id = None
    best_position = None
    best_delta = float('inf')

    # For each unvisited customer k, find the minimum feasible insertion delta and best (v, pos) for it
    for k in unvisited:
        min_delta_for_k = float('inf')
        best_v_for_k = None
        best_pos_for_k = None

        for v in range(vehicle_num):
            route = solution.routes[v]
            current_load = get_route_load_func(v, solution)
            customer_demand = demands[k]

            # Check capacity feasibility for this vehicle (independent of position)
            if current_load + customer_demand > capacity:
                continue

            # Now check all possible insertion positions: 0 (before first) to len(route) (after last)
            for pos in range(len(route) + 1):
                if pos == 0:  # Insert at beginning
                    if len(route) == 0:
                        # New route: full cost for depot -> k -> depot
                        delta = 2 * distance_matrix[depot, k]
                    else:
                        # Before first customer
                        a = depot
                        b = route[0]
                        delta = distance_matrix[a, k] + distance_matrix[k, b] - distance_matrix[a, b]
                elif pos == len(route):  # Insert at end
                    if len(route) == 0:
                        # Should not reach here as pos=0 handles empty
                        continue
                    a = route[-1]
                    b = depot
                    delta = distance_matrix[a, k] + distance_matrix[k, b] - distance_matrix[a, b]
                else:  # Insert between route[pos-1] and route[pos]
                    a = route[pos - 1]
                    b = route[pos]
                    delta = distance_matrix[a, k] + distance_matrix[k, b] - distance_matrix[a, b]

                # Update min for this k if better
                if delta < min_delta_for_k:
                    min_delta_for_k = delta
                    best_v_for_k = v
                    best_pos_for_k = pos

        # If a feasible insertion found for this k, compare its min_delta to overall best
        if min_delta_for_k < float('inf') and min_delta_for_k < best_delta:
            best_delta = min_delta_for_k
            best_k = k
            best_vehicle_id = best_v_for_k
            best_position = best_pos_for_k

    # If no feasible insertion found for any k, return None (though assumes feasible problem)
    if best_k is None:
        return None, {}

    # Return the operator for the best insertion (note: InsertOperator works on empty lists via insert(0, node))
    operator = InsertOperator(best_vehicle_id, best_k, best_position)
    return operator, {}