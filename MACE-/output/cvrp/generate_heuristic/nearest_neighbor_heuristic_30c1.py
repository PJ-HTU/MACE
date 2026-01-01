from src.problems.cvrp.components import *
import numpy as np

def nearest_neighbor_heuristic_30c1(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """
    Description for this heuristic algorithm.
    This implements the Nearest Neighbor Heuristic for CVRP, a constructive greedy method that builds the solution by sequentially filling vehicles.
    It starts with the first vehicle (ID 0) and repeatedly appends the closest unvisited customer (measured by distance_matrix from the last node in the route or depot if empty) that fits the remaining capacity.
    When no more customers can be added to the current vehicle, it advances to the next vehicle.
    The process continues across vehicles until all customers are served.
    This is step-by-step: each call returns an operator to add one customer or updates the state to switch vehicles without an operator.
    Hyper-parameters:
    - max_iter_per_vehicle (int, default=problem_state['node_num']): Maximum iterations (additions) per vehicle to prevent potential loops, though unlikely in this setup.
    The algorithm proceeds as follows:
    1. Retrieve unvisited customers using get_unvisited_customers.
    2. If no unvisited customers, return None, {} (all served).
    3. Get or default current_vehicle from algorithm_data (starts at 0).
    4. Compute the last_node: depot if route empty, else last customer in route.
    5. Find feasible candidates: unvisited customers that can_add_to_route for current_vehicle.
    6. If candidates exist, select the one with minimum distance from last_node using distance_matrix.
       - Create and return AppendOperator(current_vehicle, selected_customer), with algorithm_data updated to keep current_vehicle.
       - Check against max_iter_per_vehicle: track 'adds_per_vehicle' in algorithm_data; if exceeded, switch vehicle.
    7. If no candidates for current_vehicle:
       - Attempt to advance to next_vehicle = current_vehicle + 1.
       - If next_vehicle < vehicle_num, return None, {'current_vehicle': next_vehicle}.
       - If at end of vehicles but unvisited remain, search for any vehicle that can accept at least one unvisited customer (starting from 0 for wrap-around).
         - If found (prefer smallest ID), return None, {'current_vehicle': that_vehicle}.
         - If no vehicle can accept any (infeasible instance), return None, {}.
    8. The algorithm ensures progress toward visiting all customers by always attempting to add if possible; returns None only if all served or truly infeasible.
    9. Uses distance_matrix for proximity (no Euclidean needed as coords not provided).
    10. Maintains validity via can_add_to_route (capacity check) and unvisited selection; assumes instance is feasible (total capacity >= total demand, no demand > capacity).

    Args:
        problem_state (dict): The dictionary contains the problem state.. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): 2D array representing distances between all nodes (used for nearest selection).
            - vehicle_num (int): Total number of available vehicles (defines route list length).
            - capacity (int): Capacity per vehicle (used implicitly via helpers).
            - depot (int): Index of the depot node (starting point for empty routes).
            - current_solution (Solution): Current partial solution to extend (routes list of lists).
            - get_unvisited_customers (callable): Retrieves list of unvisited customer IDs (excludes depot).
            - can_add_to_route (callable): Checks if a customer can be added to a vehicle without exceeding capacity.
            - get_route_load (callable): Computes current load of a vehicle's route (used if needed for remaining capacity checks).
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - current_vehicle (int, optional): ID of the vehicle currently being filled (0 to vehicle_num-1); defaults to 0 if absent.
            - adds_per_vehicle (dict, optional): Tracks additions per vehicle {vehicle_id: count}; used for max_iter enforcement.
        kwargs: Hyper-parameters with defaults.
            - max_iter_per_vehicle (int): Max additions per vehicle; default = problem_state['node_num'].

    Returns:
        AppendOperator: Instance to append the selected nearest customer to the current vehicle's route.
        dict: Updated algorithm_data with 'current_vehicle' and 'adds_per_vehicle' if applicable.
        Or (None, dict): When switching vehicles, updates 'current_vehicle'.
        Or (None, {}): When all customers served or no feasible addition possible (infeasible case).
    """
    # Retrieve necessary data from problem_state (never modify problem_state)
    distance_matrix = problem_state['distance_matrix']
    vehicle_num = problem_state['vehicle_num']
    depot = problem_state['depot']
    sol = problem_state['current_solution']
    get_unvisited = problem_state['get_unvisited_customers']
    can_add = problem_state['can_add_to_route']
    get_load = problem_state['get_route_load']
    node_num = problem_state['node_num']  # For default max_iter

    # Hyper-parameters with defaults
    max_iter_per_vehicle = kwargs.get('max_iter_per_vehicle', node_num)

    # Get unvisited customers
    unvisited = get_unvisited(sol)
    if not unvisited:
        return None, {}  # All customers served, no operator needed

    # Get or initialize state from algorithm_data
    current_vehicle = algorithm_data.get('current_vehicle', 0)
    adds_per_vehicle = algorithm_data.get('adds_per_vehicle', {})

    # Ensure adds_per_vehicle is a dict
    if current_vehicle not in adds_per_vehicle:
        adds_per_vehicle[current_vehicle] = 0

    # Compute last_node for current vehicle
    routes = sol.routes
    if len(routes[current_vehicle]) == 0:
        last_node = depot
    else:
        last_node = routes[current_vehicle][-1]

    # Get feasible candidates for current vehicle
    candidates = [c for c in unvisited if can_add(current_vehicle, c, sol)]

    # Check if max iterations reached for this vehicle
    if adds_per_vehicle[current_vehicle] >= max_iter_per_vehicle and candidates:
        # Switch vehicle even if candidates exist (enforce limit)
        candidates = []

    if candidates:
        # Select the nearest candidate
        # Compute distances, handle empty (though checked)
        dists = [distance_matrix[last_node, c] for c in candidates]
        # No need for empty check as candidates non-empty
        min_dist_idx = np.argmin(dists)
        selected_customer = candidates[min_dist_idx]

        # Create operator
        operator = AppendOperator(current_vehicle, selected_customer)

        # Update adds count
        adds_per_vehicle[current_vehicle] += 1

        # Updated algorithm_data: keep current vehicle
        updated_data = {
            'current_vehicle': current_vehicle,
            'adds_per_vehicle': adds_per_vehicle
        }
        return operator, updated_data
    else:
        # Cannot add to current vehicle, try to advance
        next_vehicle = current_vehicle + 1
        if next_vehicle < vehicle_num:
            # Simply advance
            updated_data = {
                'current_vehicle': next_vehicle,
                'adds_per_vehicle': adds_per_vehicle  # Preserve counts
            }
            # Reset adds for new vehicle if switching
            if next_vehicle not in adds_per_vehicle:
                adds_per_vehicle[next_vehicle] = 0
            return None, updated_data

        # At end of vehicles, search for any vehicle that can accept at least one unvisited
        found_vehicle = None
        for v in range(vehicle_num):
            v_candidates = [c for c in unvisited if can_add(v, c, sol)]
            if v_candidates:
                # Check if not at max iter
                v_adds = adds_per_vehicle.get(v, 0)
                if v_adds < max_iter_per_vehicle:
                    found_vehicle = v
                    break  # Prefer smallest ID

        if found_vehicle is not None:
            # Update adds if needed (though < max)
            if found_vehicle not in adds_per_vehicle:
                adds_per_vehicle[found_vehicle] = 0
            updated_data = {
                'current_vehicle': found_vehicle,
                'adds_per_vehicle': adds_per_vehicle
            }
            return None, updated_data
        else:
            # No vehicle can add any unvisited (infeasible or all served, but checked earlier)
            # Per CRITICAL, only here if truly cannot progress
            return None, {}