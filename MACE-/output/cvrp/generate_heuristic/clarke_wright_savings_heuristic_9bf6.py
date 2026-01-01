from src.problems.cvrp.components import *
def clarke_wright_savings_heuristic_9bf6(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """Clarke-Wright Savings Heuristic adapted as a sequential constructive method for CVRP.
    This heuristic builds the solution step-by-step by greedily adding unvisited customers to existing routes or starting new routes based on computed savings.
    It mimics the merging idea of Clarke-Wright by preferring to extend routes where the saving (compared to serving the customer separately) is maximized.
    Savings for appending customer j to a route ending at i: distance(i, depot) + distance(depot, j) - distance(i, j).
    The process continues until all customers are served. Each call adds one customer.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): 2D array representing distances between all nodes (used to compute savings and select closest for new routes).
            - depot (int): The index of the depot node (used in savings calculations and selecting customers for new routes).
            - current_solution (Solution): The current partial solution instance (to access routes and determine route ends).
            - get_unvisited_customers (callable): Function to retrieve the list of unvisited customer nodes (to identify candidates for addition).
            - can_add_to_route (callable): Function to check if a customer can be added to a specific vehicle's route without exceeding capacity.
        algorithm_data (dict): Not used in this implementation, as the heuristic is stateless and recomputes based on the current solution each time.
        **kwargs: Hyper-parameters for the algorithm:
            - saving_threshold (float, optional): The minimum savings value required to prefer extending an existing route over starting a new one. If the best extension saving is below this threshold and a new route is possible, a new route is started instead. Defaults to 0.0. Lower values (e.g., negative) make it more aggressive in extending routes.

    Returns:
        tuple[AppendOperator or None, dict]: Returns an AppendOperator to add the next customer to the best route (or new vehicle) based on savings, which when applied will produce a valid partial solution. If all customers are served (no unvisited), returns (None, {}). The second element is always an empty dict as no updates to algorithm_data are needed. If no feasible action is possible (e.g., all routes full, no more vehicles, unvisited remain), returns (None, {}), though instances are assumed feasible so this should only occur when complete.
    """
    # Extract necessary data from problem_state (never modify problem_state)
    distance_matrix = problem_state['distance_matrix']
    depot = problem_state['depot']
    current_solution = problem_state['current_solution']
    get_unvisited_customers = problem_state['get_unvisited_customers']
    can_add_to_route = problem_state['can_add_to_route']

    # Hyper-parameter: saving_threshold controls when to prefer extension vs. new route
    # Default: 0.0 (extend if saving >= 0, else consider new route if possible)
    saving_threshold = kwargs.get('saving_threshold', 0.0)

    # Get unvisited customers; if none, solution is complete
    unvisited = get_unvisited_customers(current_solution)
    if not unvisited:
        # No more customers to serve; return None to indicate completion
        return None, {}

    # Get current routes for analysis
    routes = current_solution.routes
    vehicle_num = len(routes)  # Assume routes list length equals vehicle_num

    # Find the best extension: max saving for appending to an existing route
    best_extension_saving = float('-inf')
    best_vid = -1
    best_node = -1
    has_feasible_extension = False

    # Iterate over all vehicles with non-empty routes
    for vid in range(vehicle_num):
        route = routes[vid]
        if len(route) == 0:  # Skip empty routes
            continue
        end = route[-1]  # Last customer in the route (end point)
        # Check each unvisited customer for this route
        for node in unvisited:
            # Only consider if capacity allows addition
            if can_add_to_route(vid, node, current_solution):
                has_feasible_extension = True
                # Compute saving: dist(end, depot) + dist(depot, node) - dist(end, node)
                saving = (distance_matrix[end][depot] +
                          distance_matrix[depot][node] -
                          distance_matrix[end][node])
                # Update if this is the best saving found
                if saving > best_extension_saving:
                    best_extension_saving = saving
                    best_vid = vid
                    best_node = node

    # Find available vehicles for new route (empty routes)
    available_vids = [vid for vid in range(vehicle_num) if len(routes[vid]) == 0]

    # Decide on action based on best extension and threshold
    if has_feasible_extension and best_extension_saving >= saving_threshold:
        # Prefer extending the route with the highest saving >= threshold
        # This appends to the end, producing a valid extension
        return AppendOperator(best_vid, best_node), {}
    elif available_vids:
        # No good extension (saving < threshold) or no extension possible; start a new route if possible
        # Select the unvisited customer closest to depot (minimize initial cost: 2 * dist(depot, node))
        # Since unvisited non-empty (checked earlier), min is safe
        best_node_new = min(unvisited, key=lambda n: distance_matrix[depot][n])
        # Use the first available vehicle ID
        next_vid = min(available_vids)
        # This starts a new valid single-customer route
        return AppendOperator(next_vid, best_node_new), {}
    elif has_feasible_extension:
        # No new route possible, but extensions available; use the best even if saving < threshold
        # Ensures progress toward serving all customers
        return AppendOperator(best_vid, best_node), {}
    else:
        # No feasible extensions and cannot start new route (e.g., no available vehicles, all routes full)
        # Per assumptions, instances are feasible, so this indicates completion or error; return None
        # But only truly complete if unvisited empty (already checked earlier)
        return None, {}