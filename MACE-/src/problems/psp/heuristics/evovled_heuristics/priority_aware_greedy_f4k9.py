from src.problems.psp.components import *
import random

def priority_aware_greedy_f4k9(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[CompleteVesselAssignmentOperator, dict]:
    """ This heuristic enhances the greedy construction approach by incorporating vessel priority weights to better address the unserved vessel penalty (Z1) in the objective function. While still focusing on feasible assignments with low marginal costs (Z2, Z3, Z4), it prioritizes assigning high-priority vessels first among unassigned ones, selecting the one with the highest priority weight (α_i) and, in case of ties, the lowest total assignment cost. This ensures quicker reduction of Z1 penalties for critical vessels, improving overall solution quality in early stages of construction. The algorithm iteratively builds the solution step-by-step, returning a CompleteVesselAssignmentOperator for the best candidate or None if no feasible assignments remain. Feasibility is maintained through find_feasible_assignments.

    Hyper-parameters:
        - max_eval_vessels (int, default=None): Limits the number of unassigned vessels to evaluate. If None, evaluates all; else, randomly samples up to that many for efficiency.
        - priority_tie_breaker (bool, default=True): If True, when multiple vessels have the same priority, breaks ties by selecting the one with the lowest assignment cost; if False, randomly selects among highest-priority vessels.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current solution instance.
            - vessel_priority_weights (numpy.ndarray): 1D array of priority weights for each vessel (α_i).
            - get_unassigned_vessels (callable): Function to get list of unassigned vessel IDs.
            - find_feasible_assignments (callable): Function to find feasible assignments for a vessel.
            - unserved_vessel_ratio (float): Fraction of unassigned vessels to gauge progress.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - last_assigned_vessel (int or None): ID of the last assigned vessel for potential adaptive behavior (initially None).
        **kwargs: Hyper-parameters as described above.

    Returns:
        CompleteVesselAssignmentOperator: The operator to assign the selected high-priority vessel's complete feasible service, or None if no feasible assignment exists.
        dict: Updated algorithm_data with 'last_assigned_vessel' set to the assigned vessel ID, or unchanged if none.
    """
    
    # Hyper-parameter setup with defaults
    max_eval_vessels = kwargs.get('max_eval_vessels', None)
    priority_tie_breaker = kwargs.get('priority_tie_breaker', True)

    # Extract necessary components from problem_state
    current_solution = problem_state['current_solution']
    vessel_priority_weights = problem_state['vessel_priority_weights']
    get_unassigned_vessels = problem_state['get_unassigned_vessels']
    find_feasible_assignments = problem_state['find_feasible_assignments']
    unserved_vessel_ratio = problem_state['unserved_vessel_ratio']
    
    # Extract from algorithm_data
    last_assigned_vessel = algorithm_data.get('last_assigned_vessel', None)

    # Step 1: Get list of currently unassigned vessels
    unassigned_vessels = get_unassigned_vessels(current_solution)
    if not unassigned_vessels:
        return None, algorithm_data  # No updates needed

    # Step 2: Determine vessels to evaluate (all or sampled)
    vessels_to_eval = unassigned_vessels
    if max_eval_vessels is not None and len(unassigned_vessels) > max_eval_vessels:
        random.shuffle(unassigned_vessels)
        vessels_to_eval = unassigned_vessels[:max_eval_vessels]

    # Step 3: Evaluate feasible assignments and collect candidates with priority
    candidates = []  # List of (priority, cost, vessel_id, assignment_dict) tuples
    for vessel_id in vessels_to_eval:
        feasible_assignments = find_feasible_assignments(vessel_id, current_solution)
        if feasible_assignments:
            assignment_dict = feasible_assignments[0]
            total_cost = assignment_dict['total_cost']
            priority = vessel_priority_weights[vessel_id]
            candidates.append((priority, total_cost, vessel_id, assignment_dict))

    # Step 4: Check if any feasible candidates found
    if not candidates:
        return None, algorithm_data

    # Step 5: Select the best vessel based on priority, then cost
    # Sort by priority descending, then cost ascending if tie_breaker is on
    if priority_tie_breaker:
        candidates.sort(key=lambda x: (-x[0], x[1]))  # Higher priority first, then lower cost
    else:
        # Group by priority and pick highest, then random among them
        max_priority = max(c[0] for c in candidates)
        highest_priority_candidates = [c for c in candidates if c[0] == max_priority]
        best_candidate = random.choice(highest_priority_candidates)
        candidates = [best_candidate]  # Treat as the only one

    best_priority, best_cost, best_vessel_id, best_assignment = candidates[0]

    # Step 6: Extract assignment details
    berth_id = best_assignment['berth_id']
    berth_start_time = best_assignment['start_time']
    inbound_tugboats = best_assignment['inbound_tugboats']
    outbound_tugboats = best_assignment['outbound_tugboats']

    # Step 7: Create the operator
    operator = CompleteVesselAssignmentOperator(
        vessel_id=best_vessel_id,
        berth_id=berth_id,
        start_time=berth_start_time,
        inbound_tugboats=inbound_tugboats,
        outbound_tugboats=outbound_tugboats
    )

    # Step 8: Update algorithm_data with last assigned vessel
    updated_algorithm_data = algorithm_data.copy()
    updated_algorithm_data['last_assigned_vessel'] = best_vessel_id

    return operator, updated_algorithm_data