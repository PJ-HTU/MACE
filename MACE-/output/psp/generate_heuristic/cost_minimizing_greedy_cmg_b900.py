from src.problems.psp.components import *
import random

def cost_minimizing_greedy_cmg_b900(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[CompleteVesselAssignmentOperator, dict]:
    """
    Cost-Minimizing Greedy (CMG) heuristic for the Port Scheduling Problem (PSP). This constructive heuristic starts from a partial solution and iteratively assigns one unassigned vessel at a time. In each invocation, it randomly selects one unassigned vessel, generates up to num_candidates feasible complete assignments (berth + multi-tugboat inbound/outbound services) using ETA-centered search, selects the one with the minimal assignment cost (which minimizes the marginal increase to the total scheduling cost by replacing the unserved penalty with assignment costs Z2+Z3+Z4), and returns the corresponding CompleteVesselAssignmentOperator to apply. The marginal cost increase is calculated as (assignment_cost - unserved_penalty), where unserved_penalty = lambda_1 * M * alpha_i; negative values indicate improvements. If no unassigned vessels remain or no feasible assignment is found for the selected vessel, returns (None, {}). This enables step-by-step construction over multiple calls, with random selection ensuring diverse ordering across invocations. The result is valid as find_feasible_assignments ensures constraint satisfaction (timing, capacity, horsepower, availability).

    Hyper-parameters in kwargs:
        - num_candidates (int, default=5): Maximum number of candidate assignments to generate via find_feasible_assignments (passed as max_results). Higher values increase exploration but computational time.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current partial solution instance to check availability and build upon.
            - total_scheduling_cost (float): Current total scheduling cost, used contextually for marginal increase understanding (not directly modified).
            - get_unassigned_vessels (callable): Function to retrieve list of unassigned vessel IDs from the current solution.
            - find_feasible_assignments (callable): Function to generate feasible assignment candidates for a vessel, returning list of dicts with 'berth_id', 'start_time', 'inbound_tugboats', 'outbound_tugboats', 'total_cost'.
            - objective_weights (numpy.ndarray): Array of lambda coefficients [lambda_1, ...] for objective components.
            - penalty_parameter (float): Large penalty M for unserved vessels in Z1.
            - vessel_priority_weights (numpy.ndarray): Array of alpha_i priorities for vessels.
            - vessel_num (int): Total number of vessels (for ID validation).
            - vessel_durations (numpy.ndarray): Array of D_i berthing durations (accessible if needed for internal checks, though not directly used).
            (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no specific items are necessary; it operates independently per call.

    Returns:
        tuple[CompleteVesselAssignmentOperator or None, dict]: Returns the operator to assign the selected vessel with the best candidate if feasible, along with updated algorithm_data containing 'selected_vessel' (int): ID of assigned vessel, 'marginal_cost' (float): Marginal increase to total_scheduling_cost, 'assignment_cost' (float): Z2+Z3+Z4 cost of the assignment. If no action possible (no unassigned vessels or no feasible candidates), returns (None, {}).
    """
    # Hyper-parameter: number of candidates to generate (default 5)
    num_candidates = kwargs.get('num_candidates', 5)

    # Step 1: Retrieve unassigned vessels from the current partial solution
    unassigned_vessels = problem_state['get_unassigned_vessels'](solution=problem_state['current_solution'])
    
    # If no unassigned vessels, no operator to apply (solution is complete)
    if not unassigned_vessels:
        return None, {}

    # Step 2: Randomly select one unassigned vessel to process (ensures random order across multiple heuristic calls)
    vessel_id = random.choice(unassigned_vessels)

    # Step 3: Generate feasible candidate assignments for the selected vessel using the helper
    # This searches ETA-centered times, checks resource availability, and ensures all constraints (horsepower, timing, etc.)
    candidates = problem_state['find_feasible_assignments'](
        vessel_id=vessel_id,
        max_results=num_candidates,
        solution=problem_state['current_solution']
    )
    
    # If no feasible candidates found (e.g., due to resource conflicts or time window issues), skip this vessel
    if not candidates:
        return None, {}

    # Step 4: Select the best candidate with minimal assignment cost (Z2 + Z3 + Z4)
    # Since unserved penalty is fixed for this vessel, minimizing assignment_cost minimizes marginal increase
    best_candidate = min(candidates, key=lambda cand: cand['total_cost'])

    # Step 5: Compute unserved penalty for this vessel to derive marginal cost increase
    # Penalty = lambda_1 * M * alpha_i (part of current Z1 in total_scheduling_cost)
    lambda_1 = problem_state['objective_weights'][0]
    M = problem_state['penalty_parameter']
    alpha_i = problem_state['vessel_priority_weights'][vessel_id]
    unserved_penalty = lambda_1 * M * alpha_i
    assignment_cost = best_candidate['total_cost']
    marginal_cost = assignment_cost - unserved_penalty  # Could be negative (improvement)

    # Step 6: Create the operator to apply the best assignment
    # Note: inbound_tugboats and outbound_tugboats are lists of (tugboat_id, start_time) tuples, all at same start_time per service
    operator = CompleteVesselAssignmentOperator(
        vessel_id=vessel_id,
        berth_id=best_candidate['berth_id'],
        start_time=best_candidate['start_time'],
        inbound_tugboats=best_candidate['inbound_tugboats'],
        outbound_tugboats=best_candidate['outbound_tugboats']
    )

    # Step 7: Return operator and updated algorithm_data for tracking (e.g., for hyper-heuristic analysis)
    updated_data = {
        'selected_vessel': vessel_id,
        'marginal_cost': marginal_cost,
        'assignment_cost': assignment_cost
    }

    return operator, updated_data