from src.problems.psp.components import *
import numpy as np

def earliest_arrival_first_eaf_4fe1(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[CompleteVesselAssignmentOperator, dict]:
    """
    Earliest Arrival First (EAF) heuristic: Selects the unassigned vessel with the smallest ETA and attempts to assign it to the earliest feasible berth-time slot within its time window, preferring berths with the highest remaining capacity (time_periods - current occupied duration). Uses find_feasible_assignments to generate candidate assignments and selects the best one based on earliest start_time and then highest remaining capacity. Returns a CompleteVesselAssignmentOperator for this assignment if feasible; otherwise, returns None if no unassigned vessels remain or no feasible assignment exists for the selected vessel. This enables step-by-step construction on partial solutions by iteratively assigning one vessel per heuristic call.

    Hyperparameters (in kwargs, with defaults):
    - max_candidates (int, default=5): Maximum number of candidate assignments to generate via find_feasible_assignments for the selected vessel. Higher values increase search thoroughness but computational cost.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current partial solution to extend.
            - total_scheduling_cost (float): Current total cost (used for tracking but not modified here).
            - vessel_num (int): Total number of vessels.
            - time_periods (int): Total scheduling horizon for remaining capacity calculation.
            - vessel_etas (numpy.ndarray): ETAs for sorting unassigned vessels.
            - vessel_durations (numpy.ndarray): Berthing durations for occupied time computation.
            - vessel_sizes (numpy.ndarray): Vessel sizes for compatibility (used indirectly via helpers).
            - berth_capacities (numpy.ndarray): Berth capacities for preference in ties (used if needed).
            - get_unassigned_vessels (callable): To retrieve list of unassigned vessel IDs.
            - find_feasible_assignments (callable): To generate feasible assignment candidates for the vessel.
        algorithm_data (dict): Not used in this algorithm; returned as empty dict.
        **kwargs: Hyperparameters as described above.

    Returns:
        CompleteVesselAssignmentOperator or None: Operator to assign the selected vessel if a feasible assignment is found; None otherwise (no action if all vessels assigned or no feasible slot).
        dict: Empty dict as no algorithm data is updated by this heuristic.
    """
    # Extract necessary data from problem_state (never modify original)
    current_solution = problem_state['current_solution']
    vessel_etas = problem_state['vessel_etas']
    vessel_durations = problem_state['vessel_durations']
    time_periods = problem_state['time_periods']
    get_unassigned_vessels_func = problem_state['get_unassigned_vessels']
    find_feasible_assignments_func = problem_state['find_feasible_assignments']

    # Hyperparameters with defaults
    max_candidates = kwargs.get('max_candidates', 5)

    # Step 1: Get and sort unassigned vessels by increasing ETA
    unassigned_vessels = get_unassigned_vessels_func(current_solution)
    if not unassigned_vessels:
        # No unassigned vessels; no action needed
        return None, {}

    # Sort by ETA ascending; select the earliest one for this step (enables iterative calls)
    sorted_unassigned = sorted(unassigned_vessels, key=lambda v: vessel_etas[v])
    selected_vessel = sorted_unassigned[0]

    # Step 2: Compute current occupied durations per berth for remaining capacity
    # Initialize occupied array for each berth (0 to berth_num-1, but size from context)
    # Assume berth_num from problem_state; use np.zeros for safety
    berth_num = problem_state['berth_num']
    occupied_durations = np.zeros(berth_num)
    vessel_assignments = current_solution.vessel_assignments
    for v_id, assignment in vessel_assignments.items():
        if assignment is not None:
            berth_id, _ = assignment
            occupied_durations[berth_id] += vessel_durations[v_id]

    remaining_capacities = time_periods - occupied_durations

    # Step 3: Generate feasible assignment candidates for the selected vessel
    candidates = find_feasible_assignments_func(selected_vessel, max_results=max_candidates, solution=current_solution)
    if not candidates:
        # No feasible assignment for this vessel; no action
        return None, {}

    # Step 4: Select the best candidate: earliest start_time, then highest remaining capacity
    # Sort candidates: primary key start_time asc, secondary remaining[berth_id] desc
    def sort_key(cand):
        s_time = cand['start_time']
        b_id = cand['berth_id']
        rem_cap = remaining_capacities[b_id]
        return (s_time, -rem_cap)  # Negative for descending remaining

    best_candidate = min(candidates, key=sort_key)

    # Step 5: Ensure the assignment is valid (redundant check via helper, but confirm non-empty tugboats)
    inbound_tugs = best_candidate['inbound_tugboats']
    outbound_tugs = best_candidate['outbound_tugboats']
    if not inbound_tugs or not outbound_tugs:
        return None, {}

    # All tugboats for inbound/outbound must start at same time (as per operator validation)
    inbound_times = [t for _, t in inbound_tugs]
    outbound_times = [t for _, t in outbound_tugs]
    if len(set(inbound_times)) > 1 or len(set(outbound_times)) > 1:
        return None, {}

    # Step 6: Create the operator for the best assignment
    operator = CompleteVesselAssignmentOperator(
        vessel_id=selected_vessel,
        berth_id=best_candidate['berth_id'],
        start_time=best_candidate['start_time'],
        inbound_tugboats=inbound_tugs,
        outbound_tugboats=outbound_tugs
    )

    # The operator will be applied externally; no direct modification here
    # Costs are tracked via total_scheduling_cost update post-application
    return operator, {}