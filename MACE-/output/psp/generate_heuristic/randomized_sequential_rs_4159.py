from src.problems.psp.components import *
from typing import Union, Tuple, List, Dict, Any
import random

def randomized_sequential_rs_4159(problem_state: dict, algorithm_data: dict, **kwargs) -> Tuple[Union[CompleteVesselAssignmentOperator, UnassignVesselOperator, None], Dict[str, Any]]:
    """
    Randomized Sequential (RS) heuristic for the Port Scheduling Problem (PSP). This constructive heuristic builds a solution sequentially by processing unassigned vessels in a random order. It starts from the current partial solution and attempts to assign each vessel in the permuted order by sampling random inbound start times within the vessel's time window, checking resource availability for berths and tugboats, and using greedy tugboat combinations. It applies tight scheduling where berthing starts immediately after inbound service completion and outbound starts immediately after berthing. The first feasible assignment found across samples and compatible berths is selected. If no feasible assignment is found for a vessel, it explicitly unassigns it (redundant but as per design). The process is iterative across function calls: algorithm_data maintains the random permutation of unassigned vessels and the current processing index. If the permutation is exhausted or there are no unassigned vessels, no operator is returned.

    This heuristic is diversity-oriented, relying on randomization for exploration, and ensures validity by checking availability constraints before proposing an assignment. It considers partial solutions as input, only targeting currently unassigned vessels. Costs are implicitly handled via feasibility (unserved penalties apply if unassigned; other costs via calculate_assignment_cost if needed, but here focused on feasibility). The result is valid as per helper functions' guarantees.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current partial solution instance, used as starting point for assignments.
            - get_unassigned_vessels (callable): Function to retrieve list of unassigned vessel IDs from the current solution.
            - get_vessel_time_window (callable): Function to get (earliest_start, latest_start) for a vessel's inbound service.
            - get_compatible_berths (callable): Function to get list of compatible berth IDs for a vessel.
            - is_berth_available (callable): Function to check berth availability for a time slot.
            - is_tugboat_available (callable): Function to check tugboat availability including preparation time.
            - find_tugboat_combination (callable): Function to find a valid tugboat combination for inbound/outbound service.
            - vessel_sizes (numpy.ndarray): Array of vessel sizes (S_i) for compatibility checks.
            - vessel_durations (numpy.ndarray): Array of berthing durations (D_i).
            - vessel_inbound_service_times (numpy.ndarray): Array of inbound service durations (tau^in_i).
            - vessel_outbound_service_times (numpy.ndarray): Array of outbound service durations (tau^out_i).
            - vessel_horsepower_requirements (numpy.ndarray): Array of required horsepower (P^req_i).
            - inbound_preparation_time (int): Preparation time after inbound service (rho^in).
            - outbound_preparation_time (int): Preparation time after outbound service (rho^out).
            - time_constraint_tolerance (float): Tolerance for timing sequences (epsilon_time), used implicitly in tight scheduling.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are used:
            - permutation (list[int]): Stored random permutation of unassigned vessel IDs; if absent, a new one is generated.
            - current_index (int): Current index in the permutation to process; defaults to 0 if absent.
        **kwargs: Hyper-parameters for the algorithm.
            - sample_size (int, default=3): Number of random inbound start times to sample for each vessel attempt.

    Returns:
        Tuple[Union[CompleteVesselAssignmentOperator, UnassignVesselOperator, None], dict]:
            - If there are unassigned vessels and index < len(permutation), returns the operator (CompleteVesselAssignmentOperator for feasible assignment or UnassignVesselOperator if no feasible option found) for the current vessel, and updated algorithm_data with advanced current_index.
            - If no unassigned vessels or permutation exhausted (current_index >= len(permutation)), returns (None, {}).
            - Updated algorithm_data contains 'permutation' and 'current_index' for sequential processing; empty dict if no update needed.
    """
    # Extract necessary components from problem_state (never modify problem_state)
    current_solution = problem_state['current_solution']
    get_unassigned_vessels = problem_state['get_unassigned_vessels']
    get_vessel_time_window = problem_state['get_vessel_time_window']
    get_compatible_berths = problem_state['get_compatible_berths']
    is_berth_available = problem_state['is_berth_available']
    is_tugboat_available = problem_state['is_tugboat_available']
    find_tugboat_combination = problem_state['find_tugboat_combination']
    vessel_sizes = problem_state['vessel_sizes']
    vessel_durations = problem_state['vessel_durations']
    vessel_inbound_service_times = problem_state['vessel_inbound_service_times']
    vessel_outbound_service_times = problem_state['vessel_outbound_service_times']
    vessel_horsepower_requirements = problem_state['vessel_horsepower_requirements']
    inbound_preparation_time = problem_state['inbound_preparation_time']
    outbound_preparation_time = problem_state['outbound_preparation_time']
    time_constraint_tolerance = problem_state['time_constraint_tolerance']

    # Hyper-parameter with default
    sample_size = kwargs.get('sample_size', 3)

    # Get current unassigned vessels
    unassigned_vessels = get_unassigned_vessels(current_solution)
    if not unassigned_vessels:
        # No vessels to assign; heuristic complete
        return None, {}

    # Initialize or retrieve permutation and index from algorithm_data
    if 'permutation' not in algorithm_data or algorithm_data.get('current_index', 0) >= len(algorithm_data['permutation']):
        # Generate new random permutation of current unassigned vessels
        permutation = list(unassigned_vessels)
        random.shuffle(permutation)
        current_index = 0
        updated_algorithm_data = {'permutation': permutation, 'current_index': current_index}
    else:
        # Use existing
        permutation = algorithm_data['permutation']
        current_index = algorithm_data['current_index']
        updated_algorithm_data = {'permutation': permutation, 'current_index': current_index}

    if current_index >= len(permutation):
        # Permutation exhausted; check if still unassigned (unlikely, but safe)
        return None, {}

    # Get the next vessel to process
    vessel_id = permutation[current_index]

    # Check if still unassigned (in case of external changes)
    current_unassigned = get_unassigned_vessels(current_solution)
    if vessel_id not in current_unassigned:
        # Already assigned; skip to next
        updated_algorithm_data['current_index'] = current_index + 1
        return None, updated_algorithm_data  # No operator, just advance

    # Attempt to find a feasible assignment for this vessel
    time_window = get_vessel_time_window(vessel_id)
    earliest_inbound, latest_inbound = time_window
    if earliest_inbound > latest_inbound:
        # Invalid window; cannot assign
        unassign_op = UnassignVesselOperator(vessel_id)
        updated_algorithm_data['current_index'] = current_index + 1
        return unassign_op, updated_algorithm_data

    tau_in_i = vessel_inbound_service_times[vessel_id]
    D_i = vessel_durations[vessel_id]
    tau_out_i = vessel_outbound_service_times[vessel_id]
    compatible_berths = get_compatible_berths(vessel_id)

    feasible_found = False
    for _ in range(sample_size):
        if feasible_found:
            break
        # Sample random inbound start time
        inbound_start = random.randint(earliest_inbound, latest_inbound)

        # Set tight scheduling: berthing starts immediately after inbound
        berthing_start = inbound_start + tau_in_i
        # Outbound starts immediately after berthing (within tolerance, but tight=0 <= epsilon)
        outbound_start = berthing_start + D_i

        # Try each compatible berth
        for berth_id in compatible_berths:
            if feasible_found:
                break
            # Check berth availability for berthing period
            if not is_berth_available(berth_id, berthing_start, D_i, current_solution):
                continue

            # Find inbound tugboat combination at inbound_start
            inbound_tugs, _ = find_tugboat_combination(vessel_id, inbound_start, 'inbound', current_solution)
            if inbound_tugs is None:
                continue

            # Check availability for each inbound tugboat
            all_inbound_available = all(
                is_tugboat_available(tug_id, inbound_start, tau_in_i, inbound_preparation_time, current_solution)
                for tug_id, _ in inbound_tugs
            )
            if not all_inbound_available:
                continue

            # Find outbound tugboat combination at outbound_start
            outbound_tugs, _ = find_tugboat_combination(vessel_id, outbound_start, 'outbound', current_solution)
            if outbound_tugs is None:
                continue

            # Check availability for each outbound tugboat
            all_outbound_available = all(
                is_tugboat_available(tug_id, outbound_start, tau_out_i, outbound_preparation_time, current_solution)
                for tug_id, _ in outbound_tugs
            )
            if not all_outbound_available:
                continue

            # All checks passed: this is a feasible assignment
            # Prepare tugboat lists with start times
            inbound_tugboats = [(tug_id, inbound_start) for tug_id, _ in inbound_tugs]
            outbound_tugboats = [(tug_id, outbound_start) for tug_id, _ in outbound_tugs]

            # Create the assignment operator (ensures same start times for multi-tugboats)
            assign_op = CompleteVesselAssignmentOperator(
                vessel_id, berth_id, berthing_start, inbound_tugboats, outbound_tugboats
            )
            feasible_found = True
            # Advance index
            updated_algorithm_data['current_index'] = current_index + 1
            return assign_op, updated_algorithm_data

    # No feasible assignment found after all samples and berths
    unassign_op = UnassignVesselOperator(vessel_id)
    updated_algorithm_data['current_index'] = current_index + 1
    return unassign_op, updated_algorithm_data