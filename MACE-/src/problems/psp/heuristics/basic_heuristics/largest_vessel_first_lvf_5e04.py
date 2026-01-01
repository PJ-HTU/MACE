from src.problems.psp.components import *
import numpy as np

def largest_vessel_first_lvf_5e04(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[CompleteVesselAssignmentOperator, dict]:
    """Implements the Largest Vessel First (LVF) heuristic for the Port Scheduling Problem (PSP). This constructive heuristic builds upon a partial solution by prioritizing the largest unassigned vessel (by vessel size S_i) and attempting to assign it to a compatible berth with the lowest current utilization. The assignment uses tight scheduling centered on the median of the vessel's inbound time window, with inbound tugboat service starting at the median time, berthing immediately after inbound completion, and outbound tugboat service immediately after berthing completion. Tugboat combinations are found using greedy selection, and availability is verified for berths and all selected tugboats. If a feasible assignment is found for the prioritized vessel, a CompleteVesselAssignmentOperator is returned to apply the assignment; otherwise, None is returned if no unassigned vessels remain or no feasible assignment exists for the largest unassigned vessel. This allows iterative application in a hyper-heuristic framework to build the solution step-by-step. The result ensures validity by checking resource availabilities and compatibility before returning the operator.

    Hyper-parameters:
        - max_berth_tries (int, default=10): Limits the number of compatible berths to try in order of increasing utilization (all if fewer available). Set to a low value for efficiency in large instances.
        - utilization_threshold (float, default=0.8): Skips berths with current utilization above this threshold to prefer less loaded ones. Set to 1.0 to disable.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - 'current_solution' (Solution): The current partial solution to build upon.
            - 'vessel_sizes' (numpy.ndarray): 1D array of vessel sizes for prioritization.
            - 'vessel_durations' (numpy.ndarray): 1D array of berthing durations.
            - 'vessel_inbound_service_times' (numpy.ndarray): 1D array of inbound service durations.
            - 'vessel_outbound_service_times' (numpy.ndarray): 1D array of outbound service durations.
            - 'vessel_horsepower_requirements' (numpy.ndarray): 1D array of required horsepower per vessel.
            - 'berth_capacities' (numpy.ndarray): 1D array of berth capacities for compatibility.
            - 'vessel_etas' (numpy.ndarray): 1D array of ETAs (used indirectly via time windows).
            - 'vessel_early_limits' (numpy.ndarray): 1D array of early limits (used in time windows).
            - 'vessel_late_limits' (numpy.ndarray): 1D array of late limits (used in time windows).
            - 'tugboat_horsepower' (numpy.ndarray): 1D array of tugboat horsepowers.
            - 'inbound_preparation_time' (int): Preparation time after inbound service.
            - 'outbound_preparation_time' (int): Preparation time after outbound service.
            - 'time_periods' (int): Total time horizon for utilization calculation.
            - 'get_unassigned_vessels' (callable): Returns list of unassigned vessel IDs.
            - 'get_vessel_time_window' (callable): Returns (earliest, latest) inbound start times.
            - 'get_compatible_berths' (callable): Returns list of compatible berth IDs.
            - 'is_berth_available' (callable): Checks berth availability for [start, start + duration).
            - 'is_tugboat_available' (callable): Checks tugboat availability including prep time.
            - 'find_tugboat_combination' (callable): Finds tugboat list for service at start time.
        algorithm_data (dict): Not used in this algorithm; can be empty.
        **kwargs: Optional hyper-parameters as described above.

    Returns:
        CompleteVesselAssignmentOperator or None: The operator to assign the largest unassigned vessel if feasible, including berth, berthing start time, and tugboat lists for inbound/outbound; None if no unassigned vessels or no feasible assignment found.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from problem_state (do not modify)
    current_solution = problem_state['current_solution']
    vessel_sizes = problem_state['vessel_sizes']
    vessel_durations = problem_state['vessel_durations']
    vessel_inbound_service_times = problem_state['vessel_inbound_service_times']
    vessel_outbound_service_times = problem_state['vessel_outbound_service_times']
    vessel_horsepower_requirements = problem_state['vessel_horsepower_requirements']
    berth_capacities = problem_state['berth_capacities']
    inbound_prep_time = problem_state['inbound_preparation_time']
    outbound_prep_time = problem_state['outbound_preparation_time']
    time_periods = problem_state['time_periods']
    get_unassigned_vessels = problem_state['get_unassigned_vessels']
    get_vessel_time_window = problem_state['get_vessel_time_window']
    get_compatible_berths = problem_state['get_compatible_berths']
    is_berth_available = problem_state['is_berth_available']
    is_tugboat_available = problem_state['is_tugboat_available']
    find_tugboat_combination = problem_state['find_tugboat_combination']

    # Hyper-parameters with defaults
    max_berth_tries = kwargs.get('max_berth_tries', 10)
    utilization_threshold = kwargs.get('utilization_threshold', 0.8)

    # Step 1: Get list of unassigned vessels
    unassigned_vessels = get_unassigned_vessels(current_solution)
    if not unassigned_vessels:
        # No unassigned vessels; solution is complete, return no operator
        return None, {}

    # Step 2: Prioritize by decreasing vessel size; select the largest unassigned vessel
    # Sort indices by vessel_sizes descending
    sorted_unassigned = sorted(unassigned_vessels, key=lambda vid: vessel_sizes[vid], reverse=True)
    vessel_id = sorted_unassigned[0]  # Largest first

    # Step 3: Get time window for inbound service
    earliest_inbound, latest_inbound = get_vessel_time_window(vessel_id)
    if earliest_inbound > latest_inbound:
        # Invalid time window; cannot assign this vessel
        return None, {}

    # Compute median inbound start time
    median_inbound_start = (earliest_inbound + latest_inbound) // 2

    # Compute tight schedule times
    tau_in = vessel_inbound_service_times[vessel_id]
    D = vessel_durations[vessel_id]
    tau_out = vessel_outbound_service_times[vessel_id]
    berthing_start = median_inbound_start + tau_in
    outbound_start = berthing_start + D

    # Check if schedule fits within time horizon (basic sanity check)
    if outbound_start + tau_out > time_periods:
        # Schedule overruns; try shifting earlier if possible, but for simplicity, skip this vessel
        return None, {}

    # Step 4: Compute current berth utilizations for sorting
    # Initialize utilization array
    berth_util = np.zeros(len(berth_capacities))
    for vid, assignment in current_solution.vessel_assignments.items():
        if assignment is not None:
            bid, _ = assignment
            berth_util[bid] += vessel_durations[vid]
    # Utilization as fraction of time periods
    current_utilizations = berth_util / time_periods if time_periods > 0 else np.zeros_like(berth_util)

    # Step 5: Get compatible berths and sort by increasing utilization (lowest first)
    compatible_berths = get_compatible_berths(vessel_id)
    # Filter by utilization threshold
    compatible_berths = [b for b in compatible_berths if current_utilizations[b] <= utilization_threshold]
    if not compatible_berths:
        return None, {}
    # Sort by increasing utilization
    compatible_berths.sort(key=lambda b: current_utilizations[b])
    # Limit tries
    compatible_berths = compatible_berths[:max_berth_tries]

    # Step 6: For each compatible berth in order, try to find feasible tugboat combinations and check availabilities
    for berth_id in compatible_berths:
        # Check berth availability for berthing period
        if not is_berth_available(berth_id, berthing_start, D, current_solution):
            continue

        # Find inbound tugboat combination at median_inbound_start
        inbound_result = find_tugboat_combination(vessel_id, median_inbound_start, 'inbound', current_solution)
        # Safely handle potential None return or invalid tuple from find_tugboat_combination
        if not isinstance(inbound_result, tuple) or len(inbound_result) != 2 or inbound_result[0] is None:
            continue
        inbound_tugboats = inbound_result[0]

        # Check availability for all inbound tugboats
        inbound_duration = tau_in
        inbound_available = all(
            is_tugboat_available(tug_id, median_inbound_start, inbound_duration, inbound_prep_time, current_solution)
            for tug_id, _ in inbound_tugboats
        )
        if not inbound_available:
            continue

        # Find outbound tugboat combination at outbound_start
        outbound_result = find_tugboat_combination(vessel_id, outbound_start, 'outbound', current_solution)
        # Safely handle potential None return or invalid tuple from find_tugboat_combination
        if not isinstance(outbound_result, tuple) or len(outbound_result) != 2 or outbound_result[0] is None:
            continue
        outbound_tugboats = outbound_result[0]

        # Check availability for all outbound tugboats
        outbound_duration = tau_out
        outbound_available = all(
            is_tugboat_available(tug_id, outbound_start, outbound_duration, outbound_prep_time, current_solution)
            for tug_id, _ in outbound_tugboats
        )
        if not outbound_available:
            continue

        # All checks passed: feasible assignment found
        # Create operator with berthing start, and tugboat lists (all start times are set)
        operator = CompleteVesselAssignmentOperator(
            vessel_id=vessel_id,
            berth_id=berth_id,
            start_time=berthing_start,  # Berthing start time
            inbound_tugboats=inbound_tugboats,  # List of (tug_id, start_time)
            outbound_tugboats=outbound_tugboats  # List of (tug_id, start_time)
        )
        # The operator ensures same start times for multi-tugboats as per its run logic
        return operator, {}

    # No feasible assignment found after trying berths
    return None, {}