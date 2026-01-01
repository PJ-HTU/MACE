from src.problems.jssp.components import *

def longest_processing_time_lpt_e2c3(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Longest Processing Time (LPT) heuristic for JSSP.

    This constructive heuristic selects the schedulable job whose next operation has the longest processing time.
    It builds the solution incrementally by advancing one job at a time using the AdvanceOperator.
    The selection prioritizes jobs based on the processing time of their immediate next operation.
    In case of ties (multiple jobs with the same maximum processing time), the job with the lowest job ID is chosen.
    This heuristic aims to process longer operations earlier to potentially balance machine loads and reduce makespan.

    Hyperparameters:
    None required. All kwargs are ignored, with no default values needed since the algorithm is parameter-free.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - get_schedulable_jobs (callable): Returns a list of job IDs that have remaining operations to schedule. Used to identify candidate jobs for selection.
            - get_next_operation_time (callable): Returns the processing time of the next unscheduled operation for a given job ID. Used to evaluate and compare processing times for schedulable jobs.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no items are necessary or modified; it is passed through unchanged.

    Returns:
        tuple[AdvanceOperator, dict]: Returns an AdvanceOperator instance for the selected job ID if schedulable jobs exist, along with an empty dict for updated algorithm data (no updates performed).
        If no schedulable jobs remain (all operations are scheduled), returns (None, {}) to indicate termination of construction.

    The algorithm proceeds as follows:
    1. Retrieve the list of schedulable job IDs using get_schedulable_jobs().
    2. If the list is empty, return (None, {}) as no further advancement is possible.
    3. For each schedulable job, compute the next operation's processing time using get_next_operation_time(job_id).
    4. Identify the job with the maximum processing time. If multiple jobs tie for the maximum, select the one with the lowest job ID.
    5. Create and return an AdvanceOperator for the selected job ID, which will append the job to the appropriate machine's sequence and increment the job's operation index.
    6. The returned operator ensures validity: it only advances schedulable jobs, preventing re-finished operations or violations of job sequencing constraints.
    No modifications are made to problem_state or algorithm_data directly; updates are returned separately.
    """
    # Step 1: Get the list of schedulable jobs
    schedulable_jobs = problem_state['get_schedulable_jobs']()
    
    # Step 2: Check if there are any schedulable jobs
    if not schedulable_jobs:
        # No further operations to schedule; construction is complete
        return None, {}
    
    # Step 3: Evaluate next operation times for all schedulable jobs
    # Create a list of (processing_time, -job_id) tuples to handle ties by preferring lowest job ID (using -job_id for max selection)
    candidates = []
    for job_id in schedulable_jobs:
        try:
            next_time = problem_state['get_next_operation_time'](job_id)
            # Use -job_id to ensure lowest job ID wins ties when maximizing (processing_time, -job_id)
            candidates.append((next_time, -job_id, job_id))
        except IndexError:
            # Skip if job has no remaining operations (though get_schedulable_jobs should prevent this)
            continue
    
    # Step 4: If no valid candidates (edge case), return None
    if not candidates:
        return None, {}
    
    # Step 5: Select the candidate with the maximum (processing_time, -job_id)
    # This selects the max processing_time, and for ties, the max -job_id (i.e., min job_id)
    best_candidate = max(candidates)
    selected_job_id = best_candidate[2]
    
    # Step 6: Create the AdvanceOperator for the selected job
    operator = AdvanceOperator(selected_job_id)
    
    # Step 7: Return the operator and empty updated algorithm data (no updates needed)
    return operator, {}