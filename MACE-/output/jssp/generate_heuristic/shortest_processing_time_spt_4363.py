from src.problems.jssp.components import *

def shortest_processing_time_spt_4363(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Shortest Processing Time (SPT) heuristic for JSSP. This constructive heuristic selects the schedulable job whose next operation has the smallest processing time. In case of ties (multiple jobs with the same smallest processing time), it selects the job with the lowest job ID to ensure determinism. This approach prioritizes shorter operations to potentially reduce machine idle times and improve overall schedule efficiency.

    The algorithm proceeds as follows:
    1. Retrieve the list of currently schedulable jobs using the provided callable. A schedulable job is one that has at least one remaining unscheduled operation (i.e., job_operation_index[job_id] < len(job_operation_sequence[job_id])).
    2. If no schedulable jobs are available (all operations are scheduled), return None and an empty dict, indicating the construction phase is complete and no further operator can be applied.
    3. For each schedulable job, compute the processing time of its next operation using the provided callable. This time is looked up from job_operation_time[job_id][job_operation_index[job_id]].
    4. Identify the job with the minimum next operation processing time. If multiple jobs tie for the minimum, select the one with the smallest job ID.
    5. Instantiate an AdvanceOperator for the selected job ID. This operator will advance the job by appending its next operation to the corresponding machine's sequence and incrementing the job's operation index.
    6. The returned operator, when applied, ensures validity by respecting job operation sequences and machine assignments without allowing re-finished operations or violations, as it only advances unfinished jobs.
    7. No modifications are made to problem_state or algorithm_data. The algorithm does not require or update any algorithm-specific data, so an empty dict is returned for the second output.

    Hyper-parameters: This algorithm has no hyper-parameters, so **kwargs is unused. Defaults are implicitly set by the selection logic (e.g., tie-breaking by lowest job ID).

    Necessary items from problem_state:
        - get_schedulable_jobs (callable): Returns the list of job IDs with remaining operations. Uses current_solution if solution is None.
        - get_next_operation_time (callable): Returns the processing time of the next operation for a given job ID. Uses current_solution if solution is None. Raises IndexError if no remaining operations, but this is avoided by checking schedulable_jobs first.
        - current_solution (Solution): The current partial solution instance, used implicitly by the callables to track job_operation_index and job_operation_sequence.

    algorithm_data (dict): Not used or modified by this algorithm, as no persistent state is required across iterations.

    Returns:
        AdvanceOperator: The operator instance for advancing the selected job's next operation, or None if no schedulable jobs remain (construction complete).
        dict: An empty dictionary, as no updates to algorithm data are needed.
    """
    # Step 1: Get the list of schedulable jobs
    schedulable_jobs = problem_state['get_schedulable_jobs']()
    
    # Step 2: Check if there are any schedulable jobs
    if not schedulable_jobs:
        # No more operations to schedule; construction is complete
        return None, {}
    
    # Step 3 & 4: Find the job with the smallest next operation time, breaking ties by lowest job ID
    # Collect (job_id, next_time) pairs for all schedulable jobs
    candidates = []
    for job_id in schedulable_jobs:
        next_time = problem_state['get_next_operation_time'](job_id)
        candidates.append((job_id, next_time))
    
    # Since schedulable_jobs is non-empty, candidates will not be empty
    # Sort by next_time ascending, then by job_id ascending for ties
    candidates.sort(key=lambda x: (x[1], x[0]))
    
    # Select the first (best) candidate
    selected_job_id, _ = candidates[0]
    
    # Step 5: Create the AdvanceOperator for the selected job
    operator = AdvanceOperator(selected_job_id)
    
    # Step 6 & 7: Return the operator and empty dict (no updates)
    return operator, {}