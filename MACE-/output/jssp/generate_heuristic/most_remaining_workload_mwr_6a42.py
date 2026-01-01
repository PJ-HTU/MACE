from src.problems.jssp.components import *
def most_remaining_workload_mwr_6a42(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Most Remaining Workload (MWR) heuristic for JSSP. This constructive heuristic selects the schedulable job with the largest total remaining processing time to advance its next operation. It prioritizes jobs with the most unfinished work to balance the schedule by handling heavy workloads early. Ties are broken by selecting the job with the lowest job ID to ensure determinism.

    Hyperparameters:
        This algorithm has no hyperparameters, so kwargs are ignored. Default values are implicitly set by not using any.

    Algorithm workflow:
        1. Retrieve the list of currently schedulable jobs using the get_schedulable_jobs callable from problem_state. A schedulable job is one with remaining operations (job_operation_index[job_id] < len(job_operation_sequence[job_id])).
        2. If no schedulable jobs are available (i.e., all operations are scheduled), return None, {} to indicate termination of construction; no further operators can be applied.
        3. For each schedulable job, compute the total remaining processing time using get_total_remaining_time(job_id), which sums processing times from the current operation index to the end of the job's sequence.
        4. Identify the job with the maximum remaining time value. If multiple jobs tie for the maximum, select the one with the smallest job ID.
        5. Create an AdvanceOperator instance for the selected job_id, which will append the job to the sequence of its next machine and increment the job's operation index.
        6. The returned operator ensures validity: it advances only the next operation in sequence per job, prevents re-finishing operations (as it uses current job_operation_index), and respects machine-specific scheduling without overlaps (validation handled externally via validation_solution).
        7. No updates to algorithm_data are needed for this basic heuristic, so return an empty dict. This heuristic does not rely on or modify any persistent state beyond the solution updated by the operator.

    Necessary items from problem_state:
        - get_schedulable_jobs (callable): Returns list[int] of job IDs with remaining operations. Uses current_solution if solution=None.
        - get_total_remaining_time (callable): Returns int, the sum of processing times for remaining operations of a job. Uses current_solution if solution=None. Raises IndexError if no remaining operations, but we check schedulable first.
        - current_solution (Solution): Implicitly used by the callables to access job_operation_index and job_operation_sequence for determining schedulable status and remaining times.

    algorithm_data (dict): Not used or required by this algorithm; can be empty or arbitrary.

    Returns:
        AdvanceOperator: Instance for the selected job_id to advance its next operation, or None if no schedulable jobs.
        dict: Empty dict {} as no algorithm data is updated.
    """
    # Step 1: Get schedulable jobs
    schedulable_jobs = problem_state['get_schedulable_jobs']()
    
    # Step 2: Check if any schedulable jobs exist
    if not schedulable_jobs:
        # No more operations to schedule; construction complete
        return None, {}
    
    # Step 3 & 4: Find job with max remaining time, tie-break by min job ID
    max_remaining_time = -1
    selected_job_id = float('inf')  # Initialize to a large value for min tie-break
    
    for job_id in schedulable_jobs:
        remaining_time = problem_state['get_total_remaining_time'](job_id)
        if remaining_time > max_remaining_time or (remaining_time == max_remaining_time and job_id < selected_job_id):
            max_remaining_time = remaining_time
            selected_job_id = job_id
    
    # Step 5: Create operator for selected job
    operator = AdvanceOperator(selected_job_id)
    
    # Step 6 & 7: No algorithm data updates
    return operator, {}