from src.problems.jssp.components import *

def least_remaining_workload_lwr_d24d(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Least Remaining Workload (LWR) heuristic for JSSP.

    This constructive heuristic selects the schedulable job with the smallest total remaining processing time
    to advance its next operation. It prioritizes jobs that have the least work left to complete, aiming to
    balance the schedule by finishing shorter jobs sooner. In case of ties in remaining time, it selects
    the job with the lowest job ID for determinism.

    Hyper-parameters:
        None. This algorithm has no hyperparameters; all behavior is deterministic based on the problem state.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - get_schedulable_jobs (callable): Returns the list of job IDs with remaining operations to schedule.
            - get_total_remaining_time (callable): Computes the total remaining processing time for a given job ID.
            - current_solution (Solution): The current partial solution instance, used implicitly by the callables.

        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no items are necessary or used.

    The algorithm proceeds as follows:
    1. Retrieve the list of schedulable jobs using get_schedulable_jobs(). This identifies jobs with unfinished operations.
    2. If no schedulable jobs are available (i.e., the schedule is complete), return None, {} to indicate termination.
    3. For each schedulable job, compute its total remaining processing time using get_total_remaining_time(job_id).
    4. Select the job with the minimum remaining time. If multiple jobs tie, choose the one with the smallest job ID.
    5. Create an AdvanceOperator for the selected job to append its next operation to the corresponding machine's sequence.
    6. Return the operator and an empty dict (no updates to algorithm_data are needed for this heuristic).

    This ensures validity by only advancing schedulable jobs, respecting job operation sequences and avoiding re-finished operations.
    The returned operator will produce a valid partial solution when applied, as it follows the predefined AdvanceOperator logic.
    """
    # Step 1: Get the list of schedulable jobs
    schedulable_jobs = problem_state['get_schedulable_jobs']()
    
    # Step 2: Check if there are any schedulable jobs
    if not schedulable_jobs:
        # No more operations to schedule; return None to indicate completion
        return None, {}
    
    # Step 3: Compute remaining times for all schedulable jobs and track the best candidate
    best_job_id = None
    min_remaining_time = float('inf')
    
    for job_id in schedulable_jobs:
        remaining_time = problem_state['get_total_remaining_time'](job_id)
        # Update best if this job has smaller remaining time, or equal but smaller ID
        if remaining_time < min_remaining_time or (remaining_time == min_remaining_time and job_id < best_job_id):
            min_remaining_time = remaining_time
            best_job_id = job_id
    
    # Step 4: At this point, best_job_id is guaranteed to exist since schedulable_jobs is non-empty
    # Create the operator to advance the selected job
    operator = AdvanceOperator(best_job_id)
    
    # Step 5: No updates to algorithm_data; return empty dict
    return operator, {}