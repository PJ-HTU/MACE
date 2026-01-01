from src.problems.jssp.components import AdvanceOperator
import numpy as np

def lwr_e480(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """This heuristic, Least Work Remaining (LWR), selects the schedulable job with the least total remaining processing time among all unscheduled operations for that job. It aims to prioritize jobs that are close to completion, potentially balancing the workload by finishing jobs quickly. This is a constructive heuristic that builds the solution incrementally by choosing the next job to advance based on minimal remaining workload.

    The algorithm proceeds as follows:
    1. Retrieve the list of schedulable jobs using the 'get_schedulable_jobs' helper function, which identifies jobs with at least one unscheduled operation.
    2. If no jobs are schedulable (e.g., the solution is complete), return None (no operator) and an empty algorithm_data dict, indicating no further actions are possible.
    3. For each schedulable job, compute the total remaining processing time using the 'get_total_remaining_time' helper function.
    4. Identify the job(s) with the minimal remaining time. In case of ties, select the job with the smallest job ID to ensure deterministic behavior.
    5. Return an AdvanceOperator for the selected job to advance its next operation, and an empty dict for algorithm_data (no updates needed for this heuristic).

    This heuristic assumes the current solution is partially constructed and valid, and it does not modify any data in problem_state. It uses helper functions to query the state without side effects. No hyper-parameters are used in this algorithm, so kwargs are ignored. The result is always valid as AdvanceOperator only advances eligible jobs without violating precedence or constraints.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - get_schedulable_jobs (callable): Function to get the list of schedulable job IDs based on the current solution.
            - get_total_remaining_time (callable): Function to compute the total remaining processing time for a given job.
            - current_solution (Solution): The current partial solution to pass to the helper functions.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. No items are necessary for this heuristic, as it does not track or update any data across steps.
        **kwargs: No hyper-parameters are used in this algorithm, so any provided kwargs are ignored.

    Returns:
        AdvanceOperator: An operator to advance the selected job's next operation. If no schedulable jobs exist, returns None.
        dict: An empty dictionary, as this heuristic does not update algorithm_data.
    """
    # Retrieve the list of schedulable jobs from the problem state.
    schedulable_jobs = problem_state['get_schedulable_jobs'](problem_state['current_solution'])
    
    # If there are no schedulable jobs, the solution is complete or invalid for advancement; return no operator.
    if not schedulable_jobs:
        return None, {}
    
    # Compute the total remaining processing time for each schedulable job.
    remaining_times = {job: problem_state['get_total_remaining_time'](job, problem_state['current_solution']) for job in schedulable_jobs}
    
    # Find the minimum remaining time among all schedulable jobs.
    min_remaining = min(remaining_times.values())
    
    # Identify candidate jobs with the minimum remaining time.
    candidates = [job for job, time in remaining_times.items() if time == min_remaining]
    
    # Select the job with the smallest ID in case of ties for deterministic selection.
    selected_job = min(candidates)
    
    # Return the AdvanceOperator for the selected job and an empty dict (no algorithm data updates).
    return AdvanceOperator(selected_job), {}