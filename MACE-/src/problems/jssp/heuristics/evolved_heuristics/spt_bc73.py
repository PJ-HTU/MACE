from src.problems.jssp.components import *
def spt_bc73(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """ Implements the Shortest Processing Time (SPT) heuristic for Job Shop Scheduling Problem (JSSP).
    This heuristic selects the schedulable job with the shortest processing time for its next operation and advances it using AdvanceOperator.
    It is a constructive heuristic that builds the schedule step-by-step by prioritizing jobs based on immediate operation duration to minimize idle times or bottlenecks.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - get_schedulable_jobs (callable): Helper function to retrieve the list of job IDs that have remaining operations to schedule.
            - get_next_operation_time (callable): Helper function to get the processing time of the next unscheduled operation for a given job ID.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. No items are necessary for this heuristic.

    Returns:
        AdvanceOperator: The operator to advance the selected job with the shortest next operation time. If no schedulable jobs are available (e.g., all operations are scheduled), returns None.
        dict: An empty dictionary, as this heuristic does not update algorithm_data.
    """
    # No hyper-parameters are required for this heuristic.
    # Retrieve helper functions from problem_state.
    get_schedulable_jobs_func = problem_state['get_schedulable_jobs']
    get_next_operation_time_func = problem_state['get_next_operation_time']
    
    # Get the list of schedulable jobs using the helper function.
    schedulable_jobs = get_schedulable_jobs_func()
    
    # If there are no schedulable jobs, return None and empty dict (no operator to apply).
    if not schedulable_jobs:
        return None, {}
    
    # Initialize variables to track the job with the shortest next operation time.
    selected_job = None
    min_time = float('inf')
    
    # Iterate through each schedulable job to find the one with the smallest next operation time.
    for job_id in schedulable_jobs:
        next_time = get_next_operation_time_func(job_id)
        if next_time < min_time:
            min_time = next_time
            selected_job = job_id
    
    # Create and return the AdvanceOperator for the selected job, along with empty algorithm_data.
    # This ensures the next operation of the job with the shortest processing time is scheduled.
    return AdvanceOperator(selected_job), {}