from src.problems.jssp.components import AdvanceOperator

def lpt_11ce(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """ 
    This heuristic implements the Longest Processing Time (LPT) rule for the Job Shop Scheduling Problem (JSSP).
    It is a constructive heuristic that builds the schedule step-by-step by selecting the schedulable job with the longest processing time for its next operation to advance.
    The algorithm proceeds as follows:
    1. Retrieve the list of schedulable jobs using the helper function get_schedulable_jobs.
    2. For each schedulable job, compute the processing time of its next operation using get_next_operation_time.
    3. Identify the job with the maximum next operation processing time. In case of ties, the first encountered job in the list is selected.
    4. If there are schedulable jobs, return an AdvanceOperator for the selected job to schedule its next operation.
    5. If no schedulable jobs are available (i.e., all jobs are fully scheduled), return None to indicate no further operations can be advanced.
    This ensures the solution remains valid by only advancing operations that can be scheduled without violating job precedence constraints.
    No hyper-parameters are used in this algorithm, so kwargs are ignored and defaults are not applicable.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current solution instance to check schedulable jobs and operation times.
            - get_schedulable_jobs (callable): Helper function to get the list of schedulable job IDs.
            - get_next_operation_time (callable): Helper function to get the processing time of the next operation for a job.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. No items are necessary for this algorithm.

    Returns:
        AdvanceOperator or None: The operator to advance the selected job's next operation, or None if no schedulable jobs exist.
        dict: An empty dictionary, as no algorithm data is updated.
    """
    # Retrieve the list of schedulable jobs from the current solution
    schedulable_jobs = problem_state['get_schedulable_jobs'](solution=problem_state['current_solution'])
    
    if not schedulable_jobs:
        # No schedulable jobs available, return None to indicate no operator can be applied
        return None, {}
    
    # Initialize variables to track the job with the longest next operation time
    max_time = -1
    selected_job = None
    
    # Iterate through schedulable jobs to find the one with the maximum next operation time
    for job_id in schedulable_jobs:
        next_time = problem_state['get_next_operation_time'](job_id, solution=problem_state['current_solution'])
        if next_time > max_time:
            max_time = next_time
            selected_job = job_id
    
    # Return the AdvanceOperator for the selected job, and an empty dict for algorithm_data
    return AdvanceOperator(selected_job), {}