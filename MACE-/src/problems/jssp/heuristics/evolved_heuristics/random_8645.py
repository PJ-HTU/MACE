from src.problems.jssp.components import *
import random

def random_8645(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    This heuristic implements a random selection strategy for the Job Shop Scheduling Problem (JSSP). It randomly selects a schedulable job from the list of jobs that have remaining operations to schedule and applies the AdvanceOperator to advance that job by one operation. This is a constructive heuristic that builds the solution step-by-step without any optimization criteria, purely using randomness for selection.

    Hyper-parameters in kwargs:
    - seed (int, default=42): The random seed for reproducibility of the random selection. If not provided, defaults to 42.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - get_schedulable_jobs (callable): A helper function to retrieve the list of job IDs that have remaining operations to schedule. This is used to get the pool of jobs from which to randomly select.
            - current_solution (Solution): The current partial solution instance, passed implicitly to get_schedulable_jobs if needed.
        algorithm_data (dict): Not used in this algorithm, as it does not require maintaining any algorithm-specific state across calls.

    Returns:
        If there are schedulable jobs, returns an AdvanceOperator instance for the randomly selected job and an empty dict (no updates to algorithm_data).
        If there are no schedulable jobs (i.e., all jobs are fully scheduled), returns None and an empty dict, indicating no further operations can be advanced.
    """
    # Set default value for hyper-parameters
    seed = kwargs.get('seed', 42)
    random.seed(seed)
    
    # Retrieve the list of schedulable jobs using the helper function
    schedulable_jobs = problem_state['get_schedulable_jobs'](problem_state['current_solution'])
    
    # If no schedulable jobs, return None to indicate no operator can be applied
    if not schedulable_jobs:
        return None, {}
    
    # Randomly select one job from the list of schedulable jobs
    selected_job = random.choice(schedulable_jobs)
    
    # Create and return the AdvanceOperator for the selected job, with no algorithm_data updates
    return AdvanceOperator(selected_job), {}