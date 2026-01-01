from src.problems.jssp.components import *
import random

def random_selection_b73a(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Random Selection heuristic for JSSP: Randomly selects one job from the list of schedulable jobs to advance its next operation using AdvanceOperator.
    This is a constructive heuristic that builds the schedule incrementally by randomly choosing which job to advance at each step, promoting diversity in solution exploration.
    No hyperparameters are required; all kwargs are ignored as they are not used in this algorithm.
    The algorithm proceeds as follows:
    1. Retrieve the list of schedulable jobs using the get_schedulable_jobs callable from problem_state. A schedulable job has remaining operations to schedule.
    2. If the list is empty (all operations are scheduled), return None, {} to indicate no further action is possible (construction phase complete).
    3. Otherwise, randomly select one job_id from the schedulable jobs list using random.choice.
    4. Create an AdvanceOperator instance with the selected job_id, which will append the next operation of that job to the appropriate machine's sequence.
    5. Return the operator and an empty dict, as this heuristic does not update or require any algorithm_data.
    This ensures validity by only advancing schedulable jobs, respecting job operation sequences and avoiding re-finished operations.
    The result is always valid for the construction phase, as AdvanceOperator only advances unfinished operations.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - get_schedulable_jobs (callable): Returns list[int] of job IDs with remaining operations; used to identify candidates for random selection.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no items are necessary or modified.

    Returns:
        AdvanceOperator: The operator instance for advancing the randomly selected job's next operation, or None if no schedulable jobs remain.
        dict: Empty dictionary {}, as no updates to algorithm data are performed.
    """
    # Retrieve schedulable jobs without modifying problem_state
    schedulable_jobs = problem_state['get_schedulable_jobs']()
    
    # Check if there are any schedulable jobs; if not, return None to indicate completion
    if not schedulable_jobs:
        return None, {}
    
    # Randomly select a job_id from the list
    selected_job_id = random.choice(schedulable_jobs)
    
    # Create the AdvanceOperator for the selected job
    operator = AdvanceOperator(selected_job_id)
    
    # Return operator and empty algorithm_data dict
    return operator, {}