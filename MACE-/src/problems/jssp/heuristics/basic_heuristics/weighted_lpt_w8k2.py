from src.problems.jssp.components import *
import random

def weighted_lpt_w8k2(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Weighted Longest Processing Time (LPT) heuristic for JSSP.

    This constructive heuristic selects a schedulable job probabilistically, where the selection probability is proportional to the processing time of its next operation.
    Jobs with longer next operations are more likely to be chosen, incorporating the balancing strength of LPT while introducing randomness for diversity and exploration.
    This hybrid approach aims to combine efficiency (prioritizing long operations) with adaptability (randomness to handle ties, stagnation, or varied instance distributions).
    In case of no schedulable jobs, returns None to indicate completion.
    No hyperparameters are used; kwargs are ignored.

    Args:
        problem_state (dict): The dictionary contains the problem state. Necessary items:
            - get_schedulable_jobs (callable): Returns list of schedulable job IDs.
            - get_next_operation_time (callable): Returns processing time of next operation for a job ID.
        algorithm_data (dict): Not used or modified; passed through as empty dict.

    Returns:
        tuple[AdvanceOperator, dict]: AdvanceOperator for selected job if possible, else (None, {}).

    Algorithm steps:
    1. Retrieve schedulable jobs.
    2. If none, return None, {}.
    3. Compute processing times as weights for each job.
    4. Select one job using weighted random choice.
    5. Return AdvanceOperator and empty algorithm_data.
    """
    schedulable_jobs = problem_state['get_schedulable_jobs']()
    if not schedulable_jobs:
        return None, {}
    
    weights = []
    for job_id in schedulable_jobs:
        try:
            time = problem_state['get_next_operation_time'](job_id)
            weights.append(time)
        except IndexError:
            continue
    
    if not weights:
        return None, {}
    
    selected_job_id = random.choices(schedulable_jobs, weights=weights, k=1)[0]
    operator = AdvanceOperator(selected_job_id)
    return operator, {}