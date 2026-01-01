from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution for the Job Shop Scheduling Problem (JSSP).
    A list of lists where each sublist represents the sequence of jobs scheduled on a machine, identified by its index in the outer list.
    Each job is represented by its integer identifier and is processed in the order specified within the sublist.
    Each sublist in 'job_sequences' corresponds to a machine's job processing sequence, with machines sorted by their identifier (index in the outer list).
    Each job within a sublist is sorted by its scheduled order of execution on that machine.
    Job matrix records the target operation(machine) sequence in job.
    """
    def __init__(self, job_sequences: list[list[int]], job_operation_sequence: list[list[int]], job_operation_index: list[int]):
        self.job_sequences = job_sequences
        self.job_operation_sequence = job_operation_sequence
        self.job_operation_index = job_operation_index

    def __str__(self) -> str:
        job_sequences_str = ""
        for machine_index, jobs in enumerate(self.job_sequences):
            job_sequences_str += f"machine_{machine_index}: " + "->".join(map(str, jobs)) + "\n"
        return job_sequences_str


class AdvanceOperator(BaseOperator):
    """AdvanceOperator is responsible for advancing the next operation in the job's processing sequence.
    This operator appends the next operation of the specified job to the processing queue of the machine that is scheduled to execute the operation.
    It identifies the appropriate machine based on the job's current operation index and the job matrix within the provided Solution instance."""
    def __init__(self, job_id: int):
        self.job_id = job_id

    def run(self, solution: Solution) -> Solution:
        new_job_sequences = [ops[:] for ops in solution.job_sequences]
        machine_id = solution.job_operation_sequence[self.job_id][solution.job_operation_index[self.job_id]]
        new_job_sequences[machine_id].append(self.job_id)
        job_operation_index = solution.job_operation_index
        job_operation_index[self.job_id] += 1
        return Solution(new_job_sequences, solution.job_operation_sequence, job_operation_index)