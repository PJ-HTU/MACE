import os
import time
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_function

class SingleHyperHeuristic:
    """Controller for executing a single heuristic algorithm."""
    
    def __init__(
        self,
        heuristic: str,
        problem: str,
        iterations_scale_factor: float=1.0,
        penalty_value: float=1e10,
    ) -> None:
        self.heuristic = load_function(heuristic, problem=problem)
        self.iterations_scale_factor = iterations_scale_factor
        self.penalty_value = penalty_value
    
    def run(self, env: BaseEnv, time_limit: float=None, **kwargs) -> bool:
        """Run the heuristic algorithm in the environment.
        
        Returns:
            bool: True if a valid complete solution is obtained, False otherwise.
        """
        current_steps = 0
        max_steps = int(env.construction_steps * self.iterations_scale_factor) * 1
        
        start_time = time.time()
        timeout_occurred = False
        
        while current_steps <= max_steps and env.continue_run:
            if time_limit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time > time_limit:
                    timeout_occurred = True
                    print(f"Solution timeout ({time_limit}s), returning penalty value")
                    break
            
            _ = env.run_heuristic(self.heuristic)
            current_steps += 1
        
        if timeout_occurred:
            return False
        
        return env.is_complete_solution and env.validation_solution