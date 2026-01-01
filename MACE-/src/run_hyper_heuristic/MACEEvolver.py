"""
MACE Stage Two: Time-Constrained Complementary Evolution
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Set
import os
import random
from src.run_hyper_heuristic.run_hyper_heuristic import run_hyper_heuristic
from src.run_hyper_heuristic.smoke_test import standalone_smoke_test
from src.run_hyper_heuristic.complementary_selection_simple import complementary_selection_milp


class MACEEvolver:
    """
    MACE Stage Two: Time-Constrained Complementary Evolution
    
    Implements Stage Two of Algorithm 1 from the paper:
    - Generate new heuristics through 5 evolution operators (CI, PI, SI, DI, EI)
    - Use Complementary Selection to choose optimal combinations
    - Support multiple time budget settings
    
    Evolution operator implementation status:
    - CI (Complementary Improvement)   ✅ Implemented
    - PI (Performance Improvement)     ✅ Implemented
    - SI (Specialization Improvement)  ✅ Implemented
    - DI (Diversity Improvement)       ✅ Implemented
    - EI (Efficiency Improvement)      ✅ Implemented
    """
    
    def __init__(
        self,
        problem: str,
        api_key: str,
        heuristic_dir: str,
        task_description_file: str,
        output_dir: str,
        model: str,
        population_size: int = 5,
        max_evaluations: int = 100,
        time_limit: float = 600,
        llm_config_file: str = None,
        iterations_scale_factor: float = 1.0,
        result_dir: str = "result",
        milp_time_limit: float = 600.0
    ):
        """
        Initialize MACE Evolver.
        
        Args:
            problem: Problem type (tsp, jssp, cvrp, psp)
            api_key: LLM API key
            heuristic_dir: Heuristic directory
            task_description_file: Task description file
            output_dir: Output directory
            model: LLM model name
            population_size: Portfolio size n (default=10 in paper)
            max_evaluations: Maximum evaluations Nmax (default=100 in paper)
            time_limit: Time budget Tmax (seconds)
            llm_config_file: LLM configuration file
            iterations_scale_factor: Iteration scaling factor
            result_dir: Result directory
            milp_time_limit: MILP solver time limit (seconds, default=600)
        """
        self.problem = problem
        self.api_key = api_key
        self.heuristic_dir = heuristic_dir
        self.task_description_file = task_description_file
        self.output_dir = output_dir
        self.model = model
        self.n = population_size
        self.Nmax = max_evaluations
        self.time_limit = time_limit
        self.llm_config_file = llm_config_file
        self.iterations_scale_factor = iterations_scale_factor
        self.result_dir = result_dir
        self.milp_time_limit = milp_time_limit
        
        self.evaluations = 0
        
        from src.run_hyper_heuristic.ci_operator import CIOperator
        from src.run_hyper_heuristic.pi_operator import PIOperator
        from src.run_hyper_heuristic.si_operator import SIOperator
        from src.run_hyper_heuristic.di_operator import DIOperator
        from src.run_hyper_heuristic.ei_operator import EIOperator
        
        self.ci_operator = CIOperator(
            problem=problem,
            heuristic_dir=heuristic_dir,
            task_description_file=task_description_file,
            output_dir=output_dir,
            api_key=api_key,
            model=model
        )
        
        self.pi_operator = PIOperator(
            problem=problem,
            heuristic_dir=heuristic_dir,
            task_description_file=task_description_file,
            output_dir=output_dir,
            api_key=api_key,
            model=model
        )
        
        self.si_operator = SIOperator(
            problem=problem,
            heuristic_dir=heuristic_dir,
            task_description_file=task_description_file,
            output_dir=output_dir,
            test_data_dir="output/test_data",
            api_key=api_key,
            model=model
        )

        self.di_operator = DIOperator(
            problem=problem,
            heuristic_dir=heuristic_dir,
            task_description_file=task_description_file,
            output_dir=output_dir,
            api_key=api_key,
            model=model
        )
        
        self.ei_operator = EIOperator(
            problem=problem,
            heuristic_dir=heuristic_dir,
            task_description_file=task_description_file,
            output_dir=output_dir,
            api_key=api_key,
            model=model
        )
        
        print(f"\n{'='*80}")
        print(f"🚀 MACE Evolver Initialization Complete")
        print(f"{'='*80}")
        print(f"Problem type: {problem}")
        print(f"Population size: {population_size}")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Time budget: {time_limit}s")
        print(f"Complementary selection: Gurobi MILP")
        print(f"MILP time limit: {milp_time_limit}s")
        print(f"{'='*80}\n")
    
    def initialize_population(self, results_dict: Dict) -> List[Dict]:
        """
        Initialize population P0 (Algorithm 1, line 7).
        
        Args:
            results_dict: Evaluation results of initial heuristics
                Format: {heuristic_name: [score_1, score_2, ...]}
        
        Returns:
            population: Initial population list
        """
        population = []
        for heuristic_name, heuristic_scores in results_dict.items():
            heuristic_info = {
                'name': heuristic_name,
                'performance_vector': heuristic_scores,
                'avg_performance': np.mean(heuristic_scores)
            }
            population.append(heuristic_info)
        return population
    
    def evolute_population(
        self,
        population: List[Dict],
        results_dict: Dict,
        time_limit: float
    ) -> List[Dict]:
        """
        Evolute Population - Generate n new heuristics (Algorithm 1, lines 10-15).
        
        Current implemented operators:
        - CI: Complementary Improvement
        - PI: Performance Improvement
        - SI: Specialization Improvement
        - DI: Diversity Improvement
        
        Args:
            population: Current population
            results_dict: Performance evaluation results
            time_limit: Time limit
        
        Returns:
            new_heuristics: List of newly generated heuristics
        """
        new_heuristics = []
        attempts = 0
        max_attempts = self.n * 5
        
        print(f"\n{'='*80}")
        print(f"🔄 Starting Evolute Population, target: {self.n} new heuristics")
        print(f"{'='*80}")
        
        while len(new_heuristics) < self.n and attempts < max_attempts:
            attempts += 1
            
            operator = random.choice(['CI', 'PI', 'SI', 'DI'])
            try:
                if operator == 'CI':
                    print(f"\n  🔍 [{operator}] Attempt #{len(new_heuristics)+1}/{self.n} (Total {attempts}/{max_attempts})")
                    
                    results_dict = {
                        h['name']: h['performance_vector']
                        for h in population
                    }
                    file_path, code = self.ci_operator.generate(results_dict)
                    
                elif operator == 'PI':
                    print(f"\n  🔧 [{operator}] Attempt #{len(new_heuristics)+1}/{self.n} (Total {attempts}/{max_attempts})")
                    file_path, code = self.pi_operator.generate(population)
                
                elif operator == 'SI':
                    print(f"\n  🎯 [{operator}] Attempt #{len(new_heuristics)+1}/{self.n} (Total {attempts}/{max_attempts})")
                    file_path, code = self.si_operator.generate(population)
                
                elif operator == 'DI':
                    print(f"\n  🎯 [{operator}] Attempt #{len(new_heuristics)+1}/{self.n} (Total {attempts}/{max_attempts})")
                    file_path, code = self.di_operator.generate(population)
                
                if file_path and standalone_smoke_test(file_path, problem=self.problem, api_key=self.api_key):
                    perf_vector, avg_perf = self.evaluate_heuristic(file_path, time_limit)
                    feasible = self._check_feasibility(perf_vector)
                    
                    if feasible:
                        print(f"  ✅ [{operator}] Success: {os.path.basename(file_path)}")
                        new_heuristics.append({
                            'name': file_path,
                            'performance_vector': perf_vector,
                            'avg_performance': avg_perf
                        })
                        self.evaluations += 1
                    else:
                        print(f"  ⚠️  [{operator}] Timeout, applying EI operator")
                        file_path, code = self.ei_operator.generate(file_path)
                        
                        if file_path and standalone_smoke_test(file_path, problem=self.problem, api_key=self.api_key):
                            perf_vector, avg_perf = self.evaluate_heuristic(file_path, time_limit)
                            feasible = self._check_feasibility(perf_vector)
                            
                            if feasible:
                                print(f"  ✅ [{operator}+EI] Success: {os.path.basename(file_path)}")
                                new_heuristics.append({
                                    'name': file_path,
                                    'performance_vector': perf_vector,
                                    'avg_performance': avg_perf
                                })
                                self.evaluations += 1
                            else:
                                print(f"  ❌ [{operator}+EI] Still timeout, abandoned")
                        else:
                            print(f"  ❌ [{operator}+EI] Smoke test failed")
                else:
                    print(f"  ❌ [{operator}] Smoke test failed")
                    
            except Exception as e:
                print(f"  ❌ [{operator}] Failed: {e}")
        
        print(f"\n{'='*80}")
        print(f"✓ Evolute Population completed: {len(new_heuristics)}/{self.n} new heuristics generated")
        print(f"{'='*80}\n")
        
        return new_heuristics
    
    def _check_feasibility(self, performance_vector: List[float]) -> bool:
        """Check if heuristic is feasible (no timeout)."""
        penalty_value = 1e10 - 1
        for score in performance_vector:
            if score > penalty_value:
                return False
        return True
    
    def evaluate_heuristic(
        self,
        heuristic_path: str,
        time_limit: float
    ) -> Tuple[List[float], float]:
        """Evaluate performance of a single heuristic."""
        results = run_hyper_heuristic(
            problem=self.problem,
            heuristic=heuristic_path,
            test_data="test_data",
            llm_config_file=self.llm_config_file,
            heuristic_dir=self.heuristic_dir,
            iterations_scale_factor=self.iterations_scale_factor,
            result_dir=self.result_dir,
            time_limit=time_limit
        )
        
        performance_vector = results
        avg_performance = np.mean(performance_vector)
        
        return performance_vector, avg_performance
    
    def complementary_population_management(
        self,
        current_pop: List[Dict],
        new_pop: List[Dict]
    ) -> List[Dict]:
        """
        Complementary population management - Exact solution using Gurobi MILP.
        
        Paper Equation (9): MILP model
        min η s.t. select n heuristics, minimize worst-case performance
        """
        pool = current_pop + new_pop
        
        if len(pool) <= self.n:
            return pool
        
        selected = complementary_selection_milp(
            pool=pool,
            n=self.n,
            time_limit=self.milp_time_limit,
            verbose=True
        )
        
        return selected
    
    def compute_cpi(self, population: List[Dict]) -> float:
        """
        Compute Complementary Performance Index (CPI).
        
        Paper Equation (2): CPI = (1/m) * sum_i min_{h in H} f_i(h)
        """
        if not population:
            return float('inf')
        
        m = len(population[0]['performance_vector'])
        total = 0.0
        
        for j in range(m):
            f_star = min(h['performance_vector'][j] for h in population)
            total += f_star
        
        return total / m
    
    def run(self, initial_results_dict: Dict, time_limit: float) -> List[Dict]:
        """
        Main evolution loop (Algorithm 1, lines 7-21).
        
        Args:
            initial_results_dict: Evaluation results of initial heuristics
            time_limit: Time limit
        
        Returns:
            population: Final evolved heuristic portfolio
        """
        print("\n" + "=" * 80)
        print("🚀 MACE Stage Two Evolution Started")
        print("=" * 80)
        print(f"Configuration: Population size={self.n}, Max evaluations={self.Nmax}")
        
        population = self.initialize_population(initial_results_dict)
        print(f"\n✓ Population initialized: {len(population)} heuristics")
        initial_cpi = self.compute_cpi(population)
        print(f"✓ Initial CPI: {initial_cpi:.4f}")
        
        iteration = 0
        while self.evaluations < self.Nmax:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"📈 Generation {iteration} (Evaluated: {self.evaluations}/{self.Nmax})")
            print(f"{'='*80}")
            
            new_heuristics = self.evolute_population(population, initial_results_dict, time_limit)
            
            if not new_heuristics:
                print("\n⚠️  No new heuristics generated, evolution terminated")
                break
            
            population = self.complementary_population_management(population, new_heuristics)
            
            current_cpi = self.compute_cpi(population)
            print(f"\n📊 Current CPI: {current_cpi:.4f}")
            print(f"📈 Improvement vs initial: {((initial_cpi - current_cpi) / initial_cpi * 100):.2f}%")
            
            if self.evaluations >= self.Nmax:
                print(f"\n✅ Reached max evaluations {self.Nmax}, evolution completed")
                break
        
        print("\n" + "=" * 80)
        print("🎉 MACE Stage Two Evolution Completed!")
        print("=" * 80)
        
        return population