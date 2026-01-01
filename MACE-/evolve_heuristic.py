"""
MACE Stage Two Usage Example - Integration with CI Operator
This example demonstrates how to use the new MACEEvolver class and CI operator.
"""
import numpy as np
from typing import Dict, List
from src.run_hyper_heuristic.run_hyper_heuristic import evaluate_all_heuristics
from src.run_hyper_heuristic.MACEEvolver import MACEEvolver


def main():
    """Main function."""
    
    # Configuration
    problem = "cvrp"
    api_key = "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"
    model = "x-ai/grok-code-fast-1"
    
    heuristic_dir = "basic_heuristics"
    task_description_file = f"src/problems/{problem}/task_description.txt"
    output_dir = f"src/problems/{problem}/heuristics/basic_heuristics"
    test_data = "test_data"
    result_dir = "result"
    
    population_size = 10
    max_evaluations = 3
    time_limit = 10 * 60
    iterations_scale_factor = 1.0
    
    # Step 1: Evaluate initial heuristics (output from Stage One)
    print("=" * 80)
    print("📋 Step 1: Evaluating Initial Heuristics")
    print("=" * 80)
    
    initial_results_dict = evaluate_all_heuristics(
        problem=problem,
        heuristic_dir=heuristic_dir,
        test_data=test_data,
        iterations_scale_factor=iterations_scale_factor,
        result_dir=result_dir,
        save_to_file=True,
        time_limit=time_limit
    )
    
    print("\nInitial evaluation results:")
    for h_name, scores in initial_results_dict.items():
        avg = np.mean(scores)
        print(f"  {h_name}: avg={avg:.2f}, scores={scores}")
    
    # Step 2: Run MACE Stage Two
    print("\n" + "=" * 80)
    print("📋 Step 2: Running MACE Stage Two")
    print("=" * 80)
    
    mace = MACEEvolver(
        problem=problem,
        api_key=api_key,
        heuristic_dir=heuristic_dir,
        task_description_file=task_description_file,
        output_dir=output_dir,
        model=model,
        population_size=population_size,
        max_evaluations=max_evaluations,
        time_limit=time_limit
    )
    
    final_population = mace.run(initial_results_dict, time_limit)
    
    # Step 3: Output final results
    print("\n" + "=" * 80)
    print("📊 Final Results")
    print("=" * 80)
    
    print(f"🎯 Final heuristic set size: {len(final_population)}")
    print(f"🏆 Final CPI: {mace.compute_cpi(final_population):.4f}")
    print(f"📈 Total evaluations: {mace.evaluations}")
    
    print("\n📋 Final heuristic set:")
    for i, h in enumerate(final_population, 1):
        print(f"  {i}. {h['name']}")
        print(f"     avg={h['avg_performance']:.2f}")
        print(f"     scores={h['performance_vector']}")


if __name__ == "__main__":
    main()