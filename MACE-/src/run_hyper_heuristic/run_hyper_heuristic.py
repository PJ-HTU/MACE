from openai import OpenAI
import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import search_file
import json

def run_hyper_heuristic(
    problem,
    heuristic,
    llm_config_file,
    heuristic_dir,
    test_data,
    iterations_scale_factor,
    result_dir,
    time_limit=None  # New: time limit (seconds)
):
    
    list_results = []
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    heuristic_name = heuristic.split(os.sep)[-1].split(".")[0]
    
    heuristic_pool_path = os.path.join("src", "problems", problem, "heuristics", heuristic_dir)
    if not os.path.exists(heuristic_pool_path):
        raise FileNotFoundError(f"Heuristic directory does not exist: {heuristic_pool_path}")
    heuristic_pool = [
        f for f in os.listdir(heuristic_pool_path) 
        if f != '.ipynb_checkpoints' 
    ]

    base_output_dir = (
        os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "output")
        if os.getenv("AMLT_OUTPUT_DIR")
        else "output"
    )

    hyper_heuristic = None
    experiment_name = ""
    llm_client = None

    experiment_name = heuristic_name
    hyper_heuristic = SingleHyperHeuristic(heuristic=heuristic_name, problem=problem)

    try:
        module = importlib.import_module(f"src.problems.{problem}.env")
        Env = getattr(module, "Env")
    except Exception as e:
        raise ImportError(f"Failed to import environment class for problem {problem}: {str(e)}")

    # Process test data: load all files if using a directory, otherwise use a single file
    if test_data == "test_data":
        test_data_dir = search_file("test_data", problem)
        test_data_list = [
            f for f in os.listdir(test_data_dir) 
            if f != ".ipynb_checkpoints"
        ]
    elif test_data == "smoke_data":
        test_data_dir = search_file("smoke_data", problem)
        test_data_list = [
            f for f in os.listdir(test_data_dir) 
            if f != ".ipynb_checkpoints"
        ]
    else:
        test_data_list = [test_data]
        
    for data_name in test_data_list:
        try:
            # Initialize environment, set output directory, and reset
            env = Env(data_name=data_name)
            output_dir = os.path.join(
                base_output_dir, problem, result_dir, env.data_ref_name, experiment_name
            )
            env.reset(output_dir)

            # Save parameters to parameters.txt
            params = {
                "problem": problem,
                "heuristic": heuristic,
                "llm_config_file": llm_config_file,
                "heuristic_dir": heuristic_dir,
                "test_data": test_data,
                "iterations_scale_factor": iterations_scale_factor,
                "result_dir": result_dir,
                "data_path": env.data_path,
                "time_limit": time_limit
            }
            with open(os.path.join(env.output_dir, "parameters.txt"), 'w') as f:
                f.write('\n'.join(f'{k}={v}' for k, v in params.items()))

            # Run heuristic with optional time limit
            validation_result = hyper_heuristic.run(env, time_limit=time_limit)

            # Save results if successful, otherwise apply penalty
            if validation_result:
                env.dump_result()
                list_results.append(env.key_value)
            else:
                penalty_value = 1e10
                list_results.append(penalty_value)

        except Exception as e:
            pass

    return list_results


def evaluate_all_heuristics(
    problem,
    heuristic_dir,
    test_data,
    llm_config_file=None,
    iterations_scale_factor=1.0,
    result_dir="result",
    save_to_file=True,
    time_limit=None  # New: time limit (seconds)
):
    heuristic_pool_path = os.path.join("src", "problems", problem, "heuristics", heuristic_dir)
    if not os.path.exists(heuristic_pool_path):
        raise FileNotFoundError(f"Heuristic directory does not exist: {heuristic_pool_path}")
    
    # Collect all heuristic files (without .py extension)
    all_heuristics = [
        f.replace('.py', '') for f in os.listdir(heuristic_pool_path) 
        if f.endswith('.py') and f != '__init__.py' and not f.startswith('.')
    ]
    print(f"Found {len(all_heuristics)} heuristic algorithms:")
    
    results_dict = {}
    
    for i, heuristic_name in enumerate(all_heuristics, 1):
        try:
            list_results = run_hyper_heuristic(
                problem=problem,
                heuristic=heuristic_name,
                llm_config_file=llm_config_file,
                heuristic_dir=heuristic_dir,
                test_data=test_data,
                iterations_scale_factor=iterations_scale_factor,
                result_dir=result_dir,
                time_limit=time_limit
            )
            results_dict[heuristic_name] = list_results
        except Exception as e:
            results_dict[heuristic_name] = None
    
    print(f"\n{'='*60}")
    print("Test data evaluation completed. Summary of results:")
    print(f"{'='*60}")
    
    return results_dict
