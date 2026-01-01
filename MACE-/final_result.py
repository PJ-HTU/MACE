import os
import importlib
from datetime import datetime
import pandas as pd
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.util import search_file


problem = "psp"
heuristic_dir = "basic_heuristics"
test_data = "test_data"
result_dir = "result"


heuristic_pool_path = os.path.join("src", "problems", problem, "heuristics", heuristic_dir)
if not os.path.exists(heuristic_pool_path):
    raise FileNotFoundError(f"Heuristic directory does not exist: {heuristic_pool_path}")

all_heuristics = [
    f.replace(".py", "") for f in os.listdir(heuristic_pool_path)
    if f.endswith(".py") and f != "__init__.py" and not f.startswith(".")
]

print(f"Found {len(all_heuristics)} heuristic algorithms: {all_heuristics}")


test_data_dir = search_file("test_data", problem)
test_data_list = [
    f for f in os.listdir(test_data_dir)
    if f != ".ipynb_checkpoints" and not f.startswith(".")
]

print(f"Found {len(test_data_list)} test instances: {test_data_list}")


try:
    module = importlib.import_module(f"src.problems.{problem}.env")
    Env = getattr(module, "Env")
except Exception as e:
    raise ImportError(f"Failed to import environment class for problem {problem}: {str(e)}")


base_output_dir = (
    os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "output")
    if os.getenv("AMLT_OUTPUT_DIR")
    else "output"
)
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

results_dict = {
    "algorithm": []
}

for data_name in test_data_list:
    data_ref = data_name.split(".")[0]
    results_dict[f"{data_ref}_objective"] = []
    results_dict[f"{data_ref}_runtime"] = []


print("\nStarting batch evaluation...")
print("=" * 80)

for heuristic_name in all_heuristics:
    print(f"\nTesting heuristic: {heuristic_name}")
    print("-" * 80)

    results_dict["algorithm"].append(heuristic_name)

    hyper_heuristic = SingleHyperHeuristic(
        heuristic=heuristic_name,
        problem=problem
    )

    for data_name in test_data_list:
        data_ref = data_name.split(".")[0]

        try:
            env = Env(data_name=data_name)
            experiment_name = f"{heuristic_name}.{datetime_str}"
            output_dir = os.path.join(
                base_output_dir,
                problem,
                result_dir,
                env.data_ref_name,
                experiment_name
            )
            env.reset(output_dir)

            start_time = datetime.now()
            validation_result = hyper_heuristic.run(env)
            elapsed_time = (datetime.now() - start_time).total_seconds()

            if validation_result:
                env.dump_result()
                objective_value = env.key_value
                results_dict[f"{data_ref}_objective"].append(objective_value)
                results_dict[f"{data_ref}_runtime"].append(f"{elapsed_time:.2f}s")
                print(f"  ✓ {data_name}: objective={objective_value}, time={elapsed_time:.2f}s")
            else:
                results_dict[f"{data_ref}_objective"].append("invalid")
                results_dict[f"{data_ref}_runtime"].append("-")
                print(f"  ✗ {data_name}: invalid solution")

        except Exception as e:
            results_dict[f"{data_ref}_objective"].append("error")
            results_dict[f"{data_ref}_runtime"].append("-")
            print(f"  ✗ {data_name}: error - {str(e)}")


df_results = pd.DataFrame(results_dict)
df_results.set_index("algorithm", inplace=True)

excel_output_path = os.path.join(
    base_output_dir,
    problem,
    result_dir,
    f"batch_test_results_{datetime_str}.xlsx"
)
os.makedirs(os.path.dirname(excel_output_path), exist_ok=True)
df_results.to_excel(excel_output_path, engine="openpyxl")

print("\n" + "=" * 80)
print(f"Evaluation completed. Results saved to: {excel_output_path}")
print("=" * 80)

print("\nResult preview:")
print(df_results)

df_results
