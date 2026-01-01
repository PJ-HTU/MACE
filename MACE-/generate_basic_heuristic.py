import argparse
import os
from src.pipeline.heuristic_generator import HeuristicGenerator
from src.util.llm_client.get_llm_client import get_llm_client

# Configuration parameters
problem = "tsp"
smoke_test = True
llm_config_file = os.path.join("output", "llm_config", "azure_gpt_4o.json")
source = "llm"
paper_path = None
related_problems = "all"
reference_data = None

# Prepare problem pool
problem_pool = [p for p in os.listdir(os.path.join("src", "problems")) if p != "base"]
if problem not in problem_pool:
    raise ValueError(f"Problem type '{problem}' does not exist. Available types: {problem_pool}")

# Initialize LLM client
prompt_dir = os.path.join("src", "problems", "base", "prompt")
output_dir = os.path.join("output", problem, "generate_heuristic")
os.makedirs(output_dir, exist_ok=True)

llm_client = get_llm_client(llm_config_file, prompt_dir, output_dir)

# Initialize heuristic generator
heuristic_generator = HeuristicGenerator(llm_client=llm_client, problem=problem)

# Generate heuristic based on source
if source == "llm":
    result = heuristic_generator.generate_from_llm(
        reference_data=reference_data,
        smoke_test=smoke_test
    )
else:
    raise ValueError(f"Unknown source: {source}. Available sources: ['llm', 'paper', 'related_problem']")

print(f"Heuristic generation completed. Results saved to: {output_dir}")