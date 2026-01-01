import argparse
import os
from src.pipeline.problem_state_generator import ProblemStateGenerator
from src.util.llm_client.get_llm_client import get_llm_client

# problem can be jssp; tsp; cvrp; psp
problem = "tsp" 
smoke_test = True
llm_config_file = os.path.join("output", "llm_config", "azure_gpt_4o.json")

# Initialize configuration
prompt_dir = os.path.join("src", "problems", "base", "prompt")
output_dir = os.path.join("output", problem, "generate_problem_state")

# Create LLM client
llm_client = get_llm_client(llm_config_file, prompt_dir, output_dir)

# Generate problem state
problem_state_generator = ProblemStateGenerator(llm_client=llm_client, problem=problem)
result = problem_state_generator.generate_problem_state(smoke_test=smoke_test)

print(f"Problem state generation completed. Results saved to: {result}")