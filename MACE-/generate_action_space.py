import sys
import os

project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.action_designer import ActionDesigner
from src.util.llm_client.get_llm_client import get_llm_client

# Configuration
problem = "tsp"
smoke_test = True
llm_config_file = os.path.join("output", "llm_config", "azure_gpt_4o.json")
max_try_times = 5

# Initialize directories
prompt_dir = os.path.join("src", "problems", "base", "prompt")
output_dir = os.path.join("output", problem, "generate_action_space")
os.makedirs(output_dir, exist_ok=True)

# Create LLM client
print("Initializing LLM client...")
llm_client = get_llm_client(llm_config_file, prompt_dir, output_dir)

# Create Action Designer
print("Creating Action Designer...")
action_designer = ActionDesigner(llm_client=llm_client, problem=problem)
print("✅ Initialization complete!")

# Generate action space
print("\n" + "="*60)
print("Starting action space generation...")
print("="*60 + "\n")

result_file = action_designer.generate_action_space(
    smoke_test=smoke_test,
    max_try_times=max_try_times
)

if result_file:
    print(f"\n✅ Action space generation successful!")
    print(f"📁 File location: {result_file}")
else:
    print(f"\n❌ Action space generation failed")
    print(f"Please check log files in output directory: {output_dir}")