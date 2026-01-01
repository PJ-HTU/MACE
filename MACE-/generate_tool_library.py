import sys
import os

project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.tool_library_builder import ToolLibraryBuilder
from src.util.llm_client.get_llm_client import get_llm_client

# Configuration
problem = "tsp"
smoke_test = False
llm_config_file = os.path.join("output", "llm_config", "azure_gpt_4o.json")
max_try_times = 5

# Check dependencies
print("Checking required dependency files...\n")

state_file = os.path.join("output", problem, "generate_problem_state", "problem_state.py")
if os.path.exists(state_file):
    print("✅ State Space (S) generated")
else:
    print("⚠️  State Space (S) not found")
    print(f"   Recommend running: generate_problem_state.ipynb first")

action_file = os.path.join("output", problem, "generate_action_space", "action_space.py")
comp_file = os.path.join("src", "problems", problem, "components.py")
if os.path.exists(action_file) or os.path.exists(comp_file):
    print("✅ Action Space (A) generated")
else:
    print("⚠️  Action Space (A) not found")
    print(f"   Recommend running: generate_action_space.ipynb first")

print("\nNote: Tool library can be generated without S and A, but quality may be lower")

# Initialize
prompt_dir = os.path.join("src", "problems", "base", "prompt")
output_dir = os.path.join("output", problem, "generate_tool_library")
os.makedirs(output_dir, exist_ok=True)

print("Initializing LLM client...")
llm_client = get_llm_client(llm_config_file, prompt_dir, output_dir)

print("Creating Tool Library Builder...")
tool_builder = ToolLibraryBuilder(llm_client=llm_client, problem=problem)
print("✅ Initialization complete!")

# Generate tool library
print("\n" + "="*60)
print("Starting tool library generation...")
print("="*60 + "\n")

result_file = tool_builder.generate_tool_library(
    smoke_test=smoke_test,
    max_try_times=max_try_times
)

if result_file:
    print(f"\n✅ Tool library generation successful!")
    print(f"📁 File location: {result_file}")
    print(f"📄 Documentation location: {result_file.replace('.py', '_documentation.txt')}")
else:
    print(f"\n❌ Tool library generation failed")
    print(f"Please check log files in output directory: {output_dir}")