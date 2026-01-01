{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9bd59af0-c76c-4382-9fa4-62c8f80a15f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "problem_state.txt not properly initialized, cannot check file existence\n",
      "Chat dumped to output\\tsp\\generate_problem_state\\background.txt\n",
      "Chat dumped to output\\tsp\\generate_problem_state\\instance_problem_state.txt\n",
      "Chat dumped to output\\tsp\\generate_problem_state\\solution_problem_state.txt\n",
      "Chat dumped to output\\tsp\\generate_problem_state\\observation_problem_state.txt\n",
      "Temporary save of problem_state.py for environment initialization\n",
      "=== Smoke test started ===\n",
      "Save problem state in output\\tsp\\generate_problem_state\\problem_state.py\n",
      "Problem state generation completed. Results saved to: output\\tsp\\generate_problem_state\\problem_state.py\n"
     ]
    }
   ],
   "source": [
    "import argparse\n",
    "import os\n",
    "from src.pipeline.problem_state_generator import ProblemStateGenerator\n",
    "from src.util.llm_client.get_llm_client import get_llm_client\n",
    "\n",
    "# problem can be jssp; tsp; cvrp; psp\n",
    "problem = \"tsp\" \n",
    "smoke_test = True\n",
    "llm_config_file = os.path.join(\"output\", \"llm_config\", \"azure_gpt_4o.json\")\n",
    "\n",
    "# Initialize configuration\n",
    "prompt_dir = os.path.join(\"src\", \"problems\", \"base\", \"prompt\")\n",
    "output_dir = os.path.join(\"output\", problem, \"generate_problem_state\")\n",
    "\n",
    "# Create LLM client\n",
    "llm_client = get_llm_client(llm_config_file, prompt_dir, output_dir)\n",
    "\n",
    "# Generate problem state\n",
    "problem_state_generator = ProblemStateGenerator(llm_client=llm_client, problem=problem)\n",
    "result = problem_state_generator.generate_problem_state(smoke_test=smoke_test)\n",
    "\n",
    "print(f\"Problem state generation completed. Results saved to: {result}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93a8978f-26e5-44f4-8ea6-48bb5de85a72",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
