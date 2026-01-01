{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "97dfa748-452e-46b4-bcb8-aa3d7e585172",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chat dumped to output\\tsp\\generate_heuristic\\background.txt\n",
      "Chat dumped to output\\tsp\\generate_heuristic\\heuristic_from_llm.txt\n",
      "=== Smoke test started ===\n",
      "=== Smoke test passed ===\n",
      "Chat dumped to output\\tsp\\generate_heuristic\\nearest_neighbor_a97c.txt\n",
      "Save nearest_neighbor_a97c code to output\\tsp\\generate_heuristic\\nearest_neighbor_a97c.py\n",
      "=== Smoke test started ===\n",
      "=== Smoke test passed ===\n",
      "Chat dumped to output\\tsp\\generate_heuristic\\farthest_neighbor_0541.txt\n",
      "Save farthest_neighbor_0541 code to output\\tsp\\generate_heuristic\\farthest_neighbor_0541.py\n",
      "=== Smoke test started ===\n",
      "=== Smoke test passed ===\n",
      "Chat dumped to output\\tsp\\generate_heuristic\\cheapest_insertion_f134.txt\n",
      "Save cheapest_insertion_f134 code to output\\tsp\\generate_heuristic\\cheapest_insertion_f134.py\n",
      "=== Smoke test started ===\n",
      "=== Smoke test passed ===\n",
      "Chat dumped to output\\tsp\\generate_heuristic\\nearest_insertion_e59e.txt\n",
      "Save nearest_insertion_e59e code to output\\tsp\\generate_heuristic\\nearest_insertion_e59e.py\n",
      "=== Smoke test started ===\n",
      "=== Smoke test passed ===\n",
      "Chat dumped to output\\tsp\\generate_heuristic\\random_insertion_62f9.txt\n",
      "Save random_insertion_62f9 code to output\\tsp\\generate_heuristic\\random_insertion_62f9.py\n",
      "Heuristic generation completed. Results saved to: output\\tsp\\generate_heuristic\n"
     ]
    }
   ],
   "source": [
    "import argparse\n",
    "import os\n",
    "from src.pipeline.heuristic_generator import HeuristicGenerator\n",
    "from src.util.llm_client.get_llm_client import get_llm_client\n",
    "\n",
    "# Configuration parameters\n",
    "problem = \"tsp\"\n",
    "smoke_test = True\n",
    "llm_config_file = os.path.join(\"output\", \"llm_config\", \"azure_gpt_4o.json\")\n",
    "source = \"llm\"\n",
    "paper_path = None\n",
    "related_problems = \"all\"\n",
    "reference_data = None\n",
    "\n",
    "# Prepare problem pool\n",
    "problem_pool = [p for p in os.listdir(os.path.join(\"src\", \"problems\")) if p != \"base\"]\n",
    "if problem not in problem_pool:\n",
    "    raise ValueError(f\"Problem type '{problem}' does not exist. Available types: {problem_pool}\")\n",
    "\n",
    "# Initialize LLM client\n",
    "prompt_dir = os.path.join(\"src\", \"problems\", \"base\", \"prompt\")\n",
    "output_dir = os.path.join(\"output\", problem, \"generate_heuristic\")\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "\n",
    "llm_client = get_llm_client(llm_config_file, prompt_dir, output_dir)\n",
    "\n",
    "# Initialize heuristic generator\n",
    "heuristic_generator = HeuristicGenerator(llm_client=llm_client, problem=problem)\n",
    "\n",
    "# Generate heuristic based on source\n",
    "if source == \"llm\":\n",
    "    result = heuristic_generator.generate_from_llm(\n",
    "        reference_data=reference_data,\n",
    "        smoke_test=smoke_test\n",
    "    )\n",
    "else:\n",
    "    raise ValueError(f\"Unknown source: {source}. Available sources: ['llm', 'paper', 'related_problem']\")\n",
    "\n",
    "print(f\"Heuristic generation completed. Results saved to: {output_dir}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d9d4c914-3e7c-4792-b7d9-46968b17ac76",
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
