# MACE: Modular Algorithm Construction and Evolution

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](YOUR_ARXIV_LINK)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

MACE (Modular Algorithm Construction and Evolution) is a novel framework that leverages Large Language Models (LLMs) to autonomously discover heuristic algorithms for combinatorial optimization (CO) problems. The framework addresses two critical challenges: **generalization across diverse problem structures** and **adaptation to varying runtime constraints**.
**![MACE Framework Architecture](./MACE%20Framework%20Architecture.png)**

## Key Features

✅ **Universal Generalization**: Works across structurally diverse CO problems (JSSP, TSP, CVRP, PSP) through modular S-A-T-H decomposition  
✅ **Time-Aware Evolution**: Explicitly optimizes for quality-time trade-offs through 5 evolution operators  
✅ **Complementary Portfolios**: Generates algorithm portfolios that provide robust coverage across heterogeneous instances  
✅ **Zero-Shot Capability**: Successfully handles novel problems (PSP) absent from LLM training data  
✅ **Practical Deployment**: Produces algorithms ready for emergency, routine, and strategic planning scenarios

---

## Table of Contents

- [Framework Overview](#framework-overview)
- [Usage Guide](#usage-guide)
- [Code Organization](#code-organization)
- [Supported Problems](#supported-problems)
- [Adding New Problems](#adding-new-problems)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Human-in-the-Loop Design Patterns](#human-in-the-loop-design-patterns)
- [Citation](#citation)

---

## Framework Overview

MACE operates through a **two-stage architecture**:

### Stage One: Universal Modular Decomposition

This stage decomposes algorithm design into four fundamental components, enabling LLMs to generate heuristics through structured reasoning:

**1. State Designer (S)**  
Defines the state space that captures both static problem characteristics and dynamic solution progress. The state representation provides LLMs with essential context about the problem instance and current solution status.

**2. Action Designer (A)**  
Specifies the action space comprising constructive actions (for building solutions incrementally) and improvement actions (for refining existing solutions). These actions serve as the building blocks for heuristic algorithms.

**3. Tool Library Builder (T)**  
Encapsulates complex domain logic as callable modules. Tools handle domain-specific calculations, algorithms, and optimization utilities that are difficult for LLMs to implement correctly from scratch. This enables the injection of expert knowledge beyond the LLM's training data.

**4. Policy Generator (H)**  
Generates an initial portfolio of diverse heuristics that map states to actions. Each heuristic implements a decision-making strategy by selecting appropriate actions based on the current problem state and can invoke tools from the library.

This **S-A-T-H paradigm** separates high-level algorithmic strategy from low-level implementation details, allowing LLMs to focus on decision-making logic while leveraging domain knowledge through explicit interfaces.

### Stage Two: Time-Constrained Complementary Evolution

This stage iteratively refines the heuristic portfolio through evolution operators and complementary selection:

**Evolution Operators** (5 types):

- **CI (Complementary Improvement)**: Identifies heuristic pairs with complementary strengths across instances and merges their effective strategies into a new heuristic
- **PI (Performance Improvement)**: Selects heuristics with high ranking variance and refines them to reduce performance instability  
- **SI (Specialization Improvement)**: Generates specialist heuristics targeting under-served instance types with low algorithmic differentiation
- **DI (Diversity Improvement)**: Analyzes existing strategies and generates heuristics following fundamentally different algorithmic principles
- **EI (Efficiency Improvement)**: Reduces computational complexity when heuristics exceed time budgets, optimizing for efficiency while preserving core logic

**Complementary Selection**:  
Uses a minimax MILP formulation to select the optimal subset of heuristics from candidate pools. The objective minimizes worst-case performance across all instances, ensuring robust coverage rather than just optimizing average quality.

**Multi-Budget Execution**:  
Stage Two runs independently under different time budgets (Tₘₐₓ), producing portfolios spanning the quality-time spectrum. This enables practitioners to select appropriate algorithms based on operational urgency—fast portfolios for emergency decisions, balanced portfolios for routine operations, or high-quality portfolios for strategic planning.

---

## Usage Guide

### Stage One: Generating Problem Components

Each component generation follows a similar pattern—initialize the builder with an LLM client and problem type, then call the generation method. Results are automatically saved to the `output/{problem}/` directory.

**State Space Generation:**  
The State Designer analyzes the problem and generates functions to extract instance features (static characteristics like problem size, data statistics) and solution features (dynamic metrics like completion ratio, current objective value). These provide the context LLMs need for decision-making.

**Action Space Generation:**  
The Action Designer identifies feasible operations for constructing and improving solutions. Constructive actions build solutions step-by-step, while improvement actions perform local modifications to refine quality.

**Tool Library Generation:**  
The Tool Library Builder identifies complex domain logic that should be encapsulated as reusable functions. This includes calculations requiring expert knowledge, difficult-to-implement algorithms, and computationally intensive procedures. Tools reduce code generation errors and enable LLMs to leverage domain expertise.

**Initial Heuristic Generation:**  
The Policy Generator creates diverse heuristics using the generated state space, action space, and tool library. Each heuristic implements a decision-making strategy mapping states to actions.

### Stage Two: Evolving Heuristic Portfolios

The evolution process iteratively generates new heuristics through the five operators, validates them through smoke tests, evaluates performance on validation instances, and selects optimal subsets via complementary selection. Evolution continues until convergence or reaching the maximum iteration limit.

**Key Parameters:**
- `population_size` (n): Number of heuristics in the portfolio (default: 10)
- `max_evaluations` (Nₘₐₓ): Maximum evolution iterations (default: 100)  
- `time_limit` (Tₘₐₓ): Time budget per instance in seconds
- `milp_time_limit`: Solver time limit for complementary selection

**Multi-Budget Execution:**  
Run evolution independently with different time limits (e.g., 60s, 300s, 600s) to generate portfolios for different operational scenarios. Each portfolio optimizes for its specific time constraint.

### Evaluation

The evaluation module runs heuristics on test instances and computes solution quality metrics. Results include per-instance performance and aggregate statistics. For problems with known optimal solutions, quality is measured as percentage gap from optimum.

---

## Code Organization

### Complete Mapping Table

The table below maps each component from the paper to its implementation in the codebase:

| Paper Component | Code Location | Key Class/Function | Description |
|----------------|---------------|-------------------|-------------|
| **Stage One** | | | |
| State Designer | `src/pipeline/state_designer.py` | `StateDesigner` | Generates state extraction functions |
| Action Designer | `src/pipeline/action_designer.py` | `ActionDesigner` | Generates action space definitions |
| Tool Library Builder | `src/pipeline/tool_library_builder.py` | `ToolLibraryBuilder` | Generates domain-specific helper functions |
| Policy Generator | `src/pipeline/policy_generator.py` | `PolicyGenerator` | Generates initial heuristic portfolio |
| **Stage Two** | | | |
| MACE Evolver | `src/run_hyper_heuristic/MACEEvolver.py` | `MACEEvolver` | Main evolution loop (Algorithm 1) |
| CI Operator | `src/run_hyper_heuristic/ci_operator.py` | `CIOperator` | Complementary Improvement |
| PI Operator | `src/run_hyper_heuristic/pi_operator.py` | `PIOperator` | Performance Improvement |
| SI Operator | `src/run_hyper_heuristic/si_operator.py` | `SIOperator` | Specialization Improvement |
| DI Operator | `src/run_hyper_heuristic/di_operator.py` | `DIOperator` | Diversity Improvement |
| EI Operator | `src/run_hyper_heuristic/ei_operator.py` | `EIOperator` | Efficiency Improvement |
| Complementary Selection | `src/run_hyper_heuristic/complementary_selection_simple.py` | `complementary_selection_milp` | MILP-based portfolio selection |
| **Auxiliary** | | | |
| Smoke Test | `src/run_hyper_heuristic/smoke_test.py` | `standalone_smoke_test` | Code validation and repair |
| Evaluation | `src/run_hyper_heuristic/run_hyper_heuristic.py` | `evaluate_all_heuristics` | Performance assessment |

### Project Structure

```
MACE/
├── src/
│   ├── problems/                    # Problem-specific implementations
│   │   ├── base/                   # Base classes and templates
│   │   │   ├── env.py             # Base environment class
│   │   │   ├── prompt/            # Prompt templates
│   │   │   └── ...
│   │   ├── jssp/                  # Job Shop Scheduling Problem
│   │   ├── tsp/                   # Traveling Salesman Problem
│   │   ├── cvrp/                  # Vehicle Routing Problem
│   │   └── psp/                   # Port Scheduling Problem
│   │
│   ├── pipeline/                   # Stage One: Modular Decomposition
│   │   ├── state_designer.py      # State space generation
│   │   ├── action_designer.py     # Action space generation
│   │   ├── tool_library_builder.py # Tool library generation
│   │   └── policy_generator.py    # Initial heuristic generation
│   │
│   ├── run_hyper_heuristic/       # Stage Two: Evolution
│   │   ├── MACEEvolver.py         # Main evolution controller
│   │   ├── ci_operator.py         # CI operator implementation
│   │   ├── pi_operator.py         # PI operator implementation
│   │   ├── si_operator.py         # SI operator implementation
│   │   ├── di_operator.py         # DI operator implementation
│   │   ├── ei_operator.py         # EI operator implementation
│   │   ├── complementary_selection_simple.py  # MILP selection
│   │   ├── smoke_test.py          # Code validation
│   │   └── run_hyper_heuristic.py # Evaluation utilities
│   │
│   └── util/                       # Utility modules
│       ├── llm_client/            # LLM API wrappers
│       └── ...
│
├── output/                         # Generated outputs
│   ├── llm_config/                # LLM configuration files
│   └── {problem}/
│       ├── generate_problem_state/
│       ├── generate_action_space/
│       ├── generate_tool_library/
│       ├── generate_heuristic/
│       └── evolved_heuristics/
│
├── data/                           # Benchmark datasets
│   ├── jssp/
│   ├── tsp/
│   ├── cvrp/
│   └── psp/
│
├── scripts/                        # Utility scripts
├── *.ipynb                         # Interactive notebooks
├── requirements.txt
└── README.md
```

### Problem-Specific Components

Each problem requires the following implementations:

**components.py**: Defines the `Solution` class (representing problem solutions) and `Operator` class (defining operations to modify solutions).

**env.py**: Implements the `Env` class extending `BaseEnv`, handling data loading, solution validation, objective calculation, and result recording.

**problem_state.py**: Defines state extraction functions (`get_instance_features()` and `get_solution_features()`) for representing problem and solution characteristics.

**task_description.txt**: Text-based problem description used by LLMs to understand the problem domain.

**prompt/**: Directory containing prompt templates for LLM interaction:
- `problem_description.txt` - Detailed problem formulation and domain knowledge
- `problem_state_description.txt` - Guidelines for state representation design
- `special_remind.txt` - Important constraints and implementation notes

**data/**: Training, smoke_test, and test instances organized by scale (small, medium, large).

---

## Supported Problems

### 1. Job Shop Scheduling Problem (JSSP)

Minimize makespan when scheduling n jobs on m machines with precedence constraints. Each job consists of operations that must be processed in sequence, and each operation requires a specific machine.

**Dataset**: OR-Library instances (la01-la40, swv01-swv20, ta01-ta80)

### 2. Traveling Salesman Problem (TSP)

Find the shortest tour visiting all cities exactly once and returning to the origin. The objective is to minimize total tour length given city coordinates or distance matrix.

**Dataset**: TSPLIB instances (kroA100, eil51, berlin52, pr152, and others)

### 3. Capacitated Vehicle Routing Problem (CVRP)

Route vehicles from a central depot to serve customer demands while respecting vehicle capacity constraints. The objective is to minimize total routing distance.

**Dataset**: CVRPLIB instances (X-n series, Golden instances)

### 4. Port Scheduling Problem (PSP)

Schedule vessels to berths minimizing total completion time, considering arrival times, processing times, and berth availability. This problem demonstrates MACE's zero-shot generalization—it does not appear in LLM training data, yet the framework successfully generates feasible solutions through tool-based domain knowledge injection.

**Dataset**: Generated instances (100-500 vessels, 15-40 berths, 24-48 time periods)

---

## Adding New Problems

To extend MACE to a new combinatorial optimization problem:

### Step 1: Create Problem Directory

Set up the directory structure under `src/problems/your_problem/`:
- `components.py` - Define Solution and Operator classes
- `env.py` - Implement environment for data loading and evaluation  
- `task_description.txt` - Provide problem description for LLMs
- `prompt/` - Create prompt templates (problem_description.txt, problem_state_description.txt, special_remind.txt)
- `data/` - Organize training, smoke_test, and test instances

### Step 2: Implement Core Classes

The `Solution` class should represent your problem's solution format. The `Operator` class defines operations to construct or modify solutions. The `Env` class handles instance loading, solution validation, and objective value calculation.

### Step 3: Run MACE Pipeline

Generate components using Stage One, then evolve heuristics using Stage Two. The framework automatically handles prompt construction, LLM interaction, code generation, and validation based on your problem description and implementations.

### Step 4: Evaluate and Deploy

Test generated heuristics on validation instances during evolution, then evaluate final portfolios on held-out test sets. Deploy appropriate portfolios based on your application's time constraints.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MACE.git
cd MACE
```

### 2. Set Up Environment

We recommend using Conda for environment management:

```bash
conda create -n mace python=3.8
conda activate mace
pip install -r requirements.txt
```

**Key Dependencies:**
- NumPy, Pandas (data processing)
- Gurobi (MILP solver for complementary selection)
- OpenAI/Anthropic (LLM APIs)
- LangChain (prompt management)
- NetworkX, TSPLIB95 (problem utilities)

### 3. Configure LLM

Create an LLM configuration file in `output/llm_config/`. The framework supports multiple backends:

**For Claude Sonnet 4.5**:
```json
{
    "type": "anthropic",
    "model": "claude-sonnet-4.5",
    "api_key": "YOUR_API_KEY",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 8192
}
```

### 4. Prepare Data

The datasets are the `data/` directory with standard benchmarks. manually download from:

| Problem | Source |
|---------|--------|
| JSSP    | [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html) |
| TSP     | [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) |
| CVRP    | [CVRPLIB](http://vrp.atd-lab.inf.puc-rio.br/index.php/en/) |
| PSP     | Generated (included) |

---

## Quick Start

### Using Jupyter Notebooks (Recommended)

The framework provides interactive notebooks for each stage:

**Stage One - Component Generation:**
- `generate_problem_state.ipynb` - Generate state space
- `generate_action_space.ipynb` - Generate action space  
- `generate_tool_library.ipynb` - Generate tool library
- `generate_heuristic.ipynb` - Generate initial heuristics

**Stage Two - Evolution:**
- `evolve_heuristic.ipynb` - Evolve heuristic portfolios

Simply configure the problem type and LLM settings in each notebook, then run cells sequentially.

### Command-Line Interface

For batch processing or automation:

```bash
# Stage One: Generate all components for TSP
python scripts/run_stage_one.py --problem tsp --llm_config output/llm_config/claude.json

# Stage Two: Evolve heuristics with 300s time budget
python scripts/run_stage_two.py --problem tsp --time_limit 300 --population_size 10

# Evaluate on test set
python scripts/evaluate.py --problem tsp --test_data data/tsp/test_data
```

---

## Human-in-the-Loop Design Patterns

MACE supports two complementary approaches for designing problem components, allowing practitioners to leverage domain expertise while benefiting from LLM automation:

### Approach 1: Manual Design with Domain Knowledge

For practitioners with deep domain expertise, components can be manually designed and directly implemented in the codebase:

**State Space (`problem_state.py`)**
- Manually define `get_instance_features()` and `get_solution_features()` functions
- Encode domain-specific insights about problem characteristics
- Example: For JSSP, experts might include machine load balance, critical path metrics, etc.

**Action Space (`components.py`)**
- Manually implement the `Operator` class with domain-informed operations
- Design constructive and improvement actions based on proven heuristics
- Example: For TSP, implement 2-opt, 3-opt, or nearest neighbor insertion operators

**Tool Library (`env.py`)**
- Manually code complex domain algorithms as helper methods in the `Env` class
- Encapsulate expert knowledge that would be difficult for LLMs to derive
- Example: For CVRP, implement route feasibility checks, savings calculations, or clustering algorithms

**Benefits:**
- ✅ Full control over component design
- ✅ Direct incorporation of domain expertise
- ✅ Guaranteed correctness of critical operations
- ✅ Faster iteration for experts familiar with the problem

### Approach 2: LLM-Automated Generation

For rapid prototyping or exploring novel problems, MACE can automatically generate all components using LLMs:

**Interactive Notebooks:**
- `generate_problem_state.ipynb` → Generates state extraction functions
- `generate_action_space.ipynb` → Generates action/operator definitions  
- `generate_tool_library.ipynb` → Generates domain-specific helper tools
- `generate_heuristic.ipynb` → Generates initial heuristic portfolio

**Benefits:**
- ✅ Rapid bootstrapping for new problems
- ✅ Exploration of non-obvious design choices
- ✅ Minimal manual coding required
- ✅ Effective for problems with limited expert knowledge

### Hybrid Approach (Recommended)

In practice, the most effective strategy combines both approaches:

1. **Start with LLM generation** to quickly explore the design space
2. **Review and refine** generated components using domain knowledge
3. **Manually enhance** critical tools or actions based on expert insights
4. **Iterate** between automated generation and manual refinement

**Example Workflow:**
```python
# 1. Generate initial state space with LLM
run: generate_problem_state.ipynb

# 2. Review output/your_problem/generate_problem_state/problem_state.py
# 3. Manually refine feature extraction logic based on domain knowledge
# 4. Move refined version to src/problems/your_problem/problem_state.py

# 5. Repeat for actions and tools
run: generate_action_space.ipynb
run: generate_tool_library.ipynb

# 6. Manually enhance critical components
edit: src/problems/your_problem/components.py  # Add expert-designed operators
edit: src/problems/your_problem/env.py         # Add domain-specific algorithms
```

### Design Decision Guidelines

| Scenario | Recommended Approach |
|----------|---------------------|
| **Well-studied problem** (JSSP, TSP) | Manual design or hybrid |
| **Novel problem** (PSP) | LLM generation or hybrid |
| **Time-critical deployment** | Manual design for critical paths |
| **Exploratory research** | LLM generation |
| **Domain expert available** | Hybrid (LLM + expert refinement) |
| **Limited domain knowledge** | LLM generation |

### File Locations Summary

| Component | Manual Implementation | LLM-Generated Output |
|-----------|----------------------|---------------------|
| **State Space** | `src/problems/{problem}/problem_state.py` | `output/{problem}/generate_problem_state/problem_state.py` |
| **Action Space** | `src/problems/{problem}/components.py` | `output/{problem}/generate_action_space/components.py` |
| **Tool Library** | `src/problems/{problem}/env.py` | `output/{problem}/generate_tool_library/env.py` |

**Note:** Generated files must be manually reviewed and moved to the `src/problems/{problem}/` directory before running Stage Two evolution.

---

## Citation

If you use MACE in your research, please cite:

```bibtex
@article{mace2025,
  title={Large Language Models Discover Complementary Heuristics for Combinatorial Optimization},
  author={[Your Names]},
  journal={[Journal/Conference]},
  year={2025},
  url={[Paper URL]}
}
```
