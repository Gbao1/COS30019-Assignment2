# COS30019 Assignment 2 - Part A (Tree-Based Search)

This project implements six tree-based search methods for the Route Finding problem:
- DFS
- BFS
- GBFS
- AS (A*)
- CUS1 (Hill Climbing)
- CUS2 (Uniform Cost Search)

The program reads a problem text file, runs one selected method, and prints output in the assignment-required format.

## Project Structure

- search.py: CLI entry point used by marker testing.
- route_search/: Core package.
- route_search/algorithms/: One file per search algorithm.
- problems/sample_problem.txt: Graph from assignment example.
- problems/test_cases/: 15 test problems for coverage.
- tests/run_all_methods.py: Quick smoke test runner.
- tests/verify_all_cases.py: Validation runner for all methods x all cases.

## Algorithms and Logic

### DFS
Depth-first traversal using a stack (LIFO). Explores one branch deeply before backtracking.

### BFS
Breadth-first traversal using a queue (FIFO). Expands level by level and tends to find shortest path in number of edges (for unweighted move count).

### GBFS
Greedy Best-First Search uses only heuristic h(n) to choose the next node (closest estimated distance to destination).

### AS (A*)
A* uses f(n) = g(n) + h(n), where:
- g(n): path cost from origin to current node (edge costs)
- h(n): heuristic estimate to nearest destination

### CUS1
CUS1 uses steepest-ascent hill climbing:
- At each step it picks the neighbor with the lowest heuristic value h(n).
- It moves only if that heuristic is strictly better than the current node.
- If there is no improving move (local minimum/plateau), it stops and reports NoGoal.

This intentionally behaves as a local-search strategy, so it may fail on graphs where a temporary heuristic increase is required to reach a destination.

### CUS2
CUS2 uses Uniform Cost Search (UCS):
- Frontier priority is cumulative path cost g(n).
- Expands the currently cheapest path first.
- Guarantees minimum-cost path to a destination when edge costs are non-negative.

## Tie-Breaking Rules

Implemented to align with assignment notes:
- When equal priority, smaller node id is expanded first.
- If still equal, earlier-created node is expanded first.

## Input File Format

The solver expects this format:
- Nodes:
  - node_id: (x,y)
- Edges:
  - (src,dst): cost
- Origin:
  - start_node
- Destinations:
  - nodeA; nodeB; ...

Both inline and next-line forms are supported, for example:
- Origin: 2
- or:
  - Origin:
  - 2

## How To Run

### Run one algorithm on one case

python search.py problems/test_cases/tc01_simple_chain.txt DFS

### Run all six algorithms on the sample case

python tests/run_all_methods.py

### Run full verification across all test cases

python tests/verify_all_cases.py

This executes 6 methods x 15 cases = 90 runs and checks:
- no runtime failures
- each problem file is structurally valid (origin/destination/node references and edge definitions)
- valid output shape
- returned goal is a destination
- path starts at origin
- every path transition matches a directed edge
- edge costs are non-negative (required for UCS optimality)
- NoGoal appears only when destination is unreachable

## Test Case Coverage (15 Cases)

1. tc01_simple_chain.txt: Simple reachable chain.
2. tc02_branching_weighted.txt: Branching graph with weighted alternatives.
3. tc03_unreachable_goal.txt: Destination is unreachable.
4. tc04_multi_destination.txt: Multiple destination nodes.
5. tc05_cycle_graph.txt: Cycle handling.
6. tc06_tie_break_equal_priority.txt: Tie-breaking behavior.
7. tc07_directed_trap.txt: Directed loop trap with alternate route.
8. tc08_dense_graph.txt: Dense directed graph.
9. tc09_origin_is_goal.txt: Origin is already a destination.
10. tc10_disconnected_multi_goal.txt: Mixed reachable/unreachable destinations.
11. tc11_cost_vs_steps.txt: Difference between weighted-cost and move-count behaviors.
12. tc12_decimal_costs.txt: Decimal edge costs.
13. tc13_larger_8_nodes.txt: Larger graph with more branching.
14. tc14_bidirectional_mix.txt: Mixed one-way and bidirectional links.
15. tc15_no_edges.txt: No traversable edges.

## Output Format

Program output format:
- line 1: filename method
- line 2: goal number_of_nodes
- line 3: path sequence

If no destination is found:
- line 2: NoGoal number_of_nodes
- line 3: empty

## Notes

- Built for Windows CLI testing as required.
- Code is modular to support easy maintenance and report explanation.

## Assignment 2B (TBRGS) Add-on

This repository now includes an Assignment 2B implementation scaffold under the `tbrgs` package.

### What is included

- Data processing pipeline for SCATS traffic data (long and wide formats).
- Three ML algorithms for traffic flow prediction:
  - LSTM
  - GRU
  - Random Forest (third model)
- Comprehensive model evaluation (MAE, RMSE, MAPE, inference time).
- Travel-time conversion using the assignment flow-speed equation.
- Integration with Part A via `CUS2` (UCS) for shortest path and top-k route generation.
- End-to-end CLI and GUI entry points.
- A2B test suite (`tests_a2b`) with 15+ tests.

### New files

- `config/tbrgs_defaults.json`: default configuration.
- `tbrgs/`: A2B package modules.
- `run_tbrgs_cli.py`: command-line execution for training/evaluation/routing.
- `run_tbrgs_gui.py`: GUI for origin/destination/model/top-k inputs.
- `tests_a2b/`: A2B unit tests.

### Installation (A2B)

Install required packages in your virtual environment:

```bash
pip install -r requirements_a2b.txt
```

For this workspace, use a Python 3.10-3.12 virtual environment so TensorFlow can run true LSTM/GRU training.

Example on Windows:

```bash
py -3.10 -m venv .venv310
.\.venv310\Scripts\python.exe -m pip install -r requirements_a2b.txt
```

### Run A2B from CLI

```bash
python run_tbrgs_cli.py --config config/tbrgs_defaults.json --origin 2000 --destination 3002 --top-k 5 --model best
```

### Run A2B GUI

```bash
python run_tbrgs_gui.py
```

### Run A2B tests

```bash
pytest tests_a2b -q
```

### Rubric coverage mapping

1. Data processing (9 marks):
  - `tbrgs/data_processing.py`
2. Three ML algorithms incl. LSTM and GRU (7x3 marks):
  - `tbrgs/modeling.py`
3. Comprehensive evaluation (15 marks):
  - `tbrgs/evaluation.py`
4. Part A + Part B integration (15 marks):
  - `tbrgs/topk_routing.py` uses `route_search.algorithms.cus2.solve_ucs`

