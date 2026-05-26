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

- 2A/search.py: CLI entry point used by marker testing.
- 2A/route_search/: Core package.
- 2A/route_search/algorithms/: One file per search algorithm.
- 2A/problems/sample_problem.txt: Graph from assignment example.
- 2A/problems/test_cases/: 15 test problems for coverage.
- 2A/tests/run_all_methods.py: Quick smoke test runner.
- 2A/tests/verify_all_cases.py: Validation runner for all methods x all cases.

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

python 2A/search.py 2A/problems/test_cases/tc01_simple_chain.txt DFS

### Run all six algorithms on the sample case

python 2A/tests/run_all_methods.py

### Run full verification across all test cases

python 2A/tests/verify_all_cases.py

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

This repository now includes an Assignment 2B implementation scaffold under the `2B/tbrgs` package.

### What is included

- Data processing pipeline for SCATS traffic data (long and wide formats).
- Three ML algorithms for traffic flow prediction:
  - LSTM
  - GRU
  - Random Forest (third model)
- Separate training and metric comparison across models (MAE, RMSE, MAPE, inference time).
- Direction-aware, hour-based edge flow estimation for routing.
- Comprehensive model evaluation (MAE, RMSE, MAPE, inference time).
- Travel-time conversion using the assignment flow-speed equation.
- Integration with Part A via `CUS2` (UCS) for shortest path and top-k route generation.
- End-to-end CLI and GUI entry points with interactive map node picking.
- GUI visualization of the road graph with highlighted top-k recommended routes.
- A2B test suite (`tests_a2b`) with 15+ tests.

### New files

- `2B/config/tbrgs_defaults.json`: default configuration.
- `2B/data/input/`: input data files used by TBRGS.
- `2B/data/output/`: generated metrics/results outputs.
- `2B/tbrgs/`: A2B package modules.
- `2B/run_tbrgs_cli.py`: command-line execution for training/evaluation/routing.
- `2B/run_tbrgs_gui.py`: GUI for origin/destination/model/top-k inputs.
- `2B/tests_a2b/`: A2B unit tests.

### Installation (A2B)

For this workspace, use Python 3.10-3.12 so TensorFlow can run LSTM/GRU training.

Brand-new setup (no existing `.venv` required):

```bash
Set-Location "C:\path\to\13_Intro_AI"
py -3.10 -m venv .venv310
```

If script execution is blocked when activating venv:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Option A (activate then use `python`):

```bash
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r 2B/requirements_a2b.txt
```

Option B (do not activate, call venv Python directly):

```bash
.\.venv310\Scripts\python.exe -m pip install --upgrade pip
.\.venv310\Scripts\python.exe -m pip install -r 2B/requirements_a2b.txt
```

Note: every command below can be run either with `python` (if venv is activated) or with the full venv Python path.

### Optional: Windows GPU Setup (DirectML)

If you want GPU acceleration on native Windows, use a separate environment with TensorFlow 2.10 + DirectML.

Important:
- Native Windows TensorFlow GPU via CUDA is not supported for newer TF versions.
- DirectML currently works with the TF 2.10 stack used below.
- Do not install `2B/requirements_a2b.txt` into this GPU env because it requires `tensorflow>=2.12` and will break DirectML compatibility.

Create and install GPU environment:

```bash
py -3.10 -m venv .venv-tf-gpu
.\.venv-tf-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-tf-gpu\Scripts\python.exe -m pip install "numpy<2" pandas==2.3.3 scikit-learn==1.7.2 openpyxl xlrd pytest
.\.venv-tf-gpu\Scripts\python.exe -m pip install tensorflow-cpu==2.10.0 tensorflow-directml-plugin
```

Quick GPU verification:

```bash
.\.venv-tf-gpu\Scripts\python.exe -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"
```

Expected outcome:
- TensorFlow version is `2.10.0`
- At least one GPU appears in the device list (DirectML adapter)

Run A2B with GPU environment:

```bash
Set-Location "C:\Users\Acer\OneDrive\Documents\Swin\13_Intro_AI"
.\.venv-tf-gpu\Scripts\Activate.ps1
$env:DML_VISIBLE_DEVICES="0"
$env:TF_CPP_MIN_LOG_LEVEL="2"
python 2B\run_tbrgs_gui.py
```

Notes:
- `DML_VISIBLE_DEVICES="0"` prefers the first adapter (commonly NVIDIA) and can reduce overhead from multi-adapter initialization.
- TensorFlow DirectML logs such as `Could not identify NUMA node` are informational on Windows.
- If you do not activate `.venv-tf-gpu`, run commands with `.\.venv-tf-gpu\Scripts\python.exe` instead of `python`.

### Run A2B from CLI

```bash
python 2B/run_tbrgs_cli.py --config 2B/config/tbrgs_defaults.json --origin <ORIGIN_SCATS_ID> --destination <DEST_SCATS_ID> --top-k <K> --model <lstm|gru|rf|best> --algorithm <DFS|BFS|GBFS|AS|CUS1|CUS2> --hour <0-23> --metrics-out 2B/data/output/tbrgs_metrics_summary.csv
```

If not activated, use:

```bash
.\.venv310\Scripts\python.exe 2B/run_tbrgs_cli.py --config 2B/config/tbrgs_defaults.json --origin <ORIGIN_SCATS_ID> --destination <DEST_SCATS_ID> --top-k <K> --model <lstm|gru|rf|best> --algorithm <DFS|BFS|GBFS|AS|CUS1|CUS2> --hour <0-23> --metrics-out 2B/data/output/tbrgs_metrics_summary.csv
```

Example:

```bash
python 2B/run_tbrgs_cli.py --config 2B/config/tbrgs_defaults.json --origin 2000 --destination 3002 --top-k 5 --model best --algorithm CUS2 --hour 9 --metrics-out 2B/data/output/tbrgs_metrics_summary.csv
```

Notes:
- Returned routes are ordered fastest to slowest by predicted total travel time.
- `--model best` uses the best-performing model per site from evaluation metrics.

### Run A2B GUI

```bash
python 2B/run_tbrgs_gui.py
```

If not activated:

```bash
.\.venv310\Scripts\python.exe 2B/run_tbrgs_gui.py
```

GUI workflow:
- Select model, pathfinding algorithm, and hour of day.
- Left-click a node to set origin.
- Right-click a node to set destination.
- Click Find Path to display the top-k routes and total time.

### Run A2B tests

```bash
python -m pytest 2B/tests_a2b -q
```

If not activated:

```bash
.\.venv310\Scripts\python.exe -m pytest 2B/tests_a2b -q
```

Current A2B test coverage includes:
- Data sequence construction and split behavior.
- Hourly aggregation and lookback extraction.
- Road graph neighbor constraints and routing connectivity.
- Route result properties: start/end validity, top-k bound, uniqueness, and fastest-to-slowest ordering.
- Traffic speed/time behavior under varying flow and distance.

Expected result:
- `20 passed` on a clean setup.

