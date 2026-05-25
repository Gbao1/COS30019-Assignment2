from __future__ import annotations

import argparse
from pathlib import Path

from route_search.output import format_output
from route_search.parser import parse_problem
from route_search.solver import solve


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-based search for route finding")
    parser.add_argument("filename", help="Path to problem file")
    parser.add_argument("method", help="DFS | BFS | GBFS | AS | CUS1 | CUS2")
    args = parser.parse_args()

    file_path = Path(args.filename)
    problem = parse_problem(file_path)
    goal_node, nodes_created = solve(problem, args.method)
    print(format_output(file_path.name, args.method, goal_node, nodes_created))


if __name__ == "__main__":
    main()
