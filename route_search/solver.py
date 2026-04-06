from __future__ import annotations

from .algorithms import (
    solve_astar,
    solve_astar_moves,
    solve_bfs,
    solve_dfs,
    solve_gbfs,
    solve_ids,
)
from .models import Problem, SearchNode


def solve(problem: Problem, method: str) -> tuple[SearchNode | None, int]:
    key = method.upper()
    if key == "DFS":
        return solve_dfs(problem)
    if key == "BFS":
        return solve_bfs(problem)
    if key == "GBFS":
        return solve_gbfs(problem)
    if key == "AS":
        return solve_astar(problem)
    if key == "CUS1":
        return solve_ids(problem)
    if key == "CUS2":
        return solve_astar_moves(problem)
    raise ValueError("Unsupported method. Use one of: DFS, BFS, GBFS, AS, CUS1, CUS2")
