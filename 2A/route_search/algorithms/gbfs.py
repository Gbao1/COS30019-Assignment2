from __future__ import annotations

import heapq

from ..common import expand_children
from ..heuristics import heuristic
from ..models import NodeFactory, Problem, SearchNode


def solve_gbfs(problem: Problem) -> tuple[SearchNode | None, int]:
    factory = NodeFactory()
    root = factory.create(problem.origin, None, g_cost=0.0, depth=0)

    frontier: list[tuple[float, int, int, SearchNode]] = []
    heapq.heappush(frontier, (heuristic(problem, root.state), root.state, root.created_order, root))
    explored: set[int] = set()
    frontier_best_h: dict[int, float] = {root.state: heuristic(problem, root.state)}

    while frontier:
        _, _, _, current = heapq.heappop(frontier)
        if current.state in explored:
            continue

        frontier_best_h.pop(current.state, None)
        explored.add(current.state)

        if current.state in problem.destinations:
            return current, factory.created_count

        for child_state, edge_cost in expand_children(problem, current):
            if child_state in explored:
                continue
            h = heuristic(problem, child_state)
            prev_h = frontier_best_h.get(child_state)
            if prev_h is not None and h >= prev_h:
                continue
            child = factory.create(
                state=child_state,
                parent=current,
                g_cost=current.g_cost + edge_cost,
                depth=current.depth + 1,
            )
            frontier_best_h[child_state] = h
            heapq.heappush(frontier, (h, child.state, child.created_order, child))

    return None, factory.created_count
