from __future__ import annotations

import heapq
import math

from ..common import expand_children
from ..heuristics import heuristic
from ..models import NodeFactory, Problem, SearchNode


def solve_astar(problem: Problem) -> tuple[SearchNode | None, int]:
    factory = NodeFactory()
    root = factory.create(problem.origin, None, g_cost=0.0, depth=0)

    frontier: list[tuple[float, int, int, SearchNode]] = []
    root_f = root.g_cost + heuristic(problem, root.state)
    heapq.heappush(frontier, (root_f, root.state, root.created_order, root))

    best_g: dict[int, float] = {root.state: 0.0}
    explored: set[int] = set()

    while frontier:
        _, _, _, current = heapq.heappop(frontier)
        if current.state in explored:
            continue
        explored.add(current.state)

        if current.state in problem.destinations:
            return current, factory.created_count

        for child_state, edge_cost in expand_children(problem, current):
            if child_state in explored:
                continue
            tentative_g = current.g_cost + edge_cost
            if tentative_g >= best_g.get(child_state, math.inf):
                continue

            best_g[child_state] = tentative_g
            child = factory.create(
                state=child_state,
                parent=current,
                g_cost=tentative_g,
                depth=current.depth + 1,
            )
            f = tentative_g + heuristic(problem, child_state)
            heapq.heappush(frontier, (f, child.state, child.created_order, child))

    return None, factory.created_count
