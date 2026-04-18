from __future__ import annotations

import heapq
import math

from ..common import expand_children
from ..models import NodeFactory, Problem, SearchNode


def solve_ucs(problem: Problem) -> tuple[SearchNode | None, int]:
    # CUS2: Uniform Cost Search (Dijkstra-style with directed weighted edges).
    factory = NodeFactory()
    root = factory.create(problem.origin, None, g_cost=0.0, depth=0)

    frontier: list[tuple[float, int, int, SearchNode]] = []
    heapq.heappush(frontier, (0.0, root.state, root.created_order, root))

    best_g: dict[int, float] = {root.state: 0.0}

    while frontier:
        popped_cost, _, _, current = heapq.heappop(frontier)
        if popped_cost > best_g.get(current.state, math.inf):
            continue

        if current.state in problem.destinations:
            return current, factory.created_count

        for child_state, edge_cost in expand_children(problem, current):
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
            heapq.heappush(frontier, (tentative_g, child.state, child.created_order, child))

    return None, factory.created_count


def solve_astar_moves(problem: Problem) -> tuple[SearchNode | None, int]:
    # Backward-compatible alias for existing imports.
    return solve_ucs(problem)
