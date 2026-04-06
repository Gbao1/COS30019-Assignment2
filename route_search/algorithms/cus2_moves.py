from __future__ import annotations

import heapq
import math

from ..common import expand_children
from ..heuristics import heuristic
from ..models import NodeFactory, Problem, SearchNode


def solve_astar_moves(problem: Problem) -> tuple[SearchNode | None, int]:
    # CUS2: informed shortest path in number of moves (unit step cost).
    factory = NodeFactory()
    root = factory.create(problem.origin, None, g_cost=0.0, depth=0)

    frontier: list[tuple[float, int, int, SearchNode]] = []
    root_f = heuristic(problem, root.state)
    heapq.heappush(frontier, (root_f, root.state, root.created_order, root))

    best_depth: dict[int, int] = {root.state: 0}
    explored: set[int] = set()

    while frontier:
        _, _, _, current = heapq.heappop(frontier)
        if current.state in explored:
            continue
        explored.add(current.state)

        if current.state in problem.destinations:
            return current, factory.created_count

        for child_state, _edge_cost in expand_children(problem, current):
            if child_state in explored:
                continue
            new_depth = current.depth + 1
            if new_depth >= best_depth.get(child_state, math.inf):
                continue
            best_depth[child_state] = new_depth

            child = factory.create(
                state=child_state,
                parent=current,
                g_cost=0.0,
                depth=new_depth,
            )
            # f = steps_so_far + heuristic estimate to nearest goal.
            f = new_depth + heuristic(problem, child_state)
            heapq.heappush(frontier, (f, child.state, child.created_order, child))

    return None, factory.created_count
