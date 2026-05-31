from __future__ import annotations

from collections import deque

from ..common import expand_children
from ..models import NodeFactory, Problem, SearchNode


def solve_bfs(problem: Problem) -> tuple[SearchNode | None, int]:
    factory = NodeFactory()
    root = factory.create(problem.origin, None, g_cost=0.0, depth=0)
    queue: deque[SearchNode] = deque([root])
    explored: set[int] = set()
    frontier_states: set[int] = {root.state}

    while queue:
        current = queue.popleft()
        frontier_states.discard(current.state)
        if current.state in explored:
            continue
        explored.add(current.state)

        if current.state in problem.destinations:
            return current, factory.created_count

        for child_state, edge_cost in expand_children(problem, current):
            if child_state in explored or child_state in frontier_states:
                continue
            child = factory.create(
                state=child_state,
                parent=current,
                g_cost=current.g_cost + edge_cost,
                depth=current.depth + 1,
            )
            queue.append(child)
            frontier_states.add(child_state)

    return None, factory.created_count
