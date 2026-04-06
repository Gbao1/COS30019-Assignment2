from __future__ import annotations

from typing import List

from ..common import expand_children
from ..models import NodeFactory, Problem, SearchNode


def solve_dfs(problem: Problem) -> tuple[SearchNode | None, int]:
    factory = NodeFactory()
    root = factory.create(problem.origin, None, g_cost=0.0, depth=0)
    stack: List[SearchNode] = [root]
    explored: set[int] = set()

    while stack:
        current = stack.pop()
        if current.state in explored:
            continue
        explored.add(current.state)

        if current.state in problem.destinations:
            return current, factory.created_count

        children = expand_children(problem, current)
        # Stack is LIFO, so reverse to keep ascending expansion order.
        for child_state, edge_cost in reversed(children):
            if child_state in explored:
                continue
            child = factory.create(
                state=child_state,
                parent=current,
                g_cost=current.g_cost + edge_cost,
                depth=current.depth + 1,
            )
            stack.append(child)

    return None, factory.created_count
