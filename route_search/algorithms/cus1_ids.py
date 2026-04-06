from __future__ import annotations

from ..common import expand_children
from ..models import NodeFactory, Problem, SearchNode


def solve_ids(problem: Problem) -> tuple[SearchNode | None, int]:
    # CUS1: iterative deepening DFS (uninformed).
    factory = NodeFactory()

    def dls(limit: int) -> SearchNode | None:
        root = factory.create(problem.origin, None, g_cost=0.0, depth=0)
        stack: list[SearchNode] = [root]

        while stack:
            current = stack.pop()
            if current.state in problem.destinations:
                return current
            if current.depth >= limit:
                continue

            children = expand_children(problem, current)
            for child_state, edge_cost in reversed(children):
                # Avoid immediate path cycles.
                ancestor = current
                cycle = False
                while ancestor is not None:
                    if ancestor.state == child_state:
                        cycle = True
                        break
                    ancestor = ancestor.parent
                if cycle:
                    continue

                child = factory.create(
                    state=child_state,
                    parent=current,
                    g_cost=current.g_cost + edge_cost,
                    depth=current.depth + 1,
                )
                stack.append(child)

        return None

    max_depth = max(1, len(problem.nodes) * 3)
    for depth_limit in range(max_depth + 1):
        found = dls(depth_limit)
        if found is not None:
            return found, factory.created_count

    return None, factory.created_count
