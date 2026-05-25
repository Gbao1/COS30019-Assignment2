from __future__ import annotations

import math

from .models import NodeId, Problem


def heuristic(problem: Problem, node_id: NodeId) -> float:
    x, y = problem.nodes[node_id]
    best = math.inf
    for goal in problem.destinations:
        gx, gy = problem.nodes[goal]
        d = math.dist((x, y), (gx, gy))
        if d < best:
            best = d
    return best
