from __future__ import annotations

from typing import List, Tuple

from .models import NodeId, Problem, SearchNode


def reconstruct_path(node: SearchNode) -> List[NodeId]:
    path: List[NodeId] = []
    cur: SearchNode | None = node
    while cur is not None:
        path.append(cur.state)
        cur = cur.parent
    path.reverse()
    return path


def expand_children(problem: Problem, node: SearchNode) -> List[Tuple[NodeId, float]]:
    return problem.edges.get(node.state, [])
