from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


NodeId = int


@dataclass
class Problem:
    nodes: Dict[NodeId, Tuple[float, float]]
    edges: Dict[NodeId, List[Tuple[NodeId, float]]]
    origin: NodeId
    destinations: List[NodeId]


@dataclass
class SearchNode:
    state: NodeId
    parent: Optional["SearchNode"]
    g_cost: float
    depth: int
    created_order: int = field(default=0)


class NodeFactory:
    def __init__(self) -> None:
        self.created_count = 0

    def create(
        self,
        state: NodeId,
        parent: Optional[SearchNode],
        g_cost: float,
        depth: int,
    ) -> SearchNode:
        self.created_count += 1
        return SearchNode(
            state=state,
            parent=parent,
            g_cost=g_cost,
            depth=depth,
            created_order=self.created_count,
        )
