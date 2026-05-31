from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import NodeId, Problem


def parse_problem(file_path: Path) -> Problem:
    lines = [line.strip().lstrip("\ufeff") for line in file_path.read_text(encoding="utf-8").splitlines()]

    mode = None
    nodes: Dict[NodeId, Tuple[float, float]] = {}
    edges_raw: List[Tuple[NodeId, NodeId, float]] = []
    origin: Optional[NodeId] = None
    destinations: List[NodeId] = []

    for line in lines:
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("nodes:"):
            mode = "nodes"
            continue
        if lower.startswith("edges:"):
            mode = "edges"
            continue
        if lower.startswith("origin:"):
            payload = line.split(":", 1)[1].strip()
            if payload:
                origin = int(payload)
                mode = None
            else:
                mode = "origin_value"
            continue
        if lower.startswith("destinations:"):
            payload = line.split(":", 1)[1].strip()
            if payload:
                destinations = [int(x.strip()) for x in payload.split(";") if x.strip()]
                mode = None
            else:
                mode = "destinations_value"
            continue

        if mode == "origin_value":
            origin = int(line)
            mode = None
            continue

        if mode == "destinations_value":
            destinations = [int(x.strip()) for x in line.split(";") if x.strip()]
            mode = None
            continue

        if mode == "nodes":
            # Format: 1: (4,1)
            left, right = line.split(":", 1)
            node_id = int(left.strip())
            coord = right.strip().strip("()")
            x_str, y_str = [v.strip() for v in coord.split(",")]
            nodes[node_id] = (float(x_str), float(y_str))
        elif mode == "edges":
            # Format: (2,1): 4
            left, right = line.split(":", 1)
            src_dst = left.strip().strip("()")
            src_str, dst_str = [v.strip() for v in src_dst.split(",")]
            edges_raw.append((int(src_str), int(dst_str), float(right.strip())))

    if origin is None:
        raise ValueError("Missing 'Origin:' in input file")
    if not destinations:
        raise ValueError("Missing or empty 'Destinations:' in input file")

    edges: Dict[NodeId, List[Tuple[NodeId, float]]] = {n: [] for n in nodes}
    for src, dst, cost in edges_raw:
        if src not in edges:
            edges[src] = []
        edges[src].append((dst, cost))

    for src in edges:
        edges[src].sort(key=lambda t: t[0])

    return Problem(nodes=nodes, edges=edges, origin=origin, destinations=destinations)
