from __future__ import annotations

from dataclasses import dataclass

from route_search.algorithms.cus2 import solve_ucs
from route_search.common import reconstruct_path
from route_search.models import Problem

from .network_builder import RoadGraph
from .traffic_model import travel_time_seconds


@dataclass
class RouteResult:
    path: list[int]
    total_seconds: float


def _build_weighted_adjacency(
    graph: RoadGraph,
    predicted_flow_by_site: dict[int, float],
    speed_limit_kmh: float,
    intersection_delay_seconds: float,
    use_flow_from: str,
    assume_under_capacity: bool,
) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = {node: [] for node in graph.nodes}
    for src, targets in graph.edges.items():
        for dst in targets:
            dist_km = graph.distance_km.get((src, dst))
            if dist_km is None:
                continue
            flow_site = src if use_flow_from.lower() == "start" else dst
            flow = float(predicted_flow_by_site.get(flow_site, 0.0))
            sec = travel_time_seconds(
                distance_km=dist_km,
                flow_veh_per_hour=flow,
                speed_limit_kmh=speed_limit_kmh,
                intersection_delay_seconds=intersection_delay_seconds,
                assume_under_capacity=assume_under_capacity,
            )
            adj[src].append((dst, sec))
        adj[src].sort(key=lambda x: x[0])
    return adj


def _to_problem(
    nodes: dict[int, tuple[float, float]],
    weighted_adj: dict[int, list[tuple[int, float]]],
    origin: int,
    destination: int,
) -> Problem:
    return Problem(nodes=nodes, edges=weighted_adj, origin=origin, destinations=[destination])


def _path_cost(weighted_adj: dict[int, list[tuple[int, float]]], path: list[int]) -> float:
    edge_map: dict[tuple[int, int], float] = {}
    for src, neigh in weighted_adj.items():
        for dst, c in neigh:
            edge_map[(src, dst)] = c

    total = 0.0
    for i in range(len(path) - 1):
        total += edge_map[(path[i], path[i + 1])]
    return total


def _remove_root_path_edges(
    weighted_adj: dict[int, list[tuple[int, float]]],
    root_path: list[int],
    accepted_paths: list[list[int]],
) -> dict[int, list[tuple[int, float]]]:
    cloned = {k: list(v) for k, v in weighted_adj.items()}
    for p in accepted_paths:
        if len(p) > len(root_path) and p[: len(root_path)] == root_path:
            u = p[len(root_path) - 1]
            v = p[len(root_path)]
            cloned[u] = [(dst, c) for (dst, c) in cloned[u] if dst != v]
    return cloned


def _remove_root_nodes_except_spur(
    weighted_adj: dict[int, list[tuple[int, float]]],
    root_path: list[int],
    spur_node: int,
) -> dict[int, list[tuple[int, float]]]:
    cloned = {k: list(v) for k, v in weighted_adj.items()}
    root_set = set(root_path[:-1])
    for node in root_set:
        if node == spur_node:
            continue
        cloned[node] = []
        for src in cloned:
            cloned[src] = [(dst, c) for (dst, c) in cloned[src] if dst != node]
    return cloned


def top_k_routes_with_ucs(
    graph: RoadGraph,
    origin: int,
    destination: int,
    predicted_flow_by_site: dict[int, float],
    top_k: int,
    speed_limit_kmh: float,
    intersection_delay_seconds: float,
    use_flow_from: str = "start",
    assume_under_capacity: bool = True,
) -> list[RouteResult]:
    weighted_adj = _build_weighted_adjacency(
        graph=graph,
        predicted_flow_by_site=predicted_flow_by_site,
        speed_limit_kmh=speed_limit_kmh,
        intersection_delay_seconds=intersection_delay_seconds,
        use_flow_from=use_flow_from,
        assume_under_capacity=assume_under_capacity,
    )

    first_problem = _to_problem(graph.nodes, weighted_adj, origin, destination)
    first_goal, _ = solve_ucs(first_problem)
    if first_goal is None:
        return []

    first_path = reconstruct_path(first_goal)
    accepted_paths: list[list[int]] = [first_path]
    candidates: list[tuple[list[int], float]] = []

    for _ in range(1, top_k):
        prev_path = accepted_paths[-1]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            without_same_root = _remove_root_path_edges(weighted_adj, root_path, accepted_paths)
            pruned = _remove_root_nodes_except_spur(without_same_root, root_path, spur_node)

            spur_problem = _to_problem(graph.nodes, pruned, spur_node, destination)
            spur_goal, _ = solve_ucs(spur_problem)
            if spur_goal is None:
                continue

            spur_path = reconstruct_path(spur_goal)
            total_path = root_path[:-1] + spur_path
            if total_path in accepted_paths:
                continue
            total_cost = _path_cost(weighted_adj, total_path)
            candidates.append((total_path, total_cost))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1])
        best_path, _ = candidates.pop(0)
        accepted_paths.append(best_path)

    results = [RouteResult(path=p, total_seconds=_path_cost(weighted_adj, p)) for p in accepted_paths[:top_k]]
    return results
