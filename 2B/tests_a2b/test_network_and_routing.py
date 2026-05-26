from __future__ import annotations

import pandas as pd

from tbrgs.network_builder import build_knn_road_graph, haversine_km
from tbrgs.topk_routing import top_k_routes_with_ucs


def _tiny_sites() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": [1, 2, 3, 4],
            "lat": [-37.8, -37.801, -37.802, -37.803],
            "lon": [145.0, 145.001, 145.002, 145.003],
        }
    )


def test_haversine_zero_distance() -> None:
    assert haversine_km(-37.8, 145.0, -37.8, 145.0) == 0.0


def test_build_knn_graph_has_nodes() -> None:
    graph = build_knn_road_graph(_tiny_sites(), k_neighbors=2, max_neighbor_distance_km=5.0)
    assert len(graph.nodes) == 4


def test_build_knn_graph_respects_neighbor_limit() -> None:
    graph = build_knn_road_graph(_tiny_sites(), k_neighbors=2, max_neighbor_distance_km=5.0)
    assert all(len(v) <= 2 for v in graph.edges.values())


def test_top_k_routes_returns_at_least_one_when_connected() -> None:
    graph = build_knn_road_graph(_tiny_sites(), k_neighbors=3, max_neighbor_distance_km=5.0)
    flows = {1: 500.0, 2: 600.0, 3: 700.0, 4: 800.0}
    routes = top_k_routes_with_ucs(
        graph=graph,
        origin=1,
        destination=4,
        predicted_flow_by_site=flows,
        top_k=3,
        speed_limit_kmh=60.0,
        intersection_delay_seconds=30.0,
    )
    assert len(routes) >= 1


def test_top_k_routes_respects_max_k() -> None:
    graph = build_knn_road_graph(_tiny_sites(), k_neighbors=3, max_neighbor_distance_km=5.0)
    flows = {1: 500.0, 2: 600.0, 3: 700.0, 4: 800.0}
    routes = top_k_routes_with_ucs(
        graph=graph,
        origin=1,
        destination=4,
        predicted_flow_by_site=flows,
        top_k=2,
        speed_limit_kmh=60.0,
        intersection_delay_seconds=30.0,
    )
    assert len(routes) <= 2


def test_top_k_routes_paths_start_and_end_correctly() -> None:
    graph = build_knn_road_graph(_tiny_sites(), k_neighbors=3, max_neighbor_distance_km=5.0)
    flows = {1: 500.0, 2: 600.0, 3: 700.0, 4: 800.0}
    routes = top_k_routes_with_ucs(
        graph=graph,
        origin=1,
        destination=4,
        predicted_flow_by_site=flows,
        top_k=3,
        speed_limit_kmh=60.0,
        intersection_delay_seconds=30.0,
    )
    assert all(r.path[0] == 1 and r.path[-1] == 4 for r in routes)


def test_top_k_routes_are_sorted_fastest_to_slowest() -> None:
    graph = build_knn_road_graph(_tiny_sites(), k_neighbors=3, max_neighbor_distance_km=5.0)
    flows = {1: 500.0, 2: 600.0, 3: 700.0, 4: 800.0}
    routes = top_k_routes_with_ucs(
        graph=graph,
        origin=1,
        destination=4,
        predicted_flow_by_site=flows,
        top_k=5,
        speed_limit_kmh=60.0,
        intersection_delay_seconds=30.0,
    )
    totals = [r.total_seconds for r in routes]
    assert totals == sorted(totals)


def test_top_k_routes_paths_are_unique() -> None:
    graph = build_knn_road_graph(_tiny_sites(), k_neighbors=3, max_neighbor_distance_km=5.0)
    flows = {1: 500.0, 2: 600.0, 3: 700.0, 4: 800.0}
    routes = top_k_routes_with_ucs(
        graph=graph,
        origin=1,
        destination=4,
        predicted_flow_by_site=flows,
        top_k=5,
        speed_limit_kmh=60.0,
        intersection_delay_seconds=30.0,
    )
    unique_paths = {tuple(r.path) for r in routes}
    assert len(unique_paths) == len(routes)
