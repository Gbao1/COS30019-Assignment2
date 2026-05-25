from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import load_config
from .data_processing import latest_lookback_by_site, load_traffic_data, to_hourly_flow
from .evaluation import SiteEvaluation, evaluate_all_sites
from .modeling import predict_unscaled
from .network_builder import (
    RoadGraph,
    build_knn_road_graph,
    load_scats_sites,
    load_scats_sites_from_traffic_workbook,
)
from .topk_routing import RouteResult, top_k_routes_with_ucs


@dataclass
class TBRGSContext:
    config: dict[str, Any]
    hourly_df: pd.DataFrame
    graph: RoadGraph
    site_evaluations: dict[int, SiteEvaluation]
    metrics_summary: pd.DataFrame


def build_context(config_path: str | Path | None = None) -> TBRGSContext:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    net_cfg = cfg["network"]

    traffic_df = load_traffic_data(
        file_path=data_cfg["traffic_file"],
        interval_minutes=int(data_cfg["time_interval_minutes"]),
    )
    hourly_df = to_hourly_flow(traffic_df)

    try:
        sites_df = load_scats_sites_from_traffic_workbook(data_cfg["traffic_file"])
    except Exception:
        sites_df = load_scats_sites(
            site_file=data_cfg["site_file"],
            fallback_file=data_cfg["location_fallback_file"],
        )

    sites_df = sites_df[sites_df["site_id"].isin(hourly_df["site_id"].unique())].copy()
    graph = build_knn_road_graph(
        sites_df=sites_df,
        k_neighbors=int(net_cfg["k_nearest_neighbors"]),
        max_neighbor_distance_km=float(net_cfg["max_neighbor_distance_km"]),
    )

    site_evals, metrics = evaluate_all_sites(hourly_df=hourly_df, config=cfg)

    return TBRGSContext(
        config=cfg,
        hourly_df=hourly_df,
        graph=graph,
        site_evaluations=site_evals,
        metrics_summary=metrics,
    )


def predict_next_flow_by_site(ctx: TBRGSContext, preferred_model: str = "best") -> dict[int, float]:
    lookback = int(ctx.config["data"]["lookback_steps"])
    latest = latest_lookback_by_site(ctx.hourly_df, lookback=lookback)
    flow_pred: dict[int, float] = {}

    for site_id, x_last in latest.items():
        site_eval = ctx.site_evaluations.get(site_id)
        if site_eval is None:
            continue
        model_name = site_eval.best_model_name if preferred_model == "best" else preferred_model
        pred = predict_unscaled(site_eval.models, x_last.reshape(1, -1), model_name=model_name)
        flow_pred[site_id] = float(max(0.0, pred.ravel()[0]))
    return flow_pred


def recommend_routes(
    ctx: TBRGSContext,
    origin: int,
    destination: int,
    top_k: int,
    model_name: str = "best",
) -> list[RouteResult]:
    traffic_cfg = ctx.config["traffic"]
    flow_pred = predict_next_flow_by_site(ctx=ctx, preferred_model=model_name)

    return top_k_routes_with_ucs(
        graph=ctx.graph,
        origin=origin,
        destination=destination,
        predicted_flow_by_site=flow_pred,
        top_k=top_k,
        speed_limit_kmh=float(traffic_cfg["speed_limit_kmh"]),
        intersection_delay_seconds=float(traffic_cfg["intersection_delay_seconds"]),
        use_flow_from=str(traffic_cfg["use_flow_from"]),
        assume_under_capacity=bool(traffic_cfg["assume_under_capacity"]),
    )
