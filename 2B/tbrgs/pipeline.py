from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
from .topk_routing import RouteResult, top_k_routes_with_search


@dataclass
class TBRGSContext:
    config: dict[str, Any]
    hourly_df: pd.DataFrame
    graph: RoadGraph
    site_evaluations: dict[int, SiteEvaluation]
    metrics_summary: pd.DataFrame
    hourly_profile: pd.DataFrame


def build_context(config_path: str | Path | None = None, selected_models: list[str] | None = None) -> TBRGSContext:
    cfg = load_config(config_path)
    if selected_models is not None:
        cfg.setdefault("runtime", {})["train_models"] = [m.lower() for m in selected_models]
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
    profile = (
        hourly_df.assign(hour=hourly_df["timestamp"].dt.hour)
        .groupby(["site_id", "hour"], as_index=False)["flow"]
        .mean()
        .rename(columns={"flow": "hourly_mean_flow"})
    )

    return TBRGSContext(
        config=cfg,
        hourly_df=hourly_df,
        graph=graph,
        site_evaluations=site_evals,
        metrics_summary=metrics,
        hourly_profile=profile,
    )


def predict_next_flow_by_site(
    ctx: TBRGSContext,
    preferred_model: str = "best",
    hour_of_day: int | None = None,
) -> dict[int, float]:
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

    if hour_of_day is None:
        return flow_pred

    base_mean = ctx.hourly_df.groupby("site_id", as_index=True)["flow"].mean()
    hour_rows = ctx.hourly_profile[ctx.hourly_profile["hour"] == int(hour_of_day)]
    hour_mean = hour_rows.set_index("site_id")["hourly_mean_flow"]

    adjusted: dict[int, float] = {}
    for site_id, pred in flow_pred.items():
        b = float(base_mean.get(site_id, np.nan))
        h = float(hour_mean.get(site_id, np.nan))
        if np.isnan(b) or b <= 1e-6 or np.isnan(h):
            adjusted[site_id] = pred
            continue
        adjusted[site_id] = max(0.0, pred * (h / b))
    return adjusted


def predict_directional_flow_by_edge(
    ctx: TBRGSContext,
    predicted_flow_by_site: dict[int, float],
    hour_of_day: int,
) -> dict[tuple[int, int], float]:
    hour_rows = ctx.hourly_profile[ctx.hourly_profile["hour"] == int(hour_of_day)]
    hour_mean = hour_rows.set_index("site_id")["hourly_mean_flow"]

    out: dict[tuple[int, int], float] = {}
    for src, targets in ctx.graph.edges.items():
        src_pred = float(predicted_flow_by_site.get(src, 0.0))
        src_hist = float(hour_mean.get(src, src_pred))
        for dst in targets:
            dst_pred = float(predicted_flow_by_site.get(dst, 0.0))
            dst_hist = float(hour_mean.get(dst, dst_pred))

            denom = max(src_hist + dst_hist, 1e-6)
            # Higher historical source-side activity at this hour increases
            # directional load for src->dst compared with dst->src.
            w_src = min(0.9, max(0.1, src_hist / denom))
            edge_flow = (w_src * src_pred) + ((1.0 - w_src) * dst_pred)
            out[(src, dst)] = max(0.0, edge_flow)
    return out
    return flow_pred


def recommend_routes(
    ctx: TBRGSContext,
    origin: int,
    destination: int,
    top_k: int,
    model_name: str = "best",
    path_method: str = "CUS2",
    hour_of_day: int | None = None,
) -> list[RouteResult]:
    traffic_cfg = ctx.config["traffic"]
    flow_pred = predict_next_flow_by_site(
        ctx=ctx,
        preferred_model=model_name,
        hour_of_day=hour_of_day,
    )
    flow_pred_edge = None
    if hour_of_day is not None:
        flow_pred_edge = predict_directional_flow_by_edge(
            ctx=ctx,
            predicted_flow_by_site=flow_pred,
            hour_of_day=hour_of_day,
        )

    return top_k_routes_with_search(
        graph=ctx.graph,
        origin=origin,
        destination=destination,
        predicted_flow_by_site=flow_pred,
        predicted_flow_by_link=flow_pred_edge,
        top_k=top_k,
        speed_limit_kmh=float(traffic_cfg["speed_limit_kmh"]),
        intersection_delay_seconds=float(traffic_cfg["intersection_delay_seconds"]),
        path_method=path_method,
        use_flow_from=str(traffic_cfg["use_flow_from"]),
        assume_under_capacity=bool(traffic_cfg["assume_under_capacity"]),
    )
