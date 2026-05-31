from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

# Suppress TensorFlow INFO logs in CLI output while keeping warnings/errors.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tbrgs.pipeline import build_context, recommend_routes


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "tbrgs_defaults.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic-Based Route Guidance System (Assignment 2B)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to JSON config file")
    parser.add_argument("--origin", type=int, required=False, help="Origin SCATS site number")
    parser.add_argument("--destination", type=int, required=False, help="Destination SCATS site number")
    parser.add_argument("--top-k", type=int, default=None, help="How many routes to return")
    parser.add_argument("--model", default="best", choices=["best", "lstm", "gru", "rf"], help="Prediction model")
    parser.add_argument(
        "--algorithm",
        default="CUS2",
        choices=["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"],
        help="Pathfinding algorithm",
    )
    parser.add_argument("--hour", type=int, default=None, help="Hour of day (0-23) for directional prediction")
    args = parser.parse_args()

    selected_models = None if args.model == "best" else [args.model]
    ctx = build_context(args.config, selected_models=selected_models)
    runtime_cfg = ctx.config["runtime"]

    origin = int(args.origin if args.origin is not None else runtime_cfg["default_origin"])
    destination = int(args.destination if args.destination is not None else runtime_cfg["default_destination"])
    top_k = int(args.top_k if args.top_k is not None else runtime_cfg["default_top_k"])
    hour_of_day = None if args.hour is None else max(0, min(23, int(args.hour)))

    routes = recommend_routes(
        ctx=ctx,
        origin=origin,
        destination=destination,
        top_k=top_k,
        model_name=args.model,
        path_method=args.algorithm,
        hour_of_day=hour_of_day,
    )
    if not routes:
        print("No feasible route found.")
        return

    print(f"Routes from {origin} to {destination} using model={args.model}, algorithm={args.algorithm}, hour={hour_of_day if hour_of_day is not None else 'latest'}:")

    best_model_by_site = {site_id: ev.best_model_name for site_id, ev in ctx.site_evaluations.items()}
    if args.model == "best":
        # Show model distribution so marker can see what "best" resolved to.
        model_counts = Counter(best_model_by_site.values())
        if model_counts:
            summary = ", ".join(f"{name}={count}" for name, count in sorted(model_counts.items()))
            print(f"Best-model selection across trained sites: {summary}")

    for idx, route in enumerate(routes, start=1):
        nodes = " -> ".join(str(n) for n in route.path)
        mins = route.total_seconds / 60.0

        if args.model == "best":
            route_models = [best_model_by_site.get(node) for node in route.path if node in best_model_by_site]
            unique_route_models = [m for m in dict.fromkeys(route_models) if m is not None]
            model_info = "/".join(unique_route_models) if unique_route_models else "unknown"
            print(f"{idx}. {nodes} | {mins:.2f} minutes | models on path: {model_info}")
        else:
            print(f"{idx}. {nodes} | {mins:.2f} minutes | model used: {args.model}")


if __name__ == "__main__":
    main()
