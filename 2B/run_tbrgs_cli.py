from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tbrgs.pipeline import build_context, recommend_routes


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "tbrgs_defaults.json"
DEFAULT_METRICS_OUT = SCRIPT_DIR / "data" / "output" / "tbrgs_metrics_summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic-Based Route Guidance System (Assignment 2B)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to JSON config file")
    parser.add_argument("--origin", type=int, required=False, help="Origin SCATS site number")
    parser.add_argument("--destination", type=int, required=False, help="Destination SCATS site number")
    parser.add_argument("--top-k", type=int, default=None, help="How many routes to return")
    parser.add_argument("--model", default="best", choices=["best", "lstm", "gru", "rf"], help="Prediction model")
    parser.add_argument(
        "--metrics-out",
        default=str(DEFAULT_METRICS_OUT),
        help="CSV output for model comparison",
    )
    args = parser.parse_args()

    ctx = build_context(args.config)
    runtime_cfg = ctx.config["runtime"]

    origin = int(args.origin if args.origin is not None else runtime_cfg["default_origin"])
    destination = int(args.destination if args.destination is not None else runtime_cfg["default_destination"])
    top_k = int(args.top_k if args.top_k is not None else runtime_cfg["default_top_k"])

    routes = recommend_routes(ctx=ctx, origin=origin, destination=destination, top_k=top_k, model_name=args.model)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.metrics_summary.to_csv(metrics_path, index=False)

    print("Model comparison saved to", metrics_path)
    if not routes:
        print("No feasible route found.")
        return

    print(f"Routes from {origin} to {destination} using model={args.model}:")

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
