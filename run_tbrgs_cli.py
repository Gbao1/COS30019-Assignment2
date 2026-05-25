from __future__ import annotations

import argparse

from tbrgs.pipeline import build_context, recommend_routes


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic-Based Route Guidance System (Assignment 2B)")
    parser.add_argument("--config", default="config/tbrgs_defaults.json", help="Path to JSON config file")
    parser.add_argument("--origin", type=int, required=False, help="Origin SCATS site number")
    parser.add_argument("--destination", type=int, required=False, help="Destination SCATS site number")
    parser.add_argument("--top-k", type=int, default=None, help="How many routes to return")
    parser.add_argument("--model", default="best", choices=["best", "lstm", "gru", "rf"], help="Prediction model")
    parser.add_argument("--metrics-out", default="tbrgs_metrics_summary.csv", help="CSV output for model comparison")
    args = parser.parse_args()

    ctx = build_context(args.config)
    runtime_cfg = ctx.config["runtime"]

    origin = int(args.origin if args.origin is not None else runtime_cfg["default_origin"])
    destination = int(args.destination if args.destination is not None else runtime_cfg["default_destination"])
    top_k = int(args.top_k if args.top_k is not None else runtime_cfg["default_top_k"])

    routes = recommend_routes(ctx=ctx, origin=origin, destination=destination, top_k=top_k, model_name=args.model)
    ctx.metrics_summary.to_csv(args.metrics_out, index=False)

    print("Model comparison saved to", args.metrics_out)
    if not routes:
        print("No feasible route found.")
        return

    print(f"Routes from {origin} to {destination} using model={args.model}:")
    for idx, route in enumerate(routes, start=1):
        nodes = " -> ".join(str(n) for n in route.path)
        mins = route.total_seconds / 60.0
        print(f"{idx}. {nodes} | {mins:.2f} minutes")


if __name__ == "__main__":
    main()
