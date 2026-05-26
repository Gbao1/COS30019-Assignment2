from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from .data_processing import build_supervised_sequences, split_sequences
from .modeling import SiteModels, predict_unscaled, regression_metrics, train_site_models


@dataclass
class SiteEvaluation:
    models: SiteModels
    metrics_df: pd.DataFrame
    best_model_name: str


def evaluate_site(
    hourly_site_df: pd.DataFrame,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    lstm_units: int,
    gru_units: int,
    rf_estimators: int,
    epochs: int,
    batch_size: int,
    random_seed: int,
    selected_models: list[str] | None = None,
) -> SiteEvaluation:
    series = hourly_site_df.sort_values("timestamp")["flow"].to_numpy(dtype=float)
    x, y = build_supervised_sequences(series=series, lookback=lookback, horizon=horizon)
    splits = split_sequences(x=x, y=y, train_ratio=train_ratio, val_ratio=val_ratio)

    models = train_site_models(
        x_train=splits.x_train,
        y_train=splits.y_train,
        x_val=splits.x_val,
        y_val=splits.y_val,
        lstm_units=lstm_units,
        gru_units=gru_units,
        rf_estimators=rf_estimators,
        epochs=epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        selected_models=selected_models,
    )

    active_models = ["lstm", "gru", "rf"] if selected_models is None else [m.lower() for m in selected_models]
    rows: list[dict[str, float | str | int]] = []
    for name in active_models:
        started = time.perf_counter()
        y_pred = predict_unscaled(models=models, x=splits.x_test, model_name=name)
        infer_sec = time.perf_counter() - started
        m = regression_metrics(y_true=splits.y_test, y_pred=y_pred)
        rows.append(
            {
                "model": name,
                "samples": int(len(splits.x_test)),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
                "inference_seconds": infer_sec,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best_model_name = str(metrics_df.iloc[0]["model"])
    return SiteEvaluation(models=models, metrics_df=metrics_df, best_model_name=best_model_name)


def evaluate_all_sites(hourly_df: pd.DataFrame, config: dict) -> tuple[dict[int, SiteEvaluation], pd.DataFrame]:
    data_cfg = config["data"]
    model_cfg = config["models"]

    site_results: dict[int, SiteEvaluation] = {}
    all_rows: list[pd.DataFrame] = []

    selected_models_cfg = config.get("runtime", {}).get("train_models")
    selected_models = selected_models_cfg if isinstance(selected_models_cfg, list) else None

    for site_id, site_df in hourly_df.groupby("site_id"):
        site_df = site_df.sort_values("timestamp")
        if len(site_df) < data_cfg["lookback_steps"] + data_cfg["horizon_steps"] + 5:
            continue

        result = evaluate_site(
            hourly_site_df=site_df,
            lookback=int(data_cfg["lookback_steps"]),
            horizon=int(data_cfg["horizon_steps"]),
            train_ratio=float(data_cfg["train_ratio"]),
            val_ratio=float(data_cfg["val_ratio"]),
            lstm_units=int(model_cfg["lstm_units"]),
            gru_units=int(model_cfg["gru_units"]),
            rf_estimators=int(model_cfg["random_forest_estimators"]),
            epochs=int(model_cfg["epochs"]),
            batch_size=int(model_cfg["batch_size"]),
            random_seed=int(model_cfg["random_seed"]),
            selected_models=selected_models,
        )
        site_results[int(site_id)] = result
        row = result.metrics_df.copy()
        row["site_id"] = int(site_id)
        all_rows.append(row)

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
    else:
        summary = pd.DataFrame(columns=["site_id", "model", "samples", "mae", "rmse", "mape", "inference_seconds"])
    return site_results, summary
