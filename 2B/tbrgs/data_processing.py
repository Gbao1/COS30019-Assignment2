from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for c in columns:
        cl = c.lower()
        if any(token in cl for token in candidates):
            return c
    return None


def _infer_flow_columns(columns: Sequence[str]) -> list[str]:
    flow_cols: list[str] = []
    pattern_list = [
        r"^v\d{1,2}$",
        r"^\d{1,2}:\d{2}$",
        r"^\d{1,2}\.\d{2}$",
        r"^volume_\d+$",
    ]
    for c in columns:
        cl = c.lower().strip()
        if any(re.match(p, cl) for p in pattern_list):
            flow_cols.append(c)
    return flow_cols


def _flow_col_to_step(flow_col: str) -> int:
    token = flow_col.lower().strip()
    if re.match(r"^v\d{1,2}$", token):
        return int(token[1:])
    if re.match(r"^\d{1,2}:\d{2}$", token):
        h, m = token.split(":")
        return int(h) * 4 + int(m) // 15
    if re.match(r"^\d{1,2}\.\d{2}$", token):
        h, m = token.split(".")
        return int(h) * 4 + int(m) // 15
    if re.match(r"^volume_\d+$", token):
        return int(token.split("_")[1])
    raise ValueError(f"Unsupported flow column format: {flow_col}")


def load_traffic_data(file_path: str | Path, interval_minutes: int = 15) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Traffic dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        # The assignment workbook stores traffic data in the "Data" sheet.
        xls = pd.ExcelFile(path)
        sheet_name = "Data" if "Data" in xls.sheet_names else xls.sheet_names[0]
        raw = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raw = pd.read_csv(path)

    raw.columns = [str(c).strip() for c in raw.columns]

    # Some VICRoads exports place the true header values in the first data row.
    if raw.columns.str.startswith("Unnamed:").any() and len(raw) > 0:
        first_row = raw.iloc[0].astype(str).str.strip()
        if first_row.str.contains("SCATS", case=False, na=False).any() and first_row.str.contains("V00", case=False, na=False).any():
            raw = raw.copy()
            raw.columns = first_row.tolist()
            raw = raw.iloc[1:].reset_index(drop=True)

    site_col = _find_column(raw.columns, ["scats", "site", "site_no", "site number", "scats number"])
    date_col = _find_column(raw.columns, ["date"])
    time_col = _find_column(raw.columns, ["time", "interval"])
    flow_col = _find_column(raw.columns, ["flow", "volume", "vehicles"])

    if site_col and date_col and time_col and flow_col:
        df = raw[[site_col, date_col, time_col, flow_col]].copy()
        df.columns = ["site_id", "date", "time", "flow"]
        df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
        df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
        df = df.dropna(subset=["site_id", "timestamp", "flow"]) 
        df["site_id"] = df["site_id"].astype(int)
        return df[["site_id", "timestamp", "flow"]].sort_values(["site_id", "timestamp"]).reset_index(drop=True)

    if not site_col or not date_col:
        raise ValueError(
            "Could not infer required columns. Expected either long format with site/date/time/flow "
            "or wide format with site/date and flow interval columns."
        )

    flow_cols = _infer_flow_columns(list(raw.columns))
    if not flow_cols:
        raise ValueError("No flow interval columns detected in wide-format traffic file.")

    melted = raw[[site_col, date_col] + flow_cols].melt(
        id_vars=[site_col, date_col],
        value_vars=flow_cols,
        var_name="interval_col",
        value_name="flow",
    )
    melted["step"] = melted["interval_col"].map(_flow_col_to_step)
    melted["timestamp"] = pd.to_datetime(melted[date_col], errors="coerce") + pd.to_timedelta(
        melted["step"] * interval_minutes,
        unit="minute",
    )
    melted["flow"] = pd.to_numeric(melted["flow"], errors="coerce")
    melted = melted.dropna(subset=[site_col, "timestamp", "flow"])
    melted[site_col] = pd.to_numeric(melted[site_col], errors="coerce").astype("Int64")
    melted = melted.dropna(subset=[site_col])

    result = melted.rename(columns={site_col: "site_id"})[["site_id", "timestamp", "flow"]].copy()
    result["site_id"] = result["site_id"].astype(int)
    result = result.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    return result


def to_hourly_flow(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.set_index("timestamp")
        .groupby("site_id")["flow"]
        .resample("1h")
        .sum()
        .rename("flow")
        .reset_index()
        .sort_values(["site_id", "timestamp"])
        .reset_index(drop=True)
    )
    return out


def build_supervised_sequences(series: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=float)
    if len(values) < lookback + horizon:
        return np.empty((0, lookback), dtype=float), np.empty((0, horizon), dtype=float)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for i in range(len(values) - lookback - horizon + 1):
        x_list.append(values[i : i + lookback])
        y_list.append(values[i + lookback : i + lookback + horizon])
    return np.asarray(x_list), np.asarray(y_list)


def split_sequences(x: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float) -> DatasetSplits:
    n = len(x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = max(1, n_train)
    n_val = max(1, n_val) if n - n_train > 1 else 0

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val = x[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    x_test = x[n_train + n_val :]
    y_test = y[n_train + n_val :]

    return DatasetSplits(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)


def latest_lookback_by_site(hourly_df: pd.DataFrame, lookback: int) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for site_id, g in hourly_df.groupby("site_id"):
        values = g.sort_values("timestamp")["flow"].to_numpy(dtype=float)
        if len(values) >= lookback:
            result[int(site_id)] = values[-lookback:]
    return result
