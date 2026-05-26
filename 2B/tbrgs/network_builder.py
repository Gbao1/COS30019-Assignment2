from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RoadGraph:
    nodes: dict[int, tuple[float, float]]
    edges: dict[int, list[int]]
    distance_km: dict[tuple[int, int], float]


def _clean_site_coords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["site_id"] = pd.to_numeric(out["site_id"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna().drop_duplicates(subset=["site_id"])

    # Remove obviously invalid coordinates that collapse map scaling (e.g., 0,0).
    out = out[(out["lat"] != 0.0) & (out["lon"] != 0.0)]
    out = out[(out["lat"] >= -90.0) & (out["lat"] <= 90.0)]
    out = out[(out["lon"] >= -180.0) & (out["lon"] <= 180.0)]
    out["site_id"] = out["site_id"].astype(int)
    return out


def load_scats_sites_from_traffic_workbook(traffic_file: str | Path) -> pd.DataFrame:
    path = Path(traffic_file)
    if not path.exists():
        raise FileNotFoundError(f"Traffic workbook not found: {path}")

    xls = pd.ExcelFile(path)
    if "Data" not in xls.sheet_names:
        raise ValueError("Traffic workbook does not contain expected 'Data' sheet.")

    df = pd.read_excel(path, sheet_name="Data")
    df.columns = [str(c).strip() for c in df.columns]

    if df.columns.str.startswith("Unnamed:").any() and len(df) > 0:
        first_row = df.iloc[0].astype(str).str.strip()
        if first_row.str.contains("SCATS", case=False, na=False).any():
            df = df.copy()
            df.columns = first_row.tolist()
            df = df.iloc[1:].reset_index(drop=True)

    required = ["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]
    if not all(col in df.columns for col in required):
        raise ValueError("Traffic workbook Data sheet does not include SCATS Number/NB_LATITUDE/NB_LONGITUDE.")

    out = df[required].copy()
    out.columns = ["site_id", "lat", "lon"]
    return _clean_site_coords(out)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    columns = {str(c).lower().strip(): str(c) for c in df.columns}
    for c in candidates:
        if c in columns:
            return columns[c]
    for col in df.columns:
        col_l = str(col).lower()
        if any(token in col_l for token in candidates):
            return str(col)
    return None


def load_scats_sites(site_file: str | Path, fallback_file: str | Path | None = None) -> pd.DataFrame:
    path = Path(site_file)
    if path.exists():
        try:
            df = pd.read_excel(path)
            scats_col = _find_col(df, ["scats number", "scats", "site", "site number"])
            lat_col = _find_col(df, ["lat", "latitude"])
            lon_col = _find_col(df, ["lon", "long", "longitude"])
            if scats_col and lat_col and lon_col:
                out = df[[scats_col, lat_col, lon_col]].copy()
                out.columns = ["site_id", "lat", "lon"]
                return _clean_site_coords(out)
        except Exception:
            # Some provided xls files contain legacy workbook formulas that fail in xlrd.
            # In that case we continue with the approved CSV fallback dataset.
            pass

    if fallback_file is None:
        raise ValueError("No usable SCATS site file and no fallback file provided.")

    fallback = pd.read_csv(fallback_file)
    fallback.columns = [str(c).strip() for c in fallback.columns]
    id_col = _find_col(fallback, ["tfm_id", "site", "objectid"])
    lat_col = _find_col(fallback, ["y", "lat", "latitude"])
    lon_col = _find_col(fallback, ["x", "lon", "long", "longitude"])
    if not id_col or not lat_col or not lon_col:
        raise ValueError("Fallback site file does not contain expected id/lat/lon columns.")

    out = fallback[[id_col, lat_col, lon_col]].copy()
    out.columns = ["site_id", "lat", "lon"]
    return _clean_site_coords(out)


def build_knn_road_graph(sites_df: pd.DataFrame, k_neighbors: int, max_neighbor_distance_km: float) -> RoadGraph:
    nodes = {
        int(row.site_id): (float(row.lat), float(row.lon))
        for row in sites_df.itertuples(index=False)
    }

    edges: dict[int, list[int]] = {node_id: [] for node_id in nodes}
    distance_km: dict[tuple[int, int], float] = {}

    node_ids = list(nodes.keys())
    for source in node_ids:
        lat1, lon1 = nodes[source]
        distances: list[tuple[float, int]] = []
        for target in node_ids:
            if source == target:
                continue
            lat2, lon2 = nodes[target]
            d = haversine_km(lat1, lon1, lat2, lon2)
            if d <= max_neighbor_distance_km:
                distances.append((d, target))

        distances.sort(key=lambda x: x[0])
        for d, target in distances[:k_neighbors]:
            edges[source].append(target)
            distance_km[(source, target)] = d

    return RoadGraph(nodes=nodes, edges=edges, distance_km=distance_km)
