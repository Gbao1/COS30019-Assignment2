from __future__ import annotations

import numpy as np
import pandas as pd

from tbrgs.data_processing import (
    build_supervised_sequences,
    latest_lookback_by_site,
    split_sequences,
    to_hourly_flow,
)


def test_build_sequences_count_matches_formula() -> None:
    x, y = build_supervised_sequences(np.arange(12), lookback=4, horizon=2)
    assert len(x) == 7
    assert len(y) == 7


def test_build_sequences_empty_when_not_enough_data() -> None:
    x, y = build_supervised_sequences(np.arange(5), lookback=4, horizon=2)
    assert x.size == 0
    assert y.size == 0


def test_build_sequences_shapes() -> None:
    x, y = build_supervised_sequences(np.arange(20), lookback=5, horizon=3)
    assert x.shape[1] == 5
    assert y.shape[1] == 3


def test_split_sequences_basic_sizes() -> None:
    x = np.arange(100).reshape(20, 5)
    y = np.arange(20).reshape(20, 1)
    splits = split_sequences(x, y, train_ratio=0.6, val_ratio=0.2)
    assert len(splits.x_train) == 12
    assert len(splits.x_val) == 4
    assert len(splits.x_test) == 4


def test_split_sequences_non_empty_train() -> None:
    x = np.arange(10).reshape(2, 5)
    y = np.arange(2).reshape(2, 1)
    splits = split_sequences(x, y, train_ratio=0.1, val_ratio=0.1)
    assert len(splits.x_train) >= 1


def test_to_hourly_flow_aggregates_quarter_hour() -> None:
    df = pd.DataFrame(
        {
            "site_id": [1, 1, 1, 1],
            "timestamp": pd.to_datetime(
                [
                    "2006-10-01 00:00:00",
                    "2006-10-01 00:15:00",
                    "2006-10-01 00:30:00",
                    "2006-10-01 00:45:00",
                ]
            ),
            "flow": [10, 20, 30, 40],
        }
    )
    out = to_hourly_flow(df)
    assert len(out) == 1
    assert float(out.iloc[0]["flow"]) == 100.0


def test_latest_lookback_by_site_returns_last_window() -> None:
    df = pd.DataFrame(
        {
            "site_id": [1] * 5 + [2] * 5,
            "timestamp": pd.date_range("2006-10-01", periods=10, freq="h"),
            "flow": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        }
    )
    got = latest_lookback_by_site(df, lookback=3)
    assert got[1].tolist() == [3.0, 4.0, 5.0]
    assert got[2].tolist() == [30.0, 40.0, 50.0]
