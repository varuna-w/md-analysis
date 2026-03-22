"""
Layer 3 — Timestamp gap distribution and resolution classification (graph E4).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..timestamps import classify_resolution, compute_gaps


def compute_timestamp_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-exchange inter-event gap distribution from exchange_ts_us.

    Returns DataFrame with columns: exchange, gap_us (one row per pair).
    """
    return compute_gaps(df, "exchange_ts_us")


def resolution_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify timestamp resolution per exchange.

    Returns DataFrame with one row per exchange:
      exchange, resolution, min_gap_us, quantisation_ratio, p50_gap_us, p99_gap_us.
    """
    records = []
    for exchange, grp in df.groupby("exchange"):
        info = classify_resolution(grp["exchange_ts_us"])
        info["exchange"] = exchange
        records.append(info)
    return pd.DataFrame(records)[
        ["exchange", "resolution", "min_gap_us", "quantisation_ratio",
         "p50_gap_us", "p99_gap_us"]
    ]
