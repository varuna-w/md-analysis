"""
Layer 4 — Orderbook update rates by depth rank (graphs O1, O3, O4).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..timestamps import freq_to_seconds


def _resample_count_by_exchange(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample row counts by exchange using the pre-computed ``receive_ts_dt`` column
    (added by ``add_ts_columns``).  Falls back to parsing ``received_at`` if absent.
    """
    if "receive_ts_dt" not in df.columns:
        df = df.copy()
        df["receive_ts_dt"] = pd.to_datetime(
            df["received_at"], utc=True, format="mixed"
        )
    parts = []
    for exchange, grp in df.groupby("exchange"):
        counts = grp.set_index("receive_ts_dt").resample(freq).size()
        parts.append(pd.DataFrame({
            "exchange": exchange,
            "time_bin": counts.index,
            "count": counts.values,
        }))
    if not parts:
        return pd.DataFrame(columns=["exchange", "time_bin", "count"])
    return pd.concat(parts, ignore_index=True)


def compute_update_rate_by_depth(
    ob: pd.DataFrame,
    freq: str = "1s",
) -> pd.DataFrame:
    """
    Compute orderbook update rate broken down by depth level.

    Parameters
    ----------
    ob : orderbook DataFrame with ``exchange``, ``level``, ``receive_ts_dt``
        (or ``received_at`` as fallback).
    freq : resample frequency.

    Returns
    -------
    DataFrame: exchange, level, avg_updates_per_sec, p99_updates_per_sec.
    """
    if "receive_ts_dt" not in ob.columns:
        ob = ob.copy()
        ob["receive_ts_dt"] = pd.to_datetime(
            ob["received_at"], utc=True, format="mixed"
        )

    fs = freq_to_seconds(freq)
    records = []
    for (exchange, level), grp in ob.groupby(["exchange", "level"]):
        rate = grp.set_index("receive_ts_dt").sort_index().resample(freq).size() / fs
        records.append({
            "exchange": exchange,
            "level": int(level),
            "avg_updates_per_sec": float(rate.mean()),
            "p99_updates_per_sec": float(rate.quantile(0.99)),
        })
    if not records:
        return pd.DataFrame(columns=["exchange", "level", "avg_updates_per_sec", "p99_updates_per_sec"])
    return pd.DataFrame(records).sort_values(["exchange", "level"])


def compute_delta_compression_ratio(
    ob: pd.DataFrame,
    ob_updates: pd.DataFrame,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Compute delta compression ratio over time (graph O3).

    Compression ratio = updates / snapshots within each time window.
    > 1 means deltas outnumber snapshots; < 1 means snapshots dominate.

    Returns DataFrame: exchange, time_bin, delta_ratio.
    """
    snap_counts = _resample_count_by_exchange(ob, freq)
    snap_counts.columns = ["exchange", "time_bin", "snap_count"]
    upd_counts = _resample_count_by_exchange(ob_updates, freq)
    upd_counts.columns = ["exchange", "time_bin", "upd_count"]

    merged = snap_counts.merge(upd_counts, on=["exchange", "time_bin"], how="outer").fillna(0)
    merged["delta_ratio"] = merged["upd_count"] / merged["snap_count"].clip(lower=1)
    return merged[["exchange", "time_bin", "delta_ratio"]]
