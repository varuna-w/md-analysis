"""
Layer 4 — Orderbook update rates by depth rank (graphs O1, O3, O4).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_update_rate_by_depth(
    ob: pd.DataFrame,
    freq: str = "1s",
) -> pd.DataFrame:
    """
    Compute orderbook update rate broken down by depth level.

    Parameters
    ----------
    ob : orderbook DataFrame with ``exchange``, ``level``, ``received_at``.
    freq : resample frequency.

    Returns
    -------
    DataFrame: exchange, level, avg_updates_per_sec.
    """
    if "receive_ts_us" not in ob.columns:
        ob = ob.copy()
        ob["receive_ts_us"] = (
            pd.to_datetime(ob["received_at"], utc=True, format="mixed")
            .astype("int64") // 1_000
        )

    freq_seconds = pd.tseries.frequencies.to_offset(freq).nanos / 1e9  # type: ignore
    dt = pd.to_datetime(ob["receive_ts_us"] * 1_000, unit="ns", utc=True)
    ob = ob.copy()
    ob["_dt"] = dt

    records = []
    for (exchange, level), grp in ob.groupby(["exchange", "level"]):
        grp = grp.set_index("_dt").sort_index()
        rate = grp.resample(freq).size() / freq_seconds
        records.append({
            "exchange": exchange,
            "level": int(level),
            "avg_updates_per_sec": float(rate.mean()),
            "p99_updates_per_sec": float(rate.quantile(0.99)),
        })
    return pd.DataFrame(records).sort_values(["exchange", "level"])


def compute_delta_compression_ratio(
    ob: pd.DataFrame,
    ob_updates: pd.DataFrame,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Compute delta compression ratio over time (graph O3).

    Compression ratio = updates / snapshots (within each time window).
    A ratio > 1 means deltas are more numerous; < 1 means snapshots dominate.

    Returns DataFrame: exchange, time_bin, delta_ratio.
    """
    def _resample_count(df, _freq):
        dt = pd.to_datetime(df["received_at"], utc=True, format="mixed")
        df = df.copy()
        df["_dt"] = dt
        return df.groupby("exchange").apply(
            lambda g: g.set_index("_dt").resample(_freq).size(),
            include_groups=False,
        ).reset_index().rename(columns={0: "count"})

    snap_counts = _resample_count(ob, freq)
    snap_counts.columns = ["exchange", "time_bin", "snap_count"]
    upd_counts = _resample_count(ob_updates, freq)
    upd_counts.columns = ["exchange", "time_bin", "upd_count"]

    merged = snap_counts.merge(upd_counts, on=["exchange", "time_bin"], how="outer").fillna(0)
    merged["delta_ratio"] = merged["upd_count"] / merged["snap_count"].clip(lower=1)
    return merged[["exchange", "time_bin", "delta_ratio"]]
