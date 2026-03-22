"""
Layer 3 — Feed latency analysis (graphs E5, E6).

Feed latency = receive_ts_us - exchange_ts_us.
Negative values indicate the exchange clock is ahead of the local clock
(clock drift artefact — subtract per-exchange p1 latency as floor correction).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_feed_latency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw and offset-corrected feed latency per trade.

    Adds columns:
    - ``latency_ms``: raw (receive - exchange) in milliseconds
    - ``latency_corrected_ms``: latency minus per-exchange p1 (physical floor)

    Parameters
    ----------
    df : trades DataFrame with ``exchange_ts_us`` and ``receive_ts_us``.

    Returns
    -------
    DataFrame with the two new latency columns added.
    """
    df = df.copy()
    df["latency_ms"] = (df["receive_ts_us"] - df["exchange_ts_us"]) / 1_000.0

    # Per-exchange p1 = physical minimum (network floor)
    p1 = df.groupby("exchange")["latency_ms"].quantile(0.01)
    df["latency_corrected_ms"] = df["latency_ms"] - df["exchange"].map(p1)
    return df


def latency_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute latency distribution statistics per exchange.

    Returns DataFrame with columns:
      exchange, p1, p25, p50, p75, p95, p99, p99_9, max_ms, negative_pct.
    """
    if "latency_ms" not in df.columns:
        df = compute_feed_latency(df)

    records = []
    for exchange, grp in df.groupby("exchange"):
        lat = grp["latency_ms"]
        records.append({
            "exchange": exchange,
            "p1_ms": float(lat.quantile(0.01)),
            "p25_ms": float(lat.quantile(0.25)),
            "p50_ms": float(lat.quantile(0.50)),
            "p75_ms": float(lat.quantile(0.75)),
            "p95_ms": float(lat.quantile(0.95)),
            "p99_ms": float(lat.quantile(0.99)),
            "p99_9_ms": float(lat.quantile(0.999)),
            "max_ms": float(lat.max()),
            "negative_pct": float(100.0 * (lat < 0).sum() / max(len(lat), 1)),
        })
    return pd.DataFrame(records)


def latency_drift_timeseries(
    df: pd.DataFrame,
    freq: str = "5min",
) -> pd.DataFrame:
    """
    Rolling median latency over time per exchange (graph E6).

    Returns DataFrame: exchange, time_bin, p50_latency_ms.
    """
    if "latency_ms" not in df.columns:
        df = compute_feed_latency(df)

    dt = pd.to_datetime(df["receive_ts_us"] * 1_000, unit="ns", utc=True)
    df = df.copy()
    df["_dt"] = dt

    records = []
    for exchange, grp in df.groupby("exchange"):
        grp = grp.set_index("_dt").sort_index()
        ts = grp["latency_ms"].resample(freq).median().reset_index()
        ts.columns = ["time_bin", "p50_latency_ms"]
        ts.insert(0, "exchange", exchange)
        records.append(ts)

    if not records:
        return pd.DataFrame(columns=["exchange", "time_bin", "p50_latency_ms"])
    return pd.concat(records, ignore_index=True)
