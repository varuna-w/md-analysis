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

    Mutates ``df`` in-place (adds ``latency_ms``, ``latency_corrected_ms``)
    and returns it.

    Parameters
    ----------
    df : trades DataFrame with ``exchange_ts_us`` and ``receive_ts_us``.
    """
    df["latency_ms"] = (df["receive_ts_us"] - df["exchange_ts_us"]) / 1_000.0
    p1 = df.groupby("exchange")["latency_ms"].quantile(0.01)
    df["latency_corrected_ms"] = df["latency_ms"] - df["exchange"].map(p1)
    return df


def latency_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute latency distribution statistics per exchange.

    Calls ``compute_feed_latency`` if ``latency_ms`` is not already present.

    Returns DataFrame with columns:
      exchange, p1_ms, p25_ms, p50_ms, p75_ms, p95_ms, p99_ms, p99_9_ms,
      max_ms, negative_pct.
    """
    if "latency_ms" not in df.columns:
        compute_feed_latency(df)

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

    Calls ``compute_feed_latency`` if ``latency_ms`` is not already present.

    Returns DataFrame: exchange, time_bin, p50_latency_ms.
    """
    if "latency_ms" not in df.columns:
        compute_feed_latency(df)

    records = []
    for exchange, grp in df.groupby("exchange"):
        ts = (
            grp.set_index("receive_ts_dt")
            .sort_index()["latency_ms"]
            .resample(freq)
            .median()
            .reset_index()
        )
        ts.columns = ["time_bin", "p50_latency_ms"]
        ts.insert(0, "exchange", exchange)
        records.append(ts)

    if not records:
        return pd.DataFrame(columns=["exchange", "time_bin", "p50_latency_ms"])
    return pd.concat(records, ignore_index=True)
