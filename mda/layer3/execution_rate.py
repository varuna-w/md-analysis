"""
Layer 3 — Exchange-timestamp-derived execution rates (graph E1).

Uses ``exchange_ts_us`` (from ISO 8601 string parse) rather than
``receive_ts_us`` so the rate reflects what the exchange actually
matched, not our receive jitter.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_execution_rate(
    df: pd.DataFrame,
    freq: str = "1s",
) -> pd.DataFrame:
    """
    Compute executions per second from exchange timestamps.

    Parameters
    ----------
    df : trades DataFrame with ``exchange_ts_us`` and ``exchange``.
    freq : resample frequency (default 1s).

    Returns
    -------
    DataFrame: exchange, time_bin, exec_per_sec.
    """
    freq_seconds = pd.tseries.frequencies.to_offset(freq).nanos / 1e9  # type: ignore
    dt = pd.to_datetime(df["exchange_ts_us"] * 1_000, unit="ns", utc=True)
    df = df.copy()
    df["_dt"] = dt

    records = []
    for exchange, grp in df.groupby("exchange"):
        grp = grp.set_index("_dt").sort_index()
        rate = grp.resample(freq).size() / freq_seconds
        rate = rate.reset_index()
        rate.columns = ["time_bin", "exec_per_sec"]
        rate.insert(0, "exchange", exchange)
        records.append(rate)

    if not records:
        return pd.DataFrame(columns=["exchange", "time_bin", "exec_per_sec"])
    return pd.concat(records, ignore_index=True)


def execution_rate_stats(rate_df: pd.DataFrame) -> pd.DataFrame:
    """Percentile summary of execution rates per exchange."""
    records = []
    for exchange, grp in rate_df.groupby("exchange"):
        r = grp["exec_per_sec"]
        records.append({
            "exchange": exchange,
            "p50": float(r.quantile(0.50)),
            "p95": float(r.quantile(0.95)),
            "p99": float(r.quantile(0.99)),
            "p99_9": float(r.quantile(0.999)),
            "peak": float(r.max()),
        })
    return pd.DataFrame(records)
