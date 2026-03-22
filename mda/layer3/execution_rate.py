"""
Layer 3 — Exchange-timestamp-derived execution rates (graph E1).

Uses ``exchange_ts_dt`` (pre-computed by add_ts_columns) rather than
``receive_ts_dt`` so the rate reflects what the exchange actually
matched, not our receive jitter.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..timestamps import freq_to_seconds


def compute_execution_rate(
    df: pd.DataFrame,
    freq: str = "1s",
) -> pd.DataFrame:
    """
    Compute executions per second from exchange timestamps.

    Parameters
    ----------
    df : trades DataFrame with ``exchange_ts_dt`` and ``exchange``.
    freq : resample frequency (default 1s).

    Returns
    -------
    DataFrame: exchange, time_bin, exec_per_sec.
    """
    fs = freq_to_seconds(freq)
    records = []
    for exchange, grp in df.groupby("exchange"):
        rate = (
            grp.set_index("exchange_ts_dt")
            .resample(freq)
            .size()
            / fs
        )
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
