"""
Layer 3 — Micro-burst detection (graphs E2, E3).

A micro-burst is a sequence of >= min_size consecutive trades where each
consecutive exchange-timestamp gap is <= gap_us_threshold (default 1 ms).

This characterises the worst-case burst the matching engine must absorb
within a single scheduling quantum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_microbursts(
    df: pd.DataFrame,
    gap_us_threshold: int = 1_000,
    min_size: int = 3,
) -> pd.DataFrame:
    """
    Detect micro-bursts per (exchange, symbol).

    Parameters
    ----------
    df : trades DataFrame with ``exchange_ts_us``, ``exchange``, ``symbol``.
    gap_us_threshold : max intra-burst gap in µs (default 1 ms).
    min_size : minimum burst size to report.

    Returns
    -------
    DataFrame with columns:
      exchange, symbol, burst_id, size, duration_us, start_ts_us, peak_trades_per_ms.
    """
    records = []
    for (exchange, symbol), grp in df.groupby(["exchange", "symbol"]):
        grp = grp.sort_values("exchange_ts_us").reset_index(drop=True)
        grp["gap_us"] = grp["exchange_ts_us"].diff()

        # A row is "in burst" if the gap from the previous row is <= threshold
        # We consider the first row of each burst as also part of it
        grp["in_burst"] = grp["gap_us"] <= gap_us_threshold
        # Assign burst IDs: each time a gap > threshold, new burst
        grp["burst_id"] = (~grp["in_burst"]).cumsum()

        for bid, burst in grp.groupby("burst_id"):
            if len(burst) < min_size:
                continue
            duration_us = max(
                burst["exchange_ts_us"].max() - burst["exchange_ts_us"].min(), 1
            )
            size = len(burst)
            peak_per_ms = size / max(duration_us / 1_000, 0.001)
            records.append({
                "exchange": exchange,
                "symbol": symbol,
                "burst_id": int(bid),
                "size": size,
                "duration_us": int(duration_us),
                "start_ts_us": int(burst["exchange_ts_us"].min()),
                "peak_trades_per_ms": float(peak_per_ms),
            })

    if not records:
        return pd.DataFrame(columns=[
            "exchange", "symbol", "burst_id", "size", "duration_us",
            "start_ts_us", "peak_trades_per_ms",
        ])
    return pd.DataFrame(records)


def microburst_stats(bursts: pd.DataFrame) -> pd.DataFrame:
    """Percentile summary of micro-burst sizes per exchange."""
    records = []
    for exchange, grp in bursts.groupby("exchange"):
        s = grp["size"]
        records.append({
            "exchange": exchange,
            "n_bursts": len(s),
            "p50_size": float(s.quantile(0.50)),
            "p95_size": float(s.quantile(0.95)),
            "p99_size": float(s.quantile(0.99)),
            "p99_9_size": float(s.quantile(0.999)),
            "max_size": int(s.max()),
            "p99_peak_per_ms": float(grp["peak_trades_per_ms"].quantile(0.99)),
        })
    return pd.DataFrame(records)


def microburst_heatmap(bursts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate peak_trades_per_ms by (hour, day-of-week) per exchange.

    Returns DataFrame: exchange, hour, dow, peak_trades_per_ms.
    """
    dt = pd.to_datetime(bursts["start_ts_us"] * 1_000, unit="ns", utc=True)
    bursts = bursts.copy()
    bursts["hour"] = dt.dt.hour
    bursts["dow"] = dt.dt.day_of_week
    return (
        bursts.groupby(["exchange", "hour", "dow"])["peak_trades_per_ms"]
        .max()
        .reset_index()
    )
