"""
Layer 2 — Volley detection (graph T3).

A volley is a group of trades from the same exchange with consecutive
exchange-timestamp gaps <= gap_threshold_us (default 5 ms = 5_000 µs).

Note for Binance: ms resolution means intra-volley gaps will be multiples
of 1 ms.  Volleys are still detectable (5 ms threshold > 1 ms tick) but
the intra-volley gap distribution is coarser.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_volleys(
    df: pd.DataFrame,
    gap_threshold_us: int = 5_000,
    min_size: int = 2,
    return_annotated: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect volleys per (exchange, symbol).

    Parameters
    ----------
    df : trades DataFrame with ``exchange_ts_us``, ``exchange``, ``symbol``.
    gap_threshold_us : max gap in µs between consecutive trades in a volley.
    min_size : minimum volley size to report (default 2).
    return_annotated : if True, also return the full per-row annotated DataFrame
        (adds gap_us, new_volley, volley_id columns).  Defaults to False since
        most callers only need the volley_stats summary.

    Returns
    -------
    (annotated_df, volley_stats)

    ``annotated_df``: original df with added columns gap_us / new_volley / volley_id,
        or an empty DataFrame when ``return_annotated=False``.

    ``volley_stats``: one row per volley with columns:
        exchange, symbol, volley_id, size, start_ts_us, end_ts_us, duration_us.
    """
    annotated_parts = []
    stats_parts = []

    for (exchange, symbol), grp in df.groupby(["exchange", "symbol"]):
        grp = grp.sort_values("exchange_ts_us").copy()
        grp["gap_us"] = grp["exchange_ts_us"].diff()
        grp["new_volley"] = (grp["gap_us"] > gap_threshold_us) | grp["gap_us"].isna()
        grp["volley_id"] = grp["new_volley"].cumsum()

        vstats = (
            grp.groupby("volley_id")
            .agg(
                size=("exchange_ts_us", "count"),
                start_ts_us=("exchange_ts_us", "min"),
                end_ts_us=("exchange_ts_us", "max"),
            )
            .reset_index()
        )
        vstats["duration_us"] = vstats["end_ts_us"] - vstats["start_ts_us"]
        vstats = vstats[vstats["size"] >= min_size]
        vstats.insert(0, "exchange", exchange)
        vstats.insert(1, "symbol", symbol)
        stats_parts.append(vstats)

        if return_annotated:
            annotated_parts.append(grp)

    annotated = (
        pd.concat(annotated_parts, ignore_index=True) if annotated_parts
        else pd.DataFrame()
    )
    vstats_all = (
        pd.concat(stats_parts, ignore_index=True) if stats_parts
        else pd.DataFrame()
    )
    return annotated, vstats_all


def volley_size_distribution(volley_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-exchange volley size percentiles.

    Returns DataFrame with columns: exchange, n_volleys, p50, p95, p99, p99_9, max.
    """
    records = []
    for exchange, grp in volley_stats.groupby("exchange"):
        sizes = grp["size"]
        records.append({
            "exchange": exchange,
            "n_volleys": len(sizes),
            "p50": float(sizes.quantile(0.50)),
            "p95": float(sizes.quantile(0.95)),
            "p99": float(sizes.quantile(0.99)),
            "p99_9": float(sizes.quantile(0.999)),
            "max": int(sizes.max()),
        })
    return pd.DataFrame(records)
