"""
Layer 3 — Monotonicity and out-of-order analysis (graph E7).

Measures the frequency and magnitude of backward timestamp jumps
in the exchange feed.  This informs the sequencer/resequencing window
that the matching engine must budget for.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_out_of_order(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect out-of-order events per (exchange, symbol).

    An event is out-of-order when ``exchange_ts_us[i] < exchange_ts_us[i-1]``
    after sorting by receive time (``receive_ts_us``).

    Parameters
    ----------
    df : trades DataFrame with ``exchange_ts_us``, ``receive_ts_us``,
         ``exchange``, ``symbol``.

    Returns
    -------
    (summary, events)

    ``summary``: one row per exchange with:
      exchange, n_total, n_oo, oo_rate_pct, p50_lag_us, p99_lag_us, max_lag_us.

    ``events``: all out-of-order rows with additional column ``oo_lag_us``
      (magnitude of backward jump).
    """
    summary_records = []
    event_parts = []

    for (exchange, symbol), grp in df.groupby(["exchange", "symbol"]):
        # Sort by receive time — this is the order we'd process them
        grp = grp.sort_values("receive_ts_us").reset_index(drop=True)
        grp["oo_lag_us"] = (
            grp["exchange_ts_us"].shift(1) - grp["exchange_ts_us"]
        ).clip(lower=0)
        oo_mask = grp["oo_lag_us"] > 0
        oo = grp[oo_mask].copy()
        event_parts.append(oo)

    if not event_parts:
        empty = pd.DataFrame(columns=[
            "exchange", "n_total", "n_oo", "oo_rate_pct",
            "p50_lag_us", "p99_lag_us", "max_lag_us",
        ])
        return empty, pd.DataFrame()

    all_events = pd.concat(event_parts, ignore_index=True)

    for exchange, grp in df.groupby("exchange"):
        ev = all_events[all_events["exchange"] == exchange]
        lags = ev["oo_lag_us"] if not ev.empty else pd.Series(dtype=float)
        n_total = len(grp)
        n_oo = len(ev)
        summary_records.append({
            "exchange": exchange,
            "n_total": n_total,
            "n_oo": n_oo,
            "oo_rate_pct": round(100.0 * n_oo / max(n_total, 1), 4),
            "p50_lag_us": float(lags.quantile(0.50)) if not lags.empty else 0.0,
            "p99_lag_us": float(lags.quantile(0.99)) if not lags.empty else 0.0,
            "max_lag_us": float(lags.max()) if not lags.empty else 0.0,
        })

    return pd.DataFrame(summary_records), all_events
