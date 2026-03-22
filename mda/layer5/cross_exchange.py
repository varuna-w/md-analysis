"""
Layer 5 — Cross-exchange correlation (graphs X1–X3).

X1: Cross-venue lag CDF — for each exchange pair, how quickly does a
    price move on exchange A appear on exchange B?
X2: Cumulative capacity stacked bar — total throughput across exchanges.
X3: Cascade events — rate time series annotated with price cascade windows.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from itertools import combinations


def _normalise_symbol(s: pd.Series) -> pd.Series:
    """Strip separators and upper-case for fuzzy cross-exchange symbol matching."""
    return s.str.upper().str.replace("-", "", regex=False).str.replace("/", "", regex=False)


def compute_cross_venue_lag(
    df: pd.DataFrame,
    symbol_filter: str = "BTC",
    window_us: int = 50_000,
    sample_frac: float = 0.1,
) -> pd.DataFrame:
    """
    Compute cross-venue lag CDF for a given symbol prefix.

    For each pair of exchanges (A, B), find the nearest trade on B
    within ``window_us`` of each trade on A.  Report the lag distribution.

    Parameters
    ----------
    df : trades DataFrame with ``exchange``, ``exchange_ts_us``, ``symbol``.
    symbol_filter : normalised symbol prefix to match (default ``'BTC'``).
        Separators and case are ignored, so 'BTC-USD', 'BTCUSDT', 'btc/usd'
        all match the default.
    window_us : search window in µs (default 50 ms).
    sample_frac : fraction of A-side trades to sample (for performance).

    Returns
    -------
    DataFrame: exchange_a, exchange_b, lag_us (one row per matched pair).
    """
    filtered = df[_normalise_symbol(df["symbol"]).str.startswith(symbol_filter)].copy()
    filtered = filtered.sort_values("exchange_ts_us").reset_index(drop=True)

    exchanges = filtered["exchange"].unique()
    if len(exchanges) < 2:
        return pd.DataFrame(columns=["exchange_a", "exchange_b", "lag_us"])

    parts = []
    for ex_a, ex_b in combinations(exchanges, 2):
        ts_a = filtered.loc[filtered["exchange"] == ex_a, "exchange_ts_us"].values
        ts_b = filtered.loc[filtered["exchange"] == ex_b, "exchange_ts_us"].values

        if len(ts_a) == 0 or len(ts_b) == 0:
            continue

        # Optional random sample of A-side for performance
        if sample_frac < 1.0:
            n_sample = max(100, int(len(ts_a) * sample_frac))
            ts_a = np.sort(ts_a[np.random.choice(len(ts_a), min(n_sample, len(ts_a)), replace=False)])
        else:
            ts_a = np.sort(ts_a)

        ts_b = np.sort(ts_b)

        # Vectorised: find insertion points, then check both neighbours at once
        ins = np.searchsorted(ts_b, ts_a)
        left_idx = np.clip(ins - 1, 0, len(ts_b) - 1)
        right_idx = np.clip(ins,     0, len(ts_b) - 1)

        left_lag  = np.abs(ts_b[left_idx].astype(np.int64)  - ts_a.astype(np.int64))
        right_lag = np.abs(ts_b[right_idx].astype(np.int64) - ts_a.astype(np.int64))
        min_lag = np.minimum(left_lag, right_lag)

        valid = min_lag <= window_us
        if valid.any():
            parts.append(pd.DataFrame({
                "exchange_a": ex_a,
                "exchange_b": ex_b,
                "lag_us": min_lag[valid],
            }))

    if not parts:
        return pd.DataFrame(columns=["exchange_a", "exchange_b", "lag_us"])
    return pd.concat(parts, ignore_index=True)


def compute_cumulative_capacity(
    trade_rate_pcts: pd.DataFrame,
    ob_update_rate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a cumulative capacity summary table for graph X2.

    Returns DataFrame: exchange, trade_p99_per_sec, ob_update_p99_per_sec,
                       total_p99_per_sec.
    """
    records = []
    for exchange, grp in trade_rate_pcts.groupby("exchange"):
        trade_p99 = float(grp["p99"].iloc[0]) if "p99" in grp.columns else 0.0
        ob_p99 = 0.0
        if ob_update_rate is not None:
            ob_row = ob_update_rate[ob_update_rate["exchange"] == exchange]
            if not ob_row.empty and "p99_updates_per_sec" in ob_row.columns:
                ob_p99 = float(ob_row["p99_updates_per_sec"].mean())
        records.append({
            "exchange": exchange,
            "trade_p99_per_sec": trade_p99,
            "ob_update_p99_per_sec": ob_p99,
            "total_p99_per_sec": trade_p99 + ob_p99,
        })
    return pd.DataFrame(records).sort_values("total_p99_per_sec", ascending=False)


def detect_cascade_events(
    df: pd.DataFrame,
    symbol_filter: str = "BTC",
    pct_threshold: float = 0.02,
    resample_freq: str = "1min",
) -> pd.DatetimeIndex:
    """
    Detect cascade (stress) event windows where the filtered symbol's price
    moves > ``pct_threshold`` within a single ``resample_freq`` window.

    Returns DatetimeIndex of cascade window starts.
    """
    btc = df[_normalise_symbol(df["symbol"]).str.startswith(symbol_filter)].copy()
    if btc.empty:
        return pd.DatetimeIndex([])

    btc = btc.set_index("receive_ts_dt").sort_index()
    price_1min = btc["price_f"].resample(resample_freq).median().pct_change().abs()
    return price_1min[price_1min > pct_threshold].index
