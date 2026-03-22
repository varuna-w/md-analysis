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


def compute_cross_venue_lag(
    df: pd.DataFrame,
    symbol_filter: str = "BTCUSDT",
    window_us: int = 50_000,  # 50 ms
    sample_frac: float = 0.1,  # sample 10% for performance
) -> pd.DataFrame:
    """
    Compute cross-venue lag CDF for a given symbol.

    For each pair of exchanges (A, B), find the nearest trade on B
    within window_us of each trade on A.  Report the lag distribution.

    Parameters
    ----------
    df : trades DataFrame with ``exchange``, ``exchange_ts_us``, ``symbol``.
    symbol_filter : symbol to analyse (e.g. 'BTCUSDT', 'BTC-USD', 'BTC/USD').
    window_us : search window in µs (default 50 ms).
    sample_frac : fraction of A-side trades to sample (for performance).

    Returns
    -------
    DataFrame: exchange_a, exchange_b, lag_us (one row per matched pair).
    """
    # Filter to BTC trades across all exchanges (symbol names vary by exchange)
    btc = df[df["symbol"].str.upper().str.replace("-", "").str.replace("/", "")
              .str.startswith("BTC")].copy()
    btc = btc.sort_values("exchange_ts_us").reset_index(drop=True)

    exchanges = btc["exchange"].unique()
    if len(exchanges) < 2:
        return pd.DataFrame(columns=["exchange_a", "exchange_b", "lag_us"])

    records = []
    for ex_a, ex_b in combinations(exchanges, 2):
        ts_a = btc[btc["exchange"] == ex_a]["exchange_ts_us"].values
        ts_b = btc[btc["exchange"] == ex_b]["exchange_ts_us"].values

        if len(ts_a) == 0 or len(ts_b) == 0:
            continue

        # Sample for performance
        if sample_frac < 1.0:
            n_sample = max(100, int(len(ts_a) * sample_frac))
            idx = np.random.choice(len(ts_a), min(n_sample, len(ts_a)), replace=False)
            ts_a_sample = np.sort(ts_a[idx])
        else:
            ts_a_sample = np.sort(ts_a)

        ts_b_sorted = np.sort(ts_b)

        # For each ts in A, find nearest ts in B using searchsorted
        ins = np.searchsorted(ts_b_sorted, ts_a_sample)

        for i, (ts, ins_idx) in enumerate(zip(ts_a_sample, ins)):
            candidates = []
            for j in [ins_idx - 1, ins_idx]:
                if 0 <= j < len(ts_b_sorted):
                    lag = abs(int(ts_b_sorted[j]) - int(ts))
                    if lag <= window_us:
                        candidates.append(lag)
            if candidates:
                records.append({
                    "exchange_a": ex_a,
                    "exchange_b": ex_b,
                    "lag_us": min(candidates),
                })

    if not records:
        return pd.DataFrame(columns=["exchange_a", "exchange_b", "lag_us"])
    return pd.DataFrame(records)


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
    symbol: str = "BTCUSDT",
    pct_threshold: float = 0.02,
    resample_freq: str = "1min",
) -> pd.DatetimeIndex:
    """
    Detect cascade (stress) event windows where BTC price moves > pct_threshold.

    Returns DatetimeIndex of cascade minute starts.
    """
    btc = df[
        df["symbol"].str.upper().str.replace("-", "").str.replace("/", "").str.startswith("BTC")
    ].copy()
    if btc.empty:
        return pd.DatetimeIndex([])

    btc["_dt"] = pd.to_datetime(btc["receive_ts_us"] * 1_000, unit="ns", utc=True)
    btc = btc.set_index("_dt").sort_index()

    # Use median price across exchanges to avoid single-exchange artefacts
    price_1min = btc["price_f"].resample(resample_freq).median().pct_change().abs()
    cascade_windows = price_1min[price_1min > pct_threshold].index
    return cascade_windows
