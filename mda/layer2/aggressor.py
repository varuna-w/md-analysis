"""
Layer 2 — Aggressor-side analysis (graphs T2, T4).

T2: Buy/sell imbalance — rolling 1-min buy ratio, zero-centred.
T4: Tick frequency — unique prices/sec rolling window.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_aggressor_imbalance(
    df: pd.DataFrame,
    window: str = "1min",
) -> pd.DataFrame:
    """
    Compute rolling buy-side aggressor ratio per exchange.

    Parameters
    ----------
    df : trades DataFrame with ``exchange``, ``receive_ts_us``, ``taker_side_buy``.
    window : rolling window size (default ``'1min'``).

    Returns
    -------
    DataFrame with columns: exchange, time_bin, buy_ratio, imbalance.
    ``imbalance`` = buy_ratio - 0.5 (zero-centred).
    """
    dt = pd.to_datetime(df["receive_ts_us"] * 1_000, unit="ns", utc=True)
    df = df.copy()
    df["_dt"] = dt
    df["_buy"] = df["taker_side_buy"].astype(int)

    records = []
    for exchange, grp in df.groupby("exchange"):
        grp = grp.set_index("_dt").sort_index()
        resampled = grp["_buy"].resample(window).agg(["sum", "count"])
        resampled["buy_ratio"] = resampled["sum"] / resampled["count"].clip(lower=1)
        resampled["imbalance"] = resampled["buy_ratio"] - 0.5
        resampled = resampled.reset_index()
        resampled.insert(0, "exchange", exchange)
        resampled = resampled.rename(columns={"_dt": "time_bin"})
        records.append(resampled[["exchange", "time_bin", "buy_ratio", "imbalance"]])

    if not records:
        return pd.DataFrame(columns=["exchange", "time_bin", "buy_ratio", "imbalance"])
    return pd.concat(records, ignore_index=True)


def compute_overall_buy_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Overall buy ratio and 95% CI per exchange."""
    records = []
    for exchange, grp in df.groupby("exchange"):
        buys = grp["taker_side_buy"].sum()
        total = len(grp)
        ratio = buys / max(total, 1)
        # Wilson 95% CI
        z = 1.96
        n = total
        p = ratio
        margin = z * np.sqrt(p * (1 - p) / max(n, 1))
        records.append({
            "exchange": exchange,
            "buy_ratio": round(float(ratio), 4),
            "ci_low": round(float(max(0, ratio - margin)), 4),
            "ci_high": round(float(min(1, ratio + margin)), 4),
            "n_trades": int(total),
        })
    return pd.DataFrame(records)


def compute_tick_frequency(
    df: pd.DataFrame,
    window: str = "1s",
) -> pd.DataFrame:
    """
    Compute unique prices per time window per exchange (proxy for tick activity).

    Returns DataFrame with columns: exchange, time_bin, unique_prices.
    """
    dt = pd.to_datetime(df["receive_ts_us"] * 1_000, unit="ns", utc=True)
    df = df.copy()
    df["_dt"] = dt

    records = []
    for exchange, grp in df.groupby("exchange"):
        grp = grp.set_index("_dt").sort_index()
        tick_freq = (
            grp["price"].resample(window).nunique().reset_index()
        )
        tick_freq.columns = ["time_bin", "unique_prices"]
        tick_freq.insert(0, "exchange", exchange)
        records.append(tick_freq)

    if not records:
        return pd.DataFrame(columns=["exchange", "time_bin", "unique_prices"])
    return pd.concat(records, ignore_index=True)
