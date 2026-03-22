"""
Timestamp utilities for market data analysis.

Design principles:
- Always derive high-precision timestamps from the ISO 8601 string columns
  (`timestamp`, `received_at`), NOT from `event_time_ms` which is ms-only.
- `event_time_ms` is only used for fast coarse filtering / joining.
- Binance has ms resolution (last 3 digits of µs epoch always 000).
- Kraken / Coinbase / Delta India / Delta Global have µs precision.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal


def add_ts_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived timestamp columns to a trades or orderbook DataFrame.

    New columns added:
    - ``exchange_ts_us``: int64 µs since Unix epoch, from ``timestamp`` ISO string.
      Maximum precision — µs for Kraken/Coinbase/Delta, ms-quantised for Binance.
    - ``receive_ts_us``: int64 µs since Unix epoch, from ``received_at`` ISO string.
    - ``price_f``: float64 reconstructed price (``price / 10**price_scale``).
    - ``qty_f``: float64 reconstructed quantity.
    - ``notional_usd``: ``price_f * qty_f`` (proxy; do not use for absolute sizing).
    """
    if "timestamp" in df.columns:
        df["exchange_ts_us"] = (
            pd.to_datetime(df["timestamp"], utc=True, format="mixed")
            .astype("int64")
            // 1_000
        )
    if "received_at" in df.columns:
        df["receive_ts_us"] = (
            pd.to_datetime(df["received_at"], utc=True, format="mixed")
            .astype("int64")
            // 1_000
        )
    if "price" in df.columns and "price_scale" in df.columns:
        df["price_f"] = df["price"] / (10.0 ** df["price_scale"])
    if "quantity" in df.columns and "quantity_scale" in df.columns:
        df["qty_f"] = df["quantity"] / (10.0 ** df["quantity_scale"])
    if "price_f" in df.columns and "qty_f" in df.columns:
        df["notional_usd"] = df["price_f"] * df["qty_f"]
    return df


def classify_resolution(
    exchange_ts_us: pd.Series,
) -> dict[str, object]:
    """
    Classify the effective timestamp resolution for a single exchange feed.

    Returns a dict with keys:
    - ``min_gap_us``: minimum non-zero inter-event gap in µs
    - ``quantisation_ratio``: fraction of consecutive pairs with zero gap
    - ``resolution``: ``'ms'`` if min_gap_us >= 900 else ``'µs'``
    - ``p50_gap_us``, ``p99_gap_us``: gap distribution percentiles
    """
    sorted_ts = exchange_ts_us.sort_values()
    gaps = sorted_ts.diff().dropna()
    nonzero = gaps[gaps > 0]
    if nonzero.empty:
        return {
            "min_gap_us": 0,
            "quantisation_ratio": 1.0,
            "resolution": "unknown",
            "p50_gap_us": 0,
            "p99_gap_us": 0,
        }
    min_gap = float(nonzero.min())
    return {
        "min_gap_us": min_gap,
        "quantisation_ratio": float((gaps == 0).sum() / max(len(gaps), 1)),
        "resolution": "ms" if min_gap >= 900 else "µs",
        "p50_gap_us": float(nonzero.quantile(0.50)),
        "p99_gap_us": float(nonzero.quantile(0.99)),
    }


def session_mask(
    df: pd.DataFrame,
    session: Literal["full_24h", "peak_session", "low_liq", "stress"] = "full_24h",
    btc_price: pd.Series | None = None,
) -> pd.Series:
    """
    Return a boolean mask for one of the four standard session windows.

    Parameters
    ----------
    df : DataFrame with ``receive_ts_us`` column (int64 µs).
    session : one of 'full_24h', 'peak_session', 'low_liq', 'stress'.
    btc_price : optional BTC price series indexed by receive_ts_us for stress detection.
    """
    if "receive_ts_us" not in df.columns:
        raise ValueError("DataFrame must contain 'receive_ts_us'")

    dt = pd.to_datetime(df["receive_ts_us"] * 1_000, unit="ns", utc=True)
    hour = dt.dt.hour

    if session == "full_24h":
        return pd.Series(True, index=df.index)
    elif session == "peak_session":
        return (hour >= 8) & (hour < 16)
    elif session == "low_liq":
        return (hour >= 2) & (hour < 6)
    elif session == "stress":
        if btc_price is None:
            raise ValueError("btc_price required for stress session")
        # 1-minute resampled price change > 2%
        price_1min = btc_price.resample("1min").last().pct_change().abs()
        stress_minutes = price_1min[price_1min > 0.02].index
        # Map back: check if each row's minute falls in a stress window
        minute_floor = dt.dt.floor("1min")
        return minute_floor.isin(stress_minutes)
    else:
        raise ValueError(f"Unknown session: {session!r}")
