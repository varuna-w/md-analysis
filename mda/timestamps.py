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


# ── public utilities ──────────────────────────────────────────────────────────

def ts_us_to_dt(ts_us: pd.Series) -> pd.Series:
    """Convert int64 µs-since-epoch Series to datetime64[ns, UTC]."""
    return pd.to_datetime(ts_us * 1_000, unit="ns", utc=True)


def freq_to_seconds(freq: str) -> float:
    """Convert a pandas frequency string (e.g. '1min', '1s') to seconds."""
    return pd.tseries.frequencies.to_offset(freq).nanos / 1e9  # type: ignore


def compute_gaps(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Compute per-exchange inter-event gaps from any µs timestamp column.

    Parameters
    ----------
    df : DataFrame with ``exchange`` and ``ts_col`` columns.
    ts_col : name of the int64 µs timestamp column.

    Returns
    -------
    DataFrame with columns: exchange, gap_us.
    """
    parts = []
    for exchange, grp in df.groupby("exchange"):
        gaps = grp[ts_col].sort_values().diff().dropna().values
        parts.append(pd.DataFrame({"exchange": exchange, "gap_us": gaps}))
    if not parts:
        return pd.DataFrame(columns=["exchange", "gap_us"])
    return pd.concat(parts, ignore_index=True)


def add_ts_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived timestamp columns to a trades or orderbook DataFrame.

    Mutates ``df`` in-place and returns it.

    New columns added:
    - ``exchange_ts_us``: int64 µs since Unix epoch, from ``timestamp`` ISO string.
      Maximum precision — µs for Kraken/Coinbase/Delta, ms-quantised for Binance.
    - ``exchange_ts_dt``: datetime64[ns, UTC] version of ``exchange_ts_us``.
    - ``receive_ts_us``: int64 µs since Unix epoch, from ``received_at`` ISO string.
    - ``receive_ts_dt``: datetime64[ns, UTC] version of ``receive_ts_us``.
    - ``price_f``: float64 reconstructed price (``price / 10**price_scale``).
    - ``qty_f``: float64 reconstructed quantity.
    - ``notional_usd``: ``price_f * qty_f`` (proxy; do not use for absolute sizing).

    Layer functions consume ``receive_ts_dt`` / ``exchange_ts_dt`` directly so the
    µs→ns conversion is done exactly once per load rather than once per function.
    """
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        # Use datetime64[us] intermediate to avoid the ÷1000 step (zero-copy-ish)
        df["exchange_ts_us"] = dt.astype("datetime64[us]").astype("int64")
        df["exchange_ts_dt"] = dt
    if "received_at" in df.columns:
        dt = pd.to_datetime(df["received_at"], utc=True, format="mixed")
        df["receive_ts_us"] = dt.astype("datetime64[us]").astype("int64")
        df["receive_ts_dt"] = dt
    if "price" in df.columns and "price_scale" in df.columns:
        df["price_f"] = df["price"] / (10.0 ** df["price_scale"])
    if "quantity" in df.columns and "quantity_scale" in df.columns:
        df["qty_f"] = df["quantity"] / (10.0 ** df["quantity_scale"])
    if "price_f" in df.columns and "qty_f" in df.columns:
        df["notional_usd"] = df["price_f"] * df["qty_f"]
    return df


def classify_resolution(exchange_ts_us: pd.Series) -> dict[str, object]:
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
    df : DataFrame with ``receive_ts_dt`` (or ``receive_ts_us``) column.
    session : one of 'full_24h', 'peak_session', 'low_liq', 'stress'.
    btc_price : optional BTC price series indexed by receive_ts_us for stress detection.
    """
    # Use pre-computed datetime column when available; fall back to µs conversion.
    if "receive_ts_dt" in df.columns:
        dt = df["receive_ts_dt"]
    elif "receive_ts_us" in df.columns:
        dt = ts_us_to_dt(df["receive_ts_us"])
    else:
        raise ValueError("DataFrame must contain 'receive_ts_dt' or 'receive_ts_us'")

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
        price_1min = btc_price.resample("1min").last().pct_change().abs()
        stress_minutes = price_1min[price_1min > 0.02].index
        return dt.dt.floor("1min").isin(stress_minutes)
    else:
        raise ValueError(f"Unknown session: {session!r}")
