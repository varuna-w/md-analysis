"""
Centralised data loader for the mda package.

All functions return pandas DataFrames.  Loading is always lazy-first:
  - pyarrow.dataset is used for predicate pushdown (filter before materialise).
  - Never call ds.to_table() without at least an exchange filter.
  - For orderbook/orderbook_updates, also filter by time range.

Memory budget on r6i.2xlarge (64 GB):
  trades          ~2–5 GB/day   → load 1 exchange at a time, or short time ranges
  orderbook       ~40–80 GB/day → load 1-hour slices; use Polars lazy for O5
  orderbook_updates ~3–10 GB/day → filter exchange + symbol
  latency         <10 MB        → load all at once
"""

from __future__ import annotations

import os
import tracemalloc
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

from .timestamps import add_ts_columns


# ── helpers ────────────────────────────────────────────────────────────────────

def _ds_filter(
    exchanges: list[str] | None,
    symbols: list[str] | None,
    start_dt: str | None,
    end_dt: str | None,
    ts_col: str = "event_time_ms",
) -> pa.compute.Expression | None:
    """Build a pyarrow filter expression for common push-down predicates."""
    filters: list[pa.compute.Expression] = []

    if exchanges:
        filters.append(pc.field("exchange").isin(exchanges))
    if symbols:
        filters.append(pc.field("symbol").isin(symbols))
    if start_dt:
        start_ms = int(pd.Timestamp(start_dt).timestamp() * 1_000)
        filters.append(pc.field(ts_col) >= start_ms)
    if end_dt:
        end_ms = int(pd.Timestamp(end_dt).timestamp() * 1_000)
        filters.append(pc.field(ts_col) <= end_ms)

    if not filters:
        return None
    result = filters[0]
    for f in filters[1:]:
        result = result & f
    return result


def _load_table(
    path: str,
    filt,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a parquet dataset directory with optional filter and column selection."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Parquet directory not found: {path}")
    dataset = ds.dataset(path, format="parquet")
    kwargs: dict = {}
    if filt is not None:
        kwargs["filter"] = filt
    if columns:
        kwargs["columns"] = columns
    return dataset.to_table(**kwargs).to_pandas()


# ── public API ─────────────────────────────────────────────────────────────────

def load_trades(
    data_dir: str,
    exchanges: list[str] | None = None,
    symbols: list[str] | None = None,
    start_dt: str | None = None,
    end_dt: str | None = None,
    add_ts_cols: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the trades parquet table.

    Parameters
    ----------
    data_dir : root of parquet data, e.g. ``/data/parquet``.
    exchanges : list of exchange names to include (None = all).
    symbols : list of symbols to include (None = all).
    start_dt / end_dt : ISO 8601 UTC strings for time range filtering.
    add_ts_cols : if True, add ``exchange_ts_us``, ``receive_ts_us``,
                  ``price_f``, ``qty_f``, ``notional_usd``.
    columns : explicit column selection (None = all).

    Returns
    -------
    pandas DataFrame sorted by ``receive_ts_us`` (if ts cols added) else unsorted.
    """
    path = os.path.join(data_dir, "trades")
    filt = _ds_filter(exchanges, symbols, start_dt, end_dt, "event_time_ms")
    df = _load_table(path, filt, columns)
    if add_ts_cols:
        df = add_ts_columns(df)
        df = df.sort_values("receive_ts_us").reset_index(drop=True)
    return df


def load_orderbook(
    data_dir: str,
    exchanges: list[str] | None = None,
    symbols: list[str] | None = None,
    start_dt: str | None = None,
    end_dt: str | None = None,
    add_ts_cols: bool = True,
    bbo_only: bool = False,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the orderbook parquet table (flattened — one row per price level).

    Parameters
    ----------
    bbo_only : if True, only load level == 0 rows (best bid/ask). Drastically
               reduces memory usage for spread/BBO analysis.
    """
    path = os.path.join(data_dir, "orderbook")
    filt = _ds_filter(exchanges, symbols, start_dt, end_dt, "event_time_ms")
    if bbo_only:
        level_filter = pc.field("level") == 0
        filt = filt & level_filter if filt is not None else level_filter
    df = _load_table(path, filt, columns)
    if add_ts_cols:
        df = add_ts_columns(df)
    return df


def load_orderbook_updates(
    data_dir: str,
    exchanges: list[str] | None = None,
    symbols: list[str] | None = None,
    start_dt: str | None = None,
    end_dt: str | None = None,
    add_ts_cols: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the orderbook_updates parquet table (incremental delta updates)."""
    path = os.path.join(data_dir, "orderbook_updates")
    # orderbook_updates has no event_time_ms — filter by received_at string comparison
    filt = _ds_filter(exchanges, symbols, None, None)  # exchange/symbol only
    df = _load_table(path, filt, columns)
    if start_dt or end_dt:
        if "received_at" in df.columns:
            dt_col = pd.to_datetime(df["received_at"], utc=True, format="mixed")
            if start_dt:
                df = df[dt_col >= pd.Timestamp(start_dt, tz="UTC")]
            if end_dt:
                df = df[dt_col <= pd.Timestamp(end_dt, tz="UTC")]
    if add_ts_cols:
        df = add_ts_columns(df)
    return df


def load_latency(
    data_dir: str,
    exchanges: list[str] | None = None,
    start_dt: str | None = None,
    end_dt: str | None = None,
) -> pd.DataFrame:
    """Load the latency table (<10 MB — always loads all at once)."""
    path = os.path.join(data_dir, "latency")
    filt = _ds_filter(exchanges, None, None, None)
    df = _load_table(path, filt)
    if "timestamp" in df.columns:
        df["ts_dt"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        if start_dt:
            df = df[df["ts_dt"] >= pd.Timestamp(start_dt, tz="UTC")]
        if end_dt:
            df = df[df["ts_dt"] <= pd.Timestamp(end_dt, tz="UTC")]
    return df


def memory_profile(data_dir: str, exchanges: list[str] | None = None) -> dict:
    """
    Sample memory usage for loading one exchange's trades (1-hour window).
    Returns {'peak_mb': float, 'current_mb': float}.
    """
    tracemalloc.start()
    try:
        now = pd.Timestamp.utcnow()
        start = (now - pd.Timedelta(hours=1)).isoformat()
        end = now.isoformat()
        load_trades(data_dir, exchanges=exchanges or ["binance"],
                    start_dt=start, end_dt=end)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"peak_mb": peak / 1e6}
