"""
Layer 4 — Price level lifetime analysis (graph O5).

This is the most memory-intensive analysis in the suite.
Uses Polars lazy evaluation to avoid loading all orderbook_updates at once.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_level_lifetimes_polars(
    data_dir: str,
    exchange: str,
    symbol: str,
) -> pd.DataFrame:
    """
    Compute price level lifetime for one (exchange, symbol) pair.

    Uses Polars lazy scanning to avoid loading all data into memory.
    Lifetime = last_seen_ts - first_seen_ts for each (exchange, symbol, side, price).

    Returns DataFrame: exchange, symbol, side, price, first_seen_us, last_seen_us,
                       lifetime_us.
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError("polars is required for level lifetime analysis")

    import os
    path = os.path.join(data_dir, "orderbook_updates")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"orderbook_updates directory not found: {path}")

    lf = (
        pl.scan_parquet(f"{path}/*.parquet")
        .filter(
            (pl.col("exchange") == exchange) & (pl.col("symbol") == symbol)
        )
        .with_columns([
            (
                pl.col("received_at")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
                .cast(pl.Int64) // 1_000
            ).alias("ts_us")
        ])
        .group_by(["exchange", "symbol", "side", "price"])
        .agg([
            pl.col("ts_us").min().alias("first_seen_us"),
            pl.col("ts_us").max().alias("last_seen_us"),
        ])
        .with_columns([
            (pl.col("last_seen_us") - pl.col("first_seen_us")).alias("lifetime_us")
        ])
    )

    return lf.collect().to_pandas()


def level_lifetime_stats(lifetimes: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics of level lifetimes per exchange."""
    records = []
    for exchange, grp in lifetimes.groupby("exchange"):
        lt = grp["lifetime_us"] / 1_000  # → ms
        records.append({
            "exchange": exchange,
            "n_levels": len(lt),
            "p50_lifetime_ms": float(lt.quantile(0.50)),
            "p95_lifetime_ms": float(lt.quantile(0.95)),
            "p99_lifetime_ms": float(lt.quantile(0.99)),
            "mean_lifetime_ms": float(lt.mean()),
        })
    return pd.DataFrame(records)
