"""
Layer 4 — BBO spread and depth analysis (graph O2).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_bbo(ob: pd.DataFrame) -> pd.DataFrame:
    """
    Extract BBO (best bid/offer) from flattened orderbook snapshots.

    Parameters
    ----------
    ob : orderbook DataFrame with columns:
         snapshot_id, exchange, symbol, side, price_f, qty_f, received_at.
         Must include level==0 rows only (or will be filtered here).

    Returns
    -------
    DataFrame with columns:
      snapshot_id, exchange, symbol, received_at,
      bid_price, bid_qty, ask_price, ask_qty,
      spread, spread_bps, mid_price, book_imbalance.
    """
    ob = ob[ob["level"] == 0].copy() if "level" in ob.columns else ob.copy()

    if "price_f" not in ob.columns:
        ob["price_f"] = ob["price"] / (10.0 ** ob["price_scale"])
    if "qty_f" not in ob.columns:
        ob["qty_f"] = ob["quantity"] / (10.0 ** ob["quantity_scale"])

    bids = ob[ob["side"] == "bid"][
        ["snapshot_id", "exchange", "symbol", "received_at", "price_f", "qty_f"]
    ].rename(columns={"price_f": "bid_price", "qty_f": "bid_qty"})

    asks = ob[ob["side"] == "ask"][
        ["snapshot_id", "exchange", "symbol", "received_at", "price_f", "qty_f"]
    ].rename(columns={"price_f": "ask_price", "qty_f": "ask_qty"})

    bbo = bids.merge(
        asks,
        on=["snapshot_id", "exchange", "symbol", "received_at"],
        how="inner",
    )

    bbo["spread"] = bbo["ask_price"] - bbo["bid_price"]
    bbo["mid_price"] = (bbo["bid_price"] + bbo["ask_price"]) / 2
    bbo["spread_bps"] = bbo["spread"] / bbo["mid_price"].clip(lower=1e-12) * 10_000

    # Imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty); range [-1, 1]
    total_qty = (bbo["bid_qty"] + bbo["ask_qty"]).clip(lower=1e-12)
    bbo["book_imbalance"] = (bbo["bid_qty"] - bbo["ask_qty"]) / total_qty

    return bbo


def spread_stats(bbo: pd.DataFrame) -> pd.DataFrame:
    """Spread distribution statistics per exchange."""
    records = []
    for exchange, grp in bbo.groupby("exchange"):
        s = grp["spread_bps"].dropna()
        records.append({
            "exchange": exchange,
            "p50_spread_bps": float(s.quantile(0.50)),
            "p95_spread_bps": float(s.quantile(0.95)),
            "p99_spread_bps": float(s.quantile(0.99)),
            "mean_spread_bps": float(s.mean()),
            "mean_bid_qty": float(grp["bid_qty"].mean()),
            "mean_ask_qty": float(grp["ask_qty"].mean()),
        })
    return pd.DataFrame(records)
