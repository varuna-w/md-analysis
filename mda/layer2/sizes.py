"""
Layer 2 — Trade size and notional distributions (graph T1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_size_distribution(
    df: pd.DataFrame,
    col: str = "qty_f",
) -> dict:
    """
    Fit a log-normal distribution to ``col`` and return distribution statistics.

    Parameters
    ----------
    df : trades DataFrame with ``qty_f`` and ``notional_usd`` columns.
    col : column to analyse (default ``qty_f``).

    Returns
    -------
    dict with keys:
      - per_exchange: DataFrame with exchange, p50, p95, p99, mean, std,
        lognorm_mu, lognorm_sigma, ks_stat, ks_pval
      - combined: same but for all exchanges pooled
    """
    records = []
    for exchange, grp in df.groupby("exchange"):
        vals = grp[col].dropna()
        vals = vals[vals > 0]
        if len(vals) < 10:
            continue
        # Log-normal fit
        shape, loc, scale = stats.lognorm.fit(vals, floc=0)
        mu = np.log(scale)
        sigma = shape
        ks_stat, ks_pval = stats.kstest(vals, "lognorm", args=(shape, loc, scale))
        records.append({
            "exchange": exchange,
            "n": len(vals),
            "p50": float(vals.quantile(0.50)),
            "p95": float(vals.quantile(0.95)),
            "p99": float(vals.quantile(0.99)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "lognorm_mu": float(mu),
            "lognorm_sigma": float(sigma),
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval),
        })
    return {"per_exchange": pd.DataFrame(records)}


def compute_notional_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute notional USD distribution per exchange.
    Notional = price_f * qty_f.
    """
    records = []
    for exchange, grp in df.groupby("exchange"):
        n = grp["notional_usd"].dropna()
        n = n[n > 0]
        if n.empty:
            continue
        records.append({
            "exchange": exchange,
            "n": len(n),
            "p50_usd": float(n.quantile(0.50)),
            "p95_usd": float(n.quantile(0.95)),
            "p99_usd": float(n.quantile(0.99)),
            "mean_usd": float(n.mean()),
        })
    return pd.DataFrame(records)
