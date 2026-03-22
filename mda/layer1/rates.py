"""
Layer 1 — Raw message rate characterisation.

Produces statistics for graphs R1–R5:
  R1: Rate time series (1-min, with p50/p95/p99 bands)
  R2: Inter-message interval histogram
  R3: Burst scatter (peak rate vs duration)
  R4: Session profile (p99 rate by session window)
  R5: Volume heatmap (hour × day-of-week)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


SESSION_WINDOWS = {
    "full_24h":    (None, None),
    "peak_session": (8, 16),
    "low_liq":     (2, 6),
}


def compute_rate_timeseries(
    df: pd.DataFrame,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Compute message arrival rate per exchange over time.

    Parameters
    ----------
    df : trades DataFrame with ``receive_ts_us`` and ``exchange`` columns.
    freq : pandas resample frequency string.

    Returns
    -------
    DataFrame indexed by (exchange, time_bin) with column ``msgs_per_sec``.
    """
    dt_index = pd.to_datetime(df["receive_ts_us"] * 1_000, unit="ns", utc=True)
    df = df.copy()
    df["_dt"] = dt_index

    freq_seconds = pd.tseries.frequencies.to_offset(freq).nanos / 1e9  # type: ignore

    result = (
        df.groupby("exchange")
        .apply(
            lambda g: g.set_index("_dt").resample(freq).size() / freq_seconds,
            include_groups=False,
        )
        .rename("msgs_per_sec")
        .reset_index()
    )
    result.columns = ["exchange", "time_bin", "msgs_per_sec"]
    return result


def compute_rate_percentiles(rate_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute p50/p95/p99/p99.9 message rates per exchange.

    Parameters
    ----------
    rate_ts : output of :func:`compute_rate_timeseries`.

    Returns
    -------
    DataFrame with one row per exchange and columns for each percentile.
    """
    return (
        rate_ts.groupby("exchange")["msgs_per_sec"]
        .quantile([0.50, 0.95, 0.99, 0.999])
        .unstack()
        .rename(columns={0.50: "p50", 0.95: "p95", 0.99: "p99", 0.999: "p99_9"})
        .reset_index()
    )


def compute_intermessage_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-exchange inter-message intervals in µs.

    Parameters
    ----------
    df : trades DataFrame with ``receive_ts_us`` and ``exchange``.

    Returns
    -------
    DataFrame with columns ``exchange``, ``gap_us``.
    """
    records = []
    for exchange, grp in df.groupby("exchange"):
        sorted_ts = grp["receive_ts_us"].sort_values()
        gaps = sorted_ts.diff().dropna().values
        records.append(pd.DataFrame({"exchange": exchange, "gap_us": gaps}))
    if not records:
        return pd.DataFrame(columns=["exchange", "gap_us"])
    return pd.concat(records, ignore_index=True)


def detect_bursts(
    rate_ts: pd.DataFrame,
    quantile: float = 0.95,
    min_duration_bins: int = 2,
) -> pd.DataFrame:
    """
    Detect burst events: contiguous time windows where rate > p{quantile*100}.

    Parameters
    ----------
    rate_ts : output of :func:`compute_rate_timeseries`.
    quantile : threshold percentile (default 0.95).
    min_duration_bins : minimum number of consecutive bins to qualify as a burst.

    Returns
    -------
    DataFrame with columns:
      exchange, burst_id, start_time, end_time, duration_sec,
      peak_rate, mean_rate, n_bins
    """
    records = []
    for exchange, grp in rate_ts.groupby("exchange"):
        grp = grp.sort_values("time_bin").reset_index(drop=True)
        threshold = grp["msgs_per_sec"].quantile(quantile)
        grp["above"] = grp["msgs_per_sec"] > threshold
        grp["burst_id"] = (~grp["above"]).cumsum()

        for bid, burst in grp[grp["above"]].groupby("burst_id"):
            if len(burst) < min_duration_bins:
                continue
            start = burst["time_bin"].iloc[0]
            end = burst["time_bin"].iloc[-1]
            # Duration: use time difference between first and last bin
            duration_sec = max((end - start).total_seconds(), 1.0)
            records.append({
                "exchange": exchange,
                "burst_id": int(bid),
                "start_time": start,
                "end_time": end,
                "duration_sec": duration_sec,
                "peak_rate": float(burst["msgs_per_sec"].max()),
                "mean_rate": float(burst["msgs_per_sec"].mean()),
                "n_bins": len(burst),
            })

    if not records:
        return pd.DataFrame(columns=[
            "exchange", "burst_id", "start_time", "end_time",
            "duration_sec", "peak_rate", "mean_rate", "n_bins",
        ])
    return pd.DataFrame(records)


def compute_session_profile(
    df: pd.DataFrame,
    rate_freq: str = "1min",
) -> pd.DataFrame:
    """
    Compute p99 message rates broken down by session window.

    Parameters
    ----------
    df : trades DataFrame with ``receive_ts_us`` and ``exchange``.

    Returns
    -------
    DataFrame with columns: exchange, session, p99_rate, p95_rate, mean_rate.
    """
    dt = pd.to_datetime(df["receive_ts_us"] * 1_000, unit="ns", utc=True)
    df = df.copy()
    df["_dt"] = dt
    df["_hour"] = dt.dt.hour

    records = []
    for session, (h_start, h_end) in SESSION_WINDOWS.items():
        if h_start is None:
            mask = pd.Series(True, index=df.index)
        else:
            mask = (df["_hour"] >= h_start) & (df["_hour"] < h_end)
        sub = df[mask]
        if sub.empty:
            continue
        rate_ts = compute_rate_timeseries(sub, freq=rate_freq)
        for exchange, grp in rate_ts.groupby("exchange"):
            records.append({
                "exchange": exchange,
                "session": session,
                "p99_rate": float(grp["msgs_per_sec"].quantile(0.99)),
                "p95_rate": float(grp["msgs_per_sec"].quantile(0.95)),
                "mean_rate": float(grp["msgs_per_sec"].mean()),
            })
    return pd.DataFrame(records)


def compute_volume_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trade count by (hour, day-of-week) per exchange.

    Returns
    -------
    DataFrame with columns: exchange, hour, dow, trade_count.
    """
    dt = pd.to_datetime(df["receive_ts_us"] * 1_000, unit="ns", utc=True)
    df = df.copy()
    df["_hour"] = dt.dt.hour
    df["_dow"] = dt.dt.day_of_week  # 0=Monday

    result = (
        df.groupby(["exchange", "_hour", "_dow"])
        .size()
        .reset_index(name="trade_count")
    )
    result.columns = ["exchange", "hour", "dow", "trade_count"]
    return result


def layer1_summary(df: pd.DataFrame) -> dict:
    """
    Compute all Layer 1 statistics in one call.

    Returns a dict with keys:
    ``rate_ts``, ``rate_pcts``, ``gaps``, ``bursts``,
    ``session_profile``, ``volume_heatmap``.
    """
    rate_ts = compute_rate_timeseries(df, freq="1min")
    return {
        "rate_ts": rate_ts,
        "rate_pcts": compute_rate_percentiles(rate_ts),
        "gaps": compute_intermessage_gaps(df),
        "bursts": detect_bursts(rate_ts),
        "session_profile": compute_session_profile(df),
        "volume_heatmap": compute_volume_heatmap(df),
    }
