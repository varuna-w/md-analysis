"""
All 29 Plotly chart builders for the mda analytics suite.

Every function returns a ``plotly.graph_objects.Figure``.
Call ``fig.write_html(path)`` and ``fig.write_image(path)`` to persist.

Graph ID map:
  R1–R5  Layer 1: message rates
  T1–T4  Layer 2: trade flow
  E1–E7  Layer 3: timestamp analysis
  O1–O5  Layer 4: orderbook dynamics
  X1–X3  Layer 5: cross-exchange
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

EXCHANGE_COLORS = {
    "binance": "#F0B90B",
    "binance_futures": "#E8A808",
    "coinbase": "#0052FF",
    "kraken": "#5741D9",
    "delta_india": "#00C9A7",
    "delta_global": "#00A388",
}

def _exchange_color(exchange: str) -> str:
    return EXCHANGE_COLORS.get(exchange, "#888888")


# ── Layer 1: Message Rates ──────────────────────────────────────────────────

def plot_rate_timeseries(rate_ts: pd.DataFrame, rate_pcts: pd.DataFrame) -> go.Figure:
    """R1: Rate time series with p50/p95/p99 bands per exchange."""
    fig = go.Figure()
    for exchange in rate_ts["exchange"].unique():
        grp = rate_ts[rate_ts["exchange"] == exchange].sort_values("time_bin")
        color = _exchange_color(exchange)
        fig.add_trace(go.Scatter(
            x=grp["time_bin"], y=grp["msgs_per_sec"],
            mode="lines", name=exchange,
            line=dict(color=color, width=1.2),
            opacity=0.8,
        ))
        # Add p99 annotation
        if not rate_pcts.empty:
            row = rate_pcts[rate_pcts["exchange"] == exchange]
            if not row.empty:
                p99 = float(row["p99"].iloc[0])
                fig.add_hline(y=p99, line_dash="dot", line_color=color,
                              annotation_text=f"{exchange} p99={p99:.0f}",
                              annotation_position="bottom right")

    fig.update_layout(
        title="R1: Message Arrival Rate (msgs/sec)",
        xaxis_title="Time (UTC)",
        yaxis_title="msgs/sec",
        legend_title="Exchange",
        template="plotly_dark",
        height=500,
    )
    return fig


def plot_intermessage_histogram(gaps: pd.DataFrame) -> go.Figure:
    """R2: Inter-message interval histogram (log-x)."""
    fig = go.Figure()
    for exchange in gaps["exchange"].unique():
        g = gaps[gaps["exchange"] == exchange]["gap_us"].clip(lower=1)
        fig.add_trace(go.Histogram(
            x=np.log10(g + 1), name=exchange,
            opacity=0.6,
            marker_color=_exchange_color(exchange),
            nbinsx=80,
        ))
    fig.update_layout(
        title="R2: Inter-message Interval Distribution",
        xaxis_title="log10(gap µs)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_burst_scatter(bursts: pd.DataFrame) -> go.Figure:
    """R3: Burst scatter — peak rate vs duration."""
    if bursts.empty:
        return go.Figure().update_layout(title="R3: No bursts detected")
    fig = px.scatter(
        bursts,
        x="peak_rate", y="duration_sec",
        color="exchange",
        size="n_bins",
        hover_data=["start_time", "mean_rate"],
        color_discrete_map=EXCHANGE_COLORS,
        title="R3: Burst Events — Peak Rate vs Duration",
        labels={"peak_rate": "Peak rate (msgs/sec)", "duration_sec": "Duration (sec)"},
        template="plotly_dark",
    )
    fig.update_layout(height=500)
    return fig


def plot_session_profile(session_df: pd.DataFrame) -> go.Figure:
    """R4: Grouped bar — p99 rate by session window per exchange."""
    fig = px.bar(
        session_df,
        x="exchange", y="p99_rate",
        color="session",
        barmode="group",
        title="R4: p99 Message Rate by Session Window",
        labels={"p99_rate": "p99 msgs/sec", "exchange": "Exchange"},
        template="plotly_dark",
    )
    fig.update_layout(height=450)
    return fig


def plot_volume_heatmap(heatmap: pd.DataFrame, exchange: str) -> go.Figure:
    """R5: Volume heatmap — hour × day-of-week for one exchange."""
    sub = heatmap[heatmap["exchange"] == exchange]
    pivot = sub.pivot_table(index="hour", columns="dow", values="trade_count", fill_value=0)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[dow_labels[c] for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Viridis",
        colorbar_title="Trades",
    ))
    fig.update_layout(
        title=f"R5: Trade Volume Heatmap — {exchange}",
        xaxis_title="Day of Week",
        yaxis_title="Hour (UTC)",
        template="plotly_dark",
        height=450,
    )
    return fig


# ── Layer 2: Trade Flow ────────────────────────────────────────────────────

def plot_trade_size_histogram(df: pd.DataFrame) -> go.Figure:
    """T1: Trade size (qty_f) distribution with log-normal fit overlay."""
    from scipy import stats as scipy_stats
    fig = go.Figure()
    for exchange in df["exchange"].unique():
        vals = df[df["exchange"] == exchange]["qty_f"].dropna()
        vals = vals[vals > 0]
        if vals.empty:
            continue
        fig.add_trace(go.Histogram(
            x=np.log10(vals + 1e-12), name=exchange,
            opacity=0.5,
            marker_color=_exchange_color(exchange),
            nbinsx=60,
            histnorm="probability density",
        ))
    fig.update_layout(
        title="T1: Trade Quantity Distribution (log10 scale)",
        xaxis_title="log10(qty)",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_aggressor_imbalance(imbalance: pd.DataFrame) -> go.Figure:
    """T2: Rolling buy ratio imbalance (zero-centred)."""
    fig = go.Figure()
    for exchange in imbalance["exchange"].unique():
        grp = imbalance[imbalance["exchange"] == exchange].sort_values("time_bin")
        fig.add_trace(go.Scatter(
            x=grp["time_bin"], y=grp["imbalance"],
            mode="lines", name=exchange,
            line=dict(color=_exchange_color(exchange), width=1.2),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_layout(
        title="T2: Buy/Sell Aggressor Imbalance (rolling 1-min)",
        xaxis_title="Time (UTC)",
        yaxis_title="Imbalance (buy_ratio − 0.5)",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_volley_distribution(volley_stats: pd.DataFrame) -> go.Figure:
    """T3: Histogram of volley sizes (log-y)."""
    fig = go.Figure()
    for exchange in volley_stats["exchange"].unique():
        sizes = volley_stats[volley_stats["exchange"] == exchange]["size"]
        fig.add_trace(go.Histogram(
            x=sizes, name=exchange,
            opacity=0.6,
            marker_color=_exchange_color(exchange),
            nbinsx=40,
        ))
    fig.update_layout(
        title="T3: Volley Size Distribution (trades per volley, ≤5ms gap)",
        xaxis_title="Volley size (trades)",
        yaxis_title="Count",
        yaxis_type="log",
        barmode="overlay",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_tick_frequency(tick_freq: pd.DataFrame) -> go.Figure:
    """T4: Unique prices/sec rolling time series."""
    fig = go.Figure()
    for exchange in tick_freq["exchange"].unique():
        grp = tick_freq[tick_freq["exchange"] == exchange].sort_values("time_bin")
        fig.add_trace(go.Scatter(
            x=grp["time_bin"], y=grp["unique_prices"],
            mode="lines", name=exchange,
            line=dict(color=_exchange_color(exchange), width=1),
        ))
    fig.update_layout(
        title="T4: Unique Prices/sec (tick activity)",
        xaxis_title="Time (UTC)",
        yaxis_title="Unique prices/sec",
        template="plotly_dark",
        height=450,
    )
    return fig


# ── Layer 3: Timestamp Analysis ────────────────────────────────────────────

def plot_execution_rate_ts(exec_rate: pd.DataFrame) -> go.Figure:
    """E1: Exchange-timestamp-derived execution rate time series."""
    fig = go.Figure()
    for exchange in exec_rate["exchange"].unique():
        grp = exec_rate[exec_rate["exchange"] == exchange].sort_values("time_bin")
        fig.add_trace(go.Scatter(
            x=grp["time_bin"], y=grp["exec_per_sec"],
            mode="lines", name=exchange,
            line=dict(color=_exchange_color(exchange), width=1.2),
        ))
    fig.update_layout(
        title="E1: Execution Rate (exchange timestamp, exec/sec)",
        xaxis_title="Time (UTC)",
        yaxis_title="exec/sec",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_microburst_size_hist(bursts: pd.DataFrame) -> go.Figure:
    """E2: Micro-burst size histogram (log-y)."""
    fig = go.Figure()
    for exchange in bursts["exchange"].unique():
        sizes = bursts[bursts["exchange"] == exchange]["size"]
        fig.add_trace(go.Histogram(
            x=sizes, name=exchange,
            opacity=0.6,
            marker_color=_exchange_color(exchange),
            nbinsx=30,
        ))
    fig.update_layout(
        title="E2: Micro-burst Size Distribution (≤1ms gap, min 3 trades)",
        xaxis_title="Burst size (trades)",
        yaxis_title="Count",
        yaxis_type="log",
        barmode="overlay",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_microburst_heatmap(heatmap: pd.DataFrame, exchange: str) -> go.Figure:
    """E3: Micro-burst intensity heatmap — hour × day-of-week."""
    sub = heatmap[heatmap["exchange"] == exchange]
    if sub.empty:
        return go.Figure().update_layout(title=f"E3: No micro-bursts for {exchange}")
    pivot = sub.pivot_table(
        index="hour", columns="dow", values="peak_trades_per_ms", fill_value=0
    )
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[dow_labels[c] for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Inferno",
        colorbar_title="Peak trades/ms",
    ))
    fig.update_layout(
        title=f"E3: Micro-burst Intensity — {exchange}",
        xaxis_title="Day of Week",
        yaxis_title="Hour (UTC)",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_timestamp_gap_hist(gaps: pd.DataFrame) -> go.Figure:
    """E4: Exchange timestamp inter-event gap distribution (log-x)."""
    fig = go.Figure()
    for exchange in gaps["exchange"].unique():
        g = gaps[gaps["exchange"] == exchange]["gap_us"].clip(lower=0.1)
        g = g[g > 0]
        fig.add_trace(go.Histogram(
            x=np.log10(g + 1e-6), name=exchange,
            opacity=0.6,
            marker_color=_exchange_color(exchange),
            nbinsx=80,
        ))
    fig.update_layout(
        title="E4: Exchange Timestamp Gap Distribution (reveals ms/µs resolution)",
        xaxis_title="log10(gap µs)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_latency_cdf(df: pd.DataFrame) -> go.Figure:
    """E5: Feed latency CDF per exchange."""
    fig = go.Figure()
    for exchange in df["exchange"].unique():
        lat = df[df["exchange"] == exchange]["latency_ms"].dropna().sort_values()
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        # Clip to [p1, p99.9] for readability
        p1, p99_9 = lat.quantile(0.01), lat.quantile(0.999)
        mask = (lat >= p1) & (lat <= p99_9)
        fig.add_trace(go.Scatter(
            x=lat[mask].values, y=cdf[mask.values],
            mode="lines", name=exchange,
            line=dict(color=_exchange_color(exchange)),
        ))
    fig.update_layout(
        title="E5: Feed Latency CDF (receive − exchange timestamp)",
        xaxis_title="Latency (ms)",
        yaxis_title="CDF",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_latency_drift_ts(drift: pd.DataFrame) -> go.Figure:
    """E6: Rolling median latency over time (clock drift / congestion)."""
    fig = go.Figure()
    for exchange in drift["exchange"].unique():
        grp = drift[drift["exchange"] == exchange].sort_values("time_bin")
        fig.add_trace(go.Scatter(
            x=grp["time_bin"], y=grp["p50_latency_ms"],
            mode="lines", name=exchange,
            line=dict(color=_exchange_color(exchange)),
        ))
    fig.update_layout(
        title="E6: Feed Latency Drift (5-min rolling p50)",
        xaxis_title="Time (UTC)",
        yaxis_title="Median latency (ms)",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_oo_magnitude_hist(oo_events: pd.DataFrame) -> go.Figure:
    """E7: Out-of-order event magnitude histogram."""
    if oo_events.empty or "oo_lag_us" not in oo_events.columns:
        return go.Figure().update_layout(title="E7: No out-of-order events detected")
    fig = go.Figure()
    for exchange in oo_events["exchange"].unique():
        lags = oo_events[oo_events["exchange"] == exchange]["oo_lag_us"]
        fig.add_trace(go.Histogram(
            x=lags / 1_000, name=exchange,
            opacity=0.6,
            marker_color=_exchange_color(exchange),
            nbinsx=40,
        ))
    fig.update_layout(
        title="E7: Out-of-order Event Magnitude",
        xaxis_title="Backward jump (ms)",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
    )
    return fig


# ── Layer 4: Orderbook Dynamics ────────────────────────────────────────────

def plot_update_rate_by_depth(update_rate: pd.DataFrame) -> go.Figure:
    """O1: Update rate by depth rank (bar chart)."""
    fig = px.bar(
        update_rate,
        x="level", y="avg_updates_per_sec",
        color="exchange",
        barmode="group",
        color_discrete_map=EXCHANGE_COLORS,
        title="O1: Orderbook Update Rate by Depth Level",
        labels={"level": "Depth rank", "avg_updates_per_sec": "avg updates/sec"},
        template="plotly_dark",
    )
    fig.update_layout(height=450)
    return fig


def plot_bbo_spread_depth(bbo: pd.DataFrame) -> go.Figure:
    """O2: BBO spread (bps) and quantity over time (dual-axis)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for exchange in bbo["exchange"].unique():
        grp = bbo[bbo["exchange"] == exchange].copy()
        if "ts_dt" not in grp.columns and "received_at" in grp.columns:
            grp["ts_dt"] = pd.to_datetime(grp["received_at"], utc=True, format="mixed")
        if "ts_dt" not in grp.columns:
            continue
        grp = grp.sort_values("ts_dt")
        color = _exchange_color(exchange)
        # Resample to 1-min median
        grp_r = grp.set_index("ts_dt").resample("1min").median(numeric_only=True).reset_index()
        fig.add_trace(
            go.Scatter(x=grp_r["ts_dt"], y=grp_r["spread_bps"],
                       name=f"{exchange} spread", line=dict(color=color)),
            secondary_y=False,
        )
        if "bid_qty" in grp_r.columns:
            fig.add_trace(
                go.Scatter(x=grp_r["ts_dt"], y=grp_r["bid_qty"],
                           name=f"{exchange} bid_qty", line=dict(color=color, dash="dot")),
                secondary_y=True,
            )
    fig.update_yaxes(title_text="Spread (bps)", secondary_y=False)
    fig.update_yaxes(title_text="BBO qty", secondary_y=True)
    fig.update_layout(
        title="O2: BBO Spread (bps) and Quantity",
        template="plotly_dark",
        height=500,
    )
    return fig


def plot_delta_compression_ts(delta_ratios: pd.DataFrame) -> go.Figure:
    """O3: Delta compression ratio over time."""
    fig = go.Figure()
    for exchange in delta_ratios["exchange"].unique():
        grp = delta_ratios[delta_ratios["exchange"] == exchange].sort_values("time_bin")
        fig.add_trace(go.Scatter(
            x=grp["time_bin"], y=grp["delta_ratio"],
            mode="lines", name=exchange,
            line=dict(color=_exchange_color(exchange)),
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.3,
                  annotation_text="ratio=1 (equal)")
    fig.update_layout(
        title="O3: Delta Compression Ratio (updates / snapshots)",
        xaxis_title="Time (UTC)",
        yaxis_title="Delta ratio",
        template="plotly_dark",
        height=400,
    )
    return fig


def plot_book_activity_heatmap(update_rate: pd.DataFrame, exchange: str) -> go.Figure:
    """O4: Orderbook activity heatmap — hour × depth rank."""
    sub = update_rate[update_rate["exchange"] == exchange]
    if sub.empty:
        return go.Figure().update_layout(title=f"O4: No data for {exchange}")
    pivot = sub.pivot_table(
        index="level", columns="exchange", values="avg_updates_per_sec", fill_value=0
    )
    fig = go.Figure(go.Heatmap(
        z=sub["avg_updates_per_sec"].values.reshape(-1, 1),
        y=sub["level"].values,
        colorscale="Blues",
        colorbar_title="avg upd/sec",
    ))
    fig.update_layout(
        title=f"O4: Orderbook Activity by Depth Level — {exchange}",
        yaxis_title="Depth rank",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_level_lifetime_hist(lifetimes: pd.DataFrame) -> go.Figure:
    """O5: Price level lifetime histogram (log-x)."""
    fig = go.Figure()
    for exchange in lifetimes["exchange"].unique():
        lt_ms = lifetimes[lifetimes["exchange"] == exchange]["lifetime_us"] / 1_000
        lt_ms = lt_ms.clip(lower=0.01)
        fig.add_trace(go.Histogram(
            x=np.log10(lt_ms + 1e-3), name=exchange,
            opacity=0.6,
            marker_color=_exchange_color(exchange),
            nbinsx=60,
        ))
    fig.update_layout(
        title="O5: Price Level Lifetime Distribution",
        xaxis_title="log10(lifetime ms)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_dark",
        height=450,
    )
    return fig


# ── Layer 5: Cross-Exchange ────────────────────────────────────────────────

def plot_cross_venue_lag_cdf(lags: pd.DataFrame) -> go.Figure:
    """X1: Cross-venue lag CDF per exchange pair."""
    if lags.empty:
        return go.Figure().update_layout(title="X1: No cross-venue lag data")
    fig = go.Figure()
    for (ex_a, ex_b), grp in lags.groupby(["exchange_a", "exchange_b"]):
        lag_ms = grp["lag_us"].sort_values() / 1_000
        cdf = np.arange(1, len(lag_ms) + 1) / len(lag_ms)
        name = f"{ex_a}→{ex_b}"
        fig.add_trace(go.Scatter(
            x=lag_ms.values, y=cdf, mode="lines", name=name,
        ))
    fig.update_layout(
        title="X1: Cross-Venue Lag CDF (BTC price propagation)",
        xaxis_title="Lag (ms)",
        yaxis_title="CDF",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_cumulative_capacity(capacity: pd.DataFrame) -> go.Figure:
    """X2: Stacked bar — cumulative p99 capacity across exchanges."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=capacity["exchange"],
        y=capacity["trade_p99_per_sec"],
        name="Trades p99/sec",
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        x=capacity["exchange"],
        y=capacity["ob_update_p99_per_sec"],
        name="OB updates p99/sec",
        marker_color="coral",
    ))
    fig.update_layout(
        title="X2: Cumulative p99 Capacity by Exchange",
        barmode="stack",
        xaxis_title="Exchange",
        yaxis_title="msgs/sec",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_cascade_throughput(
    rate_ts: pd.DataFrame,
    cascade_windows: pd.DatetimeIndex,
) -> go.Figure:
    """X3: Rate time series annotated with cascade event windows."""
    fig = go.Figure()
    for exchange in rate_ts["exchange"].unique():
        grp = rate_ts[rate_ts["exchange"] == exchange].sort_values("time_bin")
        fig.add_trace(go.Scatter(
            x=grp["time_bin"], y=grp["msgs_per_sec"],
            mode="lines", name=exchange,
            line=dict(color=_exchange_color(exchange), width=1.2),
        ))
    for ts in cascade_windows[:20]:  # annotate up to 20 cascade events
        fig.add_vline(x=ts, line_dash="dash", line_color="red", opacity=0.5)
    fig.update_layout(
        title="X3: Message Rate with Cascade Events (red=BTC >2% move)",
        xaxis_title="Time (UTC)",
        yaxis_title="msgs/sec",
        template="plotly_dark",
        height=500,
    )
    return fig


# ── Persistence helpers ────────────────────────────────────────────────────

def save_figure(fig: go.Figure, name: str, reports_dir: str) -> None:
    """Save a figure as HTML and PNG to the reports directory."""
    import os
    os.makedirs(reports_dir, exist_ok=True)
    html_path = os.path.join(reports_dir, f"{name}.html")
    png_path = os.path.join(reports_dir, f"{name}.png")
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, width=1400, height=fig.layout.height or 500, scale=1.5)
    except Exception as e:
        print(f"PNG export failed for {name}: {e} (kaleido may not be installed)")
    print(f"Saved: {html_path}")
