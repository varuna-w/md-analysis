"""
Layer 6 — Infrastructure translation.

Translates empirical measurements from layers 1–5 into engineering
specifications for a competing exchange:
  - Matching engine throughput target
  - Ring buffer depth
  - OMS atomic batch size
  - Clock discipline / sequencer window
  - Orderbook hot-cache depth
  - Network ingress budget

Output is a populated design_spec.md written to the reports directory.
"""

from __future__ import annotations

import os
from typing import Any
import pandas as pd


AVG_MSG_BYTES = 256  # bytes per trade/depth update (conservative)
N_EXCHANGES = 6      # exchanges monitored
SAFETY_FACTOR = 3.0  # 3× p99 for ME throughput target
RING_BUFFER_MARGIN = 5.0  # 5× burst depth for ring buffer


def compute_specs(
    l1: dict[str, Any],
    l2: dict[str, Any],
    l3: dict[str, Any],
    l4: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute all infrastructure design specs from layer summary dicts.

    Parameters
    ----------
    l1 : Layer 1 results with keys ``rate_pcts``, ``bursts``.
    l2 : Layer 2 results with keys ``volley_stats``.
    l3 : Layer 3 results with keys ``exec_rate_stats``, ``microburst_stats``,
         ``oo_summary``, ``resolution_report``.
    l4 : Layer 4 results with keys ``ob_update_rate``.

    Returns
    -------
    dict of engineering spec values.
    """
    rate_pcts = l1.get("rate_pcts", pd.DataFrame())
    bursts = l1.get("bursts", pd.DataFrame())
    volley_stats = l2.get("volley_stats", pd.DataFrame())
    exec_stats = l3.get("exec_rate_stats", pd.DataFrame())
    mb_stats = l3.get("microburst_stats", pd.DataFrame())
    oo_summary = l3.get("oo_summary", pd.DataFrame())
    res_report = l3.get("resolution_report", pd.DataFrame())
    ob_update = l4.get("ob_update_rate", pd.DataFrame())

    def _max(df, col, default=0.0):
        if df.empty or col not in df.columns:
            return default
        return float(df[col].max())

    def _p99(df, col, default=0.0):
        if df.empty or col not in df.columns:
            return default
        return float(df[col].quantile(0.99))

    # Peak reception rate across all exchanges (msgs/sec)
    p99_9_rate = _max(rate_pcts, "p99_9")
    p99_rate = _max(rate_pcts, "p99")

    # Max burst duration (seconds)
    max_burst_sec = _max(bursts, "duration_sec") if not bursts.empty else 1.0

    # Ring buffer: hold p99.9 rate × max burst duration × safety margin
    ring_buffer_depth = int(p99_9_rate * max_burst_sec * RING_BUFFER_MARGIN)

    # Network ingress (Mbps): p99.9 rate × msg size × N exchanges × 1.5 TCP overhead
    network_ingress_mbps = p99_9_rate * AVG_MSG_BYTES * N_EXCHANGES * 1.5 / 1e6

    # Matching Engine throughput target: peak execution rate × safety factor
    peak_exec_rate = _max(exec_stats, "peak")
    me_throughput_target = int(peak_exec_rate * SAFETY_FACTOR)

    # ME output queue depth: p99 micro-burst size × 10
    p99_burst_size = _max(mb_stats, "p99_size")
    me_output_queue_depth = int(p99_burst_size * 10)

    # Sequencer clock resolution: minimum timestamp gap across µs-resolution exchanges
    min_gap = _max(res_report, "min_gap_us")
    sequencer_clock_res_us = max(int(min_gap), 1)

    # Resequencing window: 2× max out-of-order lag
    max_oo_lag_us = _max(oo_summary, "max_lag_us")
    resequencing_window_us = int(max_oo_lag_us * 2)

    # OMS atomic batch size: p99 volley size
    p99_volley_size = _max(volley_stats, "p99")
    atomic_batch_size = int(p99_volley_size)

    # Orderbook: levels with update rate > p95
    if not ob_update.empty and "avg_updates_per_sec" in ob_update.columns:
        p95_update = ob_update["avg_updates_per_sec"].quantile(0.95)
        hot_cache_levels = int((ob_update["avg_updates_per_sec"] > p95_update).sum())
        p99_active_levels = int(ob_update["level"].quantile(0.99)) if "level" in ob_update.columns else 10
    else:
        hot_cache_levels = 5
        p99_active_levels = 20

    # Working set: p99 active levels × 256 bytes × 3× margin
    working_set_mb = p99_active_levels * AVG_MSG_BYTES * 3 / 1e6

    # Delta strategy: use deltas if update rate significantly exceeds snapshot rate
    delta_ratio_avg = _max(l4.get("delta_ratios", pd.DataFrame()), "delta_ratio")
    l2_strategy = "delta+snapshot" if delta_ratio_avg >= 0.05 else "snapshot_only"

    return {
        # Ingestion
        "p99_9_rate_per_sec": round(p99_9_rate, 1),
        "p99_rate_per_sec": round(p99_rate, 1),
        "max_burst_sec": round(max_burst_sec, 2),
        "ring_buffer_depth": ring_buffer_depth,
        "network_ingress_mbps": round(network_ingress_mbps, 2),
        # Matching Engine
        "peak_exec_rate_per_sec": round(peak_exec_rate, 1),
        "me_throughput_target": me_throughput_target,
        "me_output_queue_depth": me_output_queue_depth,
        "sequencer_clock_res_us": sequencer_clock_res_us,
        "resequencing_window_us": resequencing_window_us,
        # OMS
        "atomic_batch_size": atomic_batch_size,
        # Orderbook
        "hot_cache_levels": hot_cache_levels,
        "p99_active_levels": p99_active_levels,
        "working_set_mb": round(working_set_mb, 3),
        "l2_strategy": l2_strategy,
    }


SPEC_TEMPLATE = """\
# Market Data Exchange — Infrastructure Design Specification

Generated from empirical analysis of Binance, Binance Futures, Coinbase,
Kraken, Delta India, Delta Global feeds.

---

## 1. Message Ingestion

| Parameter | Value | Derivation |
|-----------|-------|------------|
| p99.9 arrival rate (msgs/sec) | **{p99_9_rate_per_sec}** | L1 rate_pcts.p99_9, all exchanges |
| p99 arrival rate (msgs/sec) | **{p99_rate_per_sec}** | L1 rate_pcts.p99 |
| Max burst duration | **{max_burst_sec} s** | L1 burst detection |
| **Ring buffer depth** | **{ring_buffer_depth}** | p99.9 × max_burst × 5× margin |
| **Network ingress budget** | **{network_ingress_mbps} Mbps** | p99.9 × 256 B × 6 exch × 1.5 |

---

## 2. Matching Engine

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Peak execution rate (exec/sec) | **{peak_exec_rate_per_sec}** | L3 exec_rate_stats.peak |
| **ME throughput target** | **{me_throughput_target} exec/sec** | peak × 3× safety |
| **ME output queue depth** | **{me_output_queue_depth}** | p99 micro-burst × 10 |
| **Sequencer clock resolution** | **{sequencer_clock_res_us} µs** | L3 min non-zero timestamp gap |
| **Resequencing window** | **{resequencing_window_us} µs** | 2× max out-of-order lag |

---

## 3. OMS

| Parameter | Value | Derivation |
|-----------|-------|------------|
| **Atomic batch size** | **{atomic_batch_size}** | L2 p99 volley size |
| Position update rate | ≈ ME throughput target | Assume 1 fill per execution |

---

## 4. Orderbook

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Hot-cache depth (levels) | **{hot_cache_levels}** | L4 levels with update rate > p95 |
| p99 active levels | **{p99_active_levels}** | L4 level count distribution |
| **Working set** | **{working_set_mb} MB** | p99 levels × 256 B × 3× margin |
| **L2 update strategy** | **{l2_strategy}** | delta_ratio threshold |

---

*All values are engineering minimums.  Apply 2× safety margin for production sizing.*
"""


def write_design_spec(specs: dict[str, Any], output_path: str) -> None:
    """Write the design spec markdown file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = SPEC_TEMPLATE.format(**specs)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Design spec written to: {output_path}")
