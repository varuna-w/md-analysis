"""
mda — Market Data Analytics package.

Six-layer framework for exchange feed characterisation:
  Layer 1: Raw message rate characterisation (R1–R5)
  Layer 2: Trade flow anatomy (T1–T4)
  Layer 3: Exchange timestamp analysis / ME profiling (E1–E7)
  Layer 4: Orderbook dynamics (O1–O5)
  Layer 5: Cross-exchange correlation (X1–X3)
  Layer 6: Infrastructure translation → design_spec.md
"""

DATA_DIR = "/data/parquet"
REPORTS_DIR = "/data/notebooks/../../../reports"  # overridden by notebooks

EXCHANGES = [
    "binance",
    "binance_futures",
    "coinbase",
    "kraken",
    "delta_india",
    "delta_global",
]

# Exchanges with ms-only timestamp resolution
MS_RESOLUTION_EXCHANGES = {"binance", "binance_futures"}
# Exchanges with µs precision in ISO timestamp string
US_RESOLUTION_EXCHANGES = {"coinbase", "kraken", "delta_india", "delta_global"}
