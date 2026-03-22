#!/bin/bash
# ── Market Data Analysis — Full Run Script ─────────────────────────────────
# Executes all 6 analysis layers in sequence using nbconvert.
# Scheduled to run 12 hours after data collection restart.
#
# Usage: bash run_analysis.sh [--notebooks-dir DIR] [--reports-dir DIR]
#
# Logs to: /var/log/mda-run.log

set -euo pipefail
exec > >(tee /var/log/mda-run.log | logger -t mda-run) 2>&1
echo "=== MDA run started at $(date -u) ==="

NOTEBOOKS_DIR="${NOTEBOOKS_DIR:-/data/notebooks}"
DATA_DIR="${DATA_DIR:-/data/parquet}"
REPORTS_DIR="${REPORTS_DIR:-/data/reports}"
VENV="/opt/analytics"
MDA_SRC="/data/notebooks/md-analysis"   # repo is synced here

# Activate virtual env
source "$VENV/bin/activate"

# Ensure mda package is installed (editable install)
if [ -d "$MDA_SRC" ]; then
    pip install -e "$MDA_SRC" --quiet
else
    echo "WARNING: md-analysis source not found at $MDA_SRC, using installed mda if available"
fi

# Ensure kaleido for PNG export
pip install kaleido --quiet 2>/dev/null || echo "kaleido install failed — HTML-only export"

mkdir -p "$REPORTS_DIR"

# ── Force one last S3 sync before analysis ─────────────────────────────────
echo "=== Syncing latest data from S3 ==="
/usr/local/bin/mda-sync-from-s3 2>&1 || echo "S3 sync warning (may already be current)"

# ── Run each notebook via nbconvert ────────────────────────────────────────
NB_SRC="$MDA_SRC/notebooks"
NB_OUT="$NOTEBOOKS_DIR/executed"
mkdir -p "$NB_OUT"

NOTEBOOKS=(
    "L1_message_rates.ipynb"
    "L2_trade_flow.ipynb"
    "L3_timestamp_analysis.ipynb"
    "L4_orderbook_dynamics.ipynb"
    "L5_cross_exchange.ipynb"
    "L6_design_translation.ipynb"
)

# Export env vars for notebooks to pick up
export DATA_DIR REPORTS_DIR

FAILED=()
for nb in "${NOTEBOOKS[@]}"; do
    nb_path="$NB_SRC/$nb"
    echo ""
    echo "=== Running $nb at $(date -u) ==="
    if [ ! -f "$nb_path" ]; then
        echo "SKIP: $nb_path not found"
        continue
    fi
    jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output "$NB_OUT/$nb" \
        "$nb_path" && echo "OK: $nb" || {
            echo "FAILED: $nb"
            FAILED+=("$nb")
        }
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== MDA run complete at $(date -u) ==="
echo "Reports written to: $REPORTS_DIR"
ls "$REPORTS_DIR" 2>/dev/null || echo "(no reports yet)"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED notebooks: ${FAILED[*]}"
    exit 1
fi

echo "=== All notebooks completed successfully ==="
