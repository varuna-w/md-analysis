#!/bin/bash
# ─── Analytics Instance Bootstrap ─────────────────────────────────────────────
# Installs: Python analytics stack, JupyterLab, AWS CLI, sync tooling
# Runs as root on first boot via EC2 user-data.
set -euo pipefail
exec > >(tee /var/log/mda-bootstrap.log | logger -t mda-bootstrap) 2>&1
echo "=== MDA bootstrap started at $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
S3_BUCKET="${s3_bucket}"
AWS_REGION="${aws_region}"
JUPYTER_PORT="${jupyter_port}"
ANALYTICS_USER="ubuntu"
VENV="/opt/analytics"
DATA_DIR="/data"
NOTEBOOKS_DIR="$DATA_DIR/notebooks"
PARQUET_DIR="$DATA_DIR/parquet"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -q
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential libssl-dev libffi-dev \
    git curl wget unzip \
    htop nvme-cli \
    awscli

# ── 2. Mount EBS data volume ──────────────────────────────────────────────────
DEVICE="/dev/xvdf"
if ! blkid "$DEVICE" &>/dev/null; then
    echo "Formatting $DEVICE as xfs..."
    mkfs -t xfs "$DEVICE"
fi
mkdir -p "$DATA_DIR"
if ! mountpoint -q "$DATA_DIR"; then
    mount "$DEVICE" "$DATA_DIR"
fi
# Persist across reboots
grep -qF "$DEVICE" /etc/fstab || \
    echo "$DEVICE $DATA_DIR xfs defaults,nofail 0 2" >> /etc/fstab

mkdir -p "$NOTEBOOKS_DIR" "$PARQUET_DIR"
chown -R "$ANALYTICS_USER:$ANALYTICS_USER" "$DATA_DIR"

# ── 3. Python analytics environment ──────────────────────────────────────────
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel setuptools

pip install \
    jupyterlab==4.* \
    jupyterlab-git \
    ipywidgets \
    pandas \
    pyarrow \
    numpy \
    scipy \
    statsmodels \
    scikit-learn \
    plotly \
    matplotlib \
    seaborn \
    dask[dataframe] \
    "dask[diagnostics]" \
    boto3 \
    s3fs \
    fsspec \
    pytz \
    tqdm \
    rich \
    polars

# ── 4. JupyterLab configuration ───────────────────────────────────────────────
JUPYTER_CONFIG_DIR="/home/$ANALYTICS_USER/.jupyter"
mkdir -p "$JUPYTER_CONFIG_DIR"

# Generate a random token for auth and save it
JUPYTER_TOKEN=$(openssl rand -hex 24)
echo "$JUPYTER_TOKEN" > "$JUPYTER_CONFIG_DIR/jupyter_token.txt"
chmod 600 "$JUPYTER_CONFIG_DIR/jupyter_token.txt"

cat > "$JUPYTER_CONFIG_DIR/jupyter_lab_config.py" << JUPYEOF
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = $JUPYTER_PORT
c.ServerApp.open_browser = False
c.ServerApp.root_dir = '$NOTEBOOKS_DIR'
c.ServerApp.token = open('$JUPYTER_CONFIG_DIR/jupyter_token.txt').read().strip()
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = False
JUPYEOF

chown -R "$ANALYTICS_USER:$ANALYTICS_USER" "$JUPYTER_CONFIG_DIR"

# ── 5. JupyterLab systemd service ────────────────────────────────────────────
cat > /etc/systemd/system/jupyterlab.service << SVCEOF
[Unit]
Description=JupyterLab Analytics Server
After=network.target mnt-data.mount

[Service]
Type=simple
User=$ANALYTICS_USER
WorkingDirectory=$NOTEBOOKS_DIR
ExecStart=$VENV/bin/jupyter lab --config=$JUPYTER_CONFIG_DIR/jupyter_lab_config.py
Restart=always
RestartSec=5
Environment=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin
Environment=AWS_DEFAULT_REGION=$AWS_REGION

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable jupyterlab
systemctl start jupyterlab

# ── 6. S3 sync script ─────────────────────────────────────────────────────────
# Syncs closed parquet files from S3 to the local data volume.
# The "currently open" file in each table dir is excluded (largest file heuristic).
cat > /usr/local/bin/mda-sync-from-s3 << 'SYNCEOF'
#!/bin/bash
# Sync parquet files from S3 to local cache.
# Usage: mda-sync-from-s3 [--full]  (--full forces re-download of all files)
set -euo pipefail

S3_BUCKET="__S3_BUCKET__"
LOCAL_DIR="/data/parquet"
TABLES="trades orderbook orderbook_updates latency"

for table in $TABLES; do
    mkdir -p "$LOCAL_DIR/$table"
    echo "Syncing $table from s3://$S3_BUCKET/$table/ ..."
    aws s3 sync "s3://$S3_BUCKET/$table/" "$LOCAL_DIR/$table/" \
        --exclude "*" \
        --include "*.parquet" \
        ${1:+--delete}
done

echo "Sync complete at $(date -u)"
SYNCEOF

sed -i "s|__S3_BUCKET__|$S3_BUCKET|g" /usr/local/bin/mda-sync-from-s3
chmod +x /usr/local/bin/mda-sync-from-s3

# ── 7. S3 sync cron — runs every 15 minutes ───────────────────────────────────
cat > /etc/cron.d/mda-s3-sync << CRONEOF
*/15 * * * * root /usr/local/bin/mda-sync-from-s3 >> /var/log/mda-s3-sync.log 2>&1
CRONEOF

# ── 8. Starter notebook ───────────────────────────────────────────────────────
cat > "$NOTEBOOKS_DIR/00_getting_started.py" << 'NBEOF'
# ── Market Data Analytics — Getting Started ───────────────────────────────────
# Run: jupyter lab (via SSH tunnel) then open this file
#
# Data layout on this instance:
#   /data/parquet/trades/           — all trade events (all exchanges)
#   /data/parquet/orderbook/        — L1/L2 orderbook snapshots
#   /data/parquet/orderbook_updates/— incremental orderbook updates
#   /data/parquet/latency/          — WS latency measurements
#
# Quick start: load all trades
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd

# Load all trade parquet files as a dataset (lazy — no data read yet)
dataset = ds.dataset('/data/parquet/trades/', format='parquet')
print(f"Total fragments: {len(list(dataset.get_fragments()))}")
print(f"Schema: {dataset.schema}")

# Filter to a single symbol across all exchanges
btc = dataset.to_table(
    filter=ds.field('symbol') == 'BTCUSDT',
    columns=['timestamp', 'exchange', 'symbol', 'price', 'price_scale', 'quantity', 'quantity_scale', 'taker_side_buy']
).to_pandas()

# Reconstruct float price
btc['price_f'] = btc['price'] / (10 ** btc['price_scale'])
btc['qty_f']   = btc['quantity'] / (10 ** btc['quantity_scale'])
btc['timestamp'] = pd.to_datetime(btc['timestamp'])
btc = btc.set_index('timestamp').sort_index()

print(btc.tail(10))
NBEOF

chown "$ANALYTICS_USER:$ANALYTICS_USER" "$NOTEBOOKS_DIR/00_getting_started.py"

# ── 9. Done ───────────────────────────────────────────────────────────────────
JUPYTER_TOKEN=$(cat "$JUPYTER_CONFIG_DIR/jupyter_token.txt")
echo ""
echo "=== MDA bootstrap complete at $(date -u) ==="
echo "JupyterLab token: $JUPYTER_TOKEN"
echo "SSH tunnel:  ssh -i ~/.ssh/md-rnd-v.pem -L $JUPYTER_PORT:localhost:$JUPYTER_PORT ubuntu@$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "Then open:   http://localhost:$JUPYTER_PORT/lab?token=$JUPYTER_TOKEN"
