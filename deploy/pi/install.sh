#!/bin/sh
# Install the BREE camera node on a Raspberry Pi (Raspberry Pi OS Bookworm Lite, 64-bit).
# Run from a copy of the repo on the Pi:   sudo deploy/pi/install.sh
# It installs four apt packages, copies six Python files, a config and a systemd unit. No pip, no venv:
# the node only needs numpy, OpenCV, PyYAML and picamera2, all packaged by Raspberry Pi OS.
# Safe to run again (upgrade): the config in /etc/bree is never overwritten.
set -eu

[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEST=/opt/bree-node

apt-get update
apt-get install -y --no-install-recommends python3-picamera2 python3-opencv python3-numpy python3-yaml chrony

id bree >/dev/null 2>&1 || useradd --system --user-group --home-dir /var/lib/bree-node --groups video bree

# Only the node's files: the trigger, capture, uplink and main loop, plus the pipeline's VideoSource
# (used by the file / webcam test source).
install -d "$DEST/bree/edge" "$DEST/bree/ingest"
for f in __init__.py trigger.py capture.py uplink.py node.py; do
    install -m 644 "$REPO/src/bree/edge/$f" "$DEST/bree/edge/$f"
done
install -m 644 "$REPO/src/bree/ingest/__init__.py" "$REPO/src/bree/ingest/source.py" "$DEST/bree/ingest/"

install -d -m 750 -o root -g bree /etc/bree
if [ ! -e /etc/bree/node.yaml ]; then
    install -m 640 -o root -g bree "$REPO/deploy/pi/node.example.yaml" /etc/bree/node.yaml
    echo "wrote /etc/bree/node.yaml: set camera.id, zones, node.hub_url and node.token before starting"
fi
install -d -o bree -g bree /var/lib/bree-node /var/lib/bree-node/spool

install -m 644 "$REPO/deploy/pi/bree-node.service" /etc/systemd/system/bree-node.service
systemctl daemon-reload
systemctl enable bree-node.service

PYTHONPATH=$DEST python3 -c "import bree.edge.node, cv2, numpy, yaml; print('node imports OK, OpenCV', cv2.__version__)"
echo "next: edit /etc/bree/node.yaml, then: sudo systemctl start bree-node && journalctl -u bree-node -f"
