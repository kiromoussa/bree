#!/usr/bin/env bash
# Pull whatever public shoplifting datasets are reachable from official sources.
# Anything needing credentials or a browser is skipped with a message; see data/README.md.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DATA="$ROOT/data"; mkdir -p "$DATA/real"

# 0) Real pedestrian footage for FPS / tracking smoke tests (OpenCV sample video, no theft labels).
[ -f "$DATA/real/vtest.avi" ] || curl -sSL --max-time 300 -o "$DATA/real/vtest.avi" \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi \
  || echo "WARN: vtest.avi download failed"

# 1) Simuletic synthetic CCTV shoplifting and 2) DCSASS: Kaggle, needs an API token.
if command -v kaggle >/dev/null 2>&1 && { [ -f ~/.kaggle/kaggle.json ] || [ -n "${KAGGLE_KEY:-}" ]; }; then
  kaggle datasets download -d simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm -p "$DATA/simuletic" --unzip \
    || echo "WARN: Simuletic download failed (see data/README.md)"
  kaggle datasets download -d mateohervas/dcsass-dataset -p "$DATA/dcsass" --unzip \
    || echo "WARN: DCSASS download failed (see data/README.md)"
else
  echo "SKIP: Kaggle datasets (Simuletic, DCSASS): no kaggle CLI / API token. See data/README.md."
fi

# 3) PoseLift: official data is a public Google Drive folder -> manual step (see data/README.md).
[ -d "$DATA/poselift_official" ] || echo "MANUAL: PoseLift -> download the official Google Drive folder into data/poselift_official/ (data/README.md)"

# 4) Toy clips (always available): rendered 2D scenes, clearly labelled toy data.
"$ROOT/.venv/bin/python" -m bree.cli render-toy --out "$DATA/toy" || echo "WARN: toy clip rendering failed"

echo "done. contents of $DATA:"; ls "$DATA"
