#!/usr/bin/env bash
# Create .venv with pinned dependencies and download model weights.
# CPU-only machines get the CPU torch wheel (much smaller); CUDA machines get the default wheel.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing it with pip"; pip install --user uv==0.8.17
fi
uv venv .venv -p 3.11 -q
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA GPU detected -> default torch wheel"
  uv pip install -p .venv -e ".[vision,export,dev]"
else
  echo "No NVIDIA GPU -> CPU torch wheel"
  uv pip install -p .venv --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple -e ".[vision,export,dev]"
fi
mkdir -p models
.venv/bin/python - <<'PY'
from pathlib import Path
from bree.hw import detect_hardware
from ultralytics import YOLO
hw = detect_hardware()
print("hardware:", hw.to_dict())
for name in {hw.detect_model, hw.pose_model, "yolo26n.pt", "yolo26n-pose.pt"}:
    dst = Path("models") / name
    if not dst.exists():
        YOLO(name)                     # downloads into the CWD
        Path(name).rename(dst)
    print("weights:", dst)
PY
