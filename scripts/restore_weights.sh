#!/bin/sh
# Put the trained weights kept in git (weights/) where the pipeline looks for them.
# The public pretrained models (YOLO26, pose, DINOv2) are fetched by `make setup`, not kept here.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/data/synth/weights" "$ROOT/models"
cp -n "$ROOT"/weights/sim_sku*.pt "$ROOT"/weights/sim_sku*.json "$ROOT/data/synth/weights/" 2>/dev/null || true
cp -n "$ROOT"/weights/conceal_*.pt "$ROOT"/weights/conceal_*.onnx "$ROOT/models/" 2>/dev/null || true
echo "weights in place:"; ls "$ROOT/data/synth/weights" "$ROOT/models" | grep -E "sim_sku|conceal_" | sort | uniq
