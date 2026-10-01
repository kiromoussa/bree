"""scripts/draw_zones.py headless path: JSON polygons -> store YAML -> load_store_config, preview."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bree.events.zones import load_store_config

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("draw_zones", ROOT / "scripts" / "draw_zones.py")
draw_zones = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draw_zones)

ZONES = {"zones": [
    {"name": "cooler_bank", "kind": "cooler", "polygon": [[20, 10], [300, 10], [300, 60], [20, 60]]},
    {"name": "register", "kind": "register", "polygon": [[250, 150], [318, 150], [318, 238], [250, 238]]},
    {"name": "door", "kind": "exit", "polygon": [[0, 170], [60, 170], [60, 240], [0, 240]]}],
    "floor_points": [[10, 230, 0, 0], [310, 230, 6, 0], [280, 40, 6, 5], [40, 40, 0, 5]]}


def _inputs(tmp_path, zones=ZONES):
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((240, 320, 3), 90, np.uint8))
    js = tmp_path / "zones.json"
    js.write_text(json.dumps(zones))
    return still, js


def test_json_to_yaml_round_trip_and_preview(tmp_path):
    still, js = _inputs(tmp_path)
    out, prev = tmp_path / "cam.yaml", tmp_path / "prev.png"
    draw_zones.main(["--from-json", str(js), "--source", str(still), "--out", str(out),
                     "--camera-id", "door_cam", "--preview", str(prev)])
    s = load_store_config(out)
    assert [z.name for z in s.zones] == ["cooler_bank", "register", "door"]
    assert s.camera_id == "door_cam" and s.resolution == (320, 240)          # from the frame
    assert s.zone_at(280, 200).name == "register"
    assert s.terminals["pos_1"] == "register" and "COKE-20OZ" in s.catalog.sku_to_category   # kept from template
    assert s.rules["foot_point"] == "bottom" and s.product_classes["bottle"] == "soda_bottle"
    from bree.track.multicam import to_floor
    assert to_floor(s.floor_homography(), 310, 230) == pytest.approx((6, 0), abs=1e-6)
    img = cv2.imread(str(prev))
    assert img.shape == (240, 320, 3) and (img != 90).any()                   # zones drawn on the frame
    # nothing else written: no grabbed frame, no temp images
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cam.yaml", "prev.png", "still.png", "zones.json"]


def test_update_keeps_existing_file_and_rejects_bad_zones(tmp_path):
    still, js = _inputs(tmp_path)
    out = tmp_path / "cam.yaml"
    draw_zones.main(["--from-json", str(js), "--source", str(still), "--out", str(out)])
    # second pass on the same file: only the door is re-drawn, the rest is kept
    js.write_text(json.dumps({"zones": [{"name": "door", "kind": "exit", "polygon": [[0, 100], [40, 100], [40, 200]]}]}))
    draw_zones.main(["--from-json", str(js), "--out", str(out), "--keep-zones"])
    s = load_store_config(out)
    assert len(s.zones) == 3 and len(s.zone("door").polygon) == 3 and s.floor_points is not None
    assert s.resolution == (320, 240)
    js.write_text(json.dumps({"zones": [{"name": "x", "kind": "fridge", "polygon": [[0, 0], [1, 0], [1, 1]]}]}))
    with pytest.raises(SystemExit, match="kind must be one of"):
        draw_zones.main(["--from-json", str(js), "--out", str(out)])
    assert len(load_store_config(out).zones) == 3                             # untouched on error
