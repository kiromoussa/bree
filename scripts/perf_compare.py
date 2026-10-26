"""Compare two `bree run` outputs frame by frame: do detections, keypoints and events agree?

  python scripts/perf_compare.py out/off out/fast

Persons are matched per frame by best box IoU (>= 0.5). Reports match rate, box IoU and keypoint pixel error
(medians / p95 / max), confidence drift, count differences and whether the event streams are identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def compare(a_dir: Path, b_dir: Path) -> dict:
    A = [json.loads(l) for l in (a_dir / "frames.jsonl").open()]
    B = [json.loads(l) for l in (b_dir / "frames.jsonl").open()]
    n = min(len(A), len(B))
    ious, dconf, kerr, matched, total, count_diff, pdiff = [], [], [], 0, 0, 0, 0
    for fa, fb in zip(A[:n], B[:n]):
        total += len(fa["persons"])
        count_diff += len(fa["persons"]) != len(fb["persons"])
        pdiff += len(fa["products"]) != len(fb["products"])
        for p in fa["persons"]:
            best = max(fb["persons"], key=lambda q: _iou(p["bbox"], q["bbox"]), default=None)
            if best is None or _iou(p["bbox"], best["bbox"]) < 0.5:
                continue
            matched += 1
            ious.append(_iou(p["bbox"], best["bbox"]))
            dconf.append(abs(p["conf"] - best["conf"]))
            if p["kpts"] and best["kpts"]:
                ka, kb = np.array(p["kpts"]), np.array(best["kpts"])
                vis = (ka[:, 2] > 0.3) & (kb[:, 2] > 0.3)
                if vis.any():
                    kerr += list(np.linalg.norm(ka[vis, :2] - kb[vis, :2], axis=1))
    ea = (a_dir / "events.jsonl").read_text().splitlines()
    eb = (b_dir / "events.jsonl").read_text().splitlines()

    def ev(lines):
        return [(e["type"], e.get("person_id"), e.get("item")) for e in map(json.loads, lines)]
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")
    return {"frames": n, "persons_a": total, "matched": matched, "match_rate": round(matched / max(total, 1), 4),
            "frames_with_different_person_count": count_diff, "frames_with_different_product_count": pdiff,
            "box_iou_median": round(q(ious, 50), 4), "box_iou_p05": round(q(ious, 5), 4),
            "conf_abs_diff_p95": round(q(dconf, 95), 4), "conf_abs_diff_max": round(max(dconf, default=0.0), 4),
            "kpt_err_px_median": round(q(kerr, 50), 3), "kpt_err_px_p95": round(q(kerr, 95), 3),
            "kpt_err_px_max": round(max(kerr, default=0.0), 2),
            "events_a": len(ea), "events_b": len(eb), "events_identical": ev(ea) == ev(eb),
            "alerts_a": len((a_dir / "alerts.jsonl").read_text().splitlines()) if (a_dir / "alerts.jsonl").exists() else 0,
            "alerts_b": len((b_dir / "alerts.jsonl").read_text().splitlines()) if (b_dir / "alerts.jsonl").exists() else 0}


if __name__ == "__main__":
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
