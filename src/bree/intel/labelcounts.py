"""Adapters for the agent-made people-count labels of the other channels (REAL-MODEL-LABELLED, not ground truth), and a scorer.

Each label file has its own layout; `load_counts` turns any of them into rows {"split", "segment", "t", "count", "xy"} (people inside the
store, box centres normalised 0-1). `score_counts` compares them with the people the flow engine had in view at that moment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_counts(path: str | Path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    rows = []
    ch = Path(path).name
    if "sets" in d:                                         # ch10
        for name, s in d["sets"].items():
            split = "heldout" if name == "held_out" else "dev"
            seg = Path(s["segment"]).stem
            for f in s["frames"]:
                rows.append({"split": split, "segment": seg, "t": f["t_s"], "count": f["n_people"], "xy": [[p["cx"], p["cy"]] for p in f["people"]]})
        return rows
    for f in d["frames"]:
        if "visible_people_inside" in f:                    # ch12
            rows.append({"split": f["split"], "segment": f["segment"], "t": f["t_s"], "count": f["visible_people_inside"], "xy": f["centres_xy"]})
        elif "person_count" in f:                           # ch14
            rows.append({"split": f["split"], "segment": Path(f["segment"]).stem, "t": f["t_s"], "count": f["person_count"],
                         "xy": [[p["cx"], p["cy"]] for p in f["people"]]})
        elif "visible_people_in_store" in f:                # ch15 (held out: segment B)
            rows.append({"split": "heldout" if f["segment"].endswith("185128") else "dev", "segment": f["segment"], "t": f["t_s"],
                         "count": f["visible_people_in_store"], "xy": f["person_centres_xy_norm"]})
        elif "visible_persons" in f:                        # ch29
            rows.append({"split": f["split"], "segment": f["segment"], "t": f["t_s"], "count": f["visible_persons"], "xy": f["centres_xy"]})
    return rows


def score_counts(frame_logs: dict[str, list], rows: list[dict], size: tuple[int, int], include_staff: bool = True) -> dict:
    """frame_logs: {segment: engine.frame_log}. Per split: exact-count rate, within-one rate, presence agreement, mean absolute error,
    and the totals (system vs labelled people)."""
    acc: dict[str, dict] = {}
    for r in rows:
        log = frame_logs.get(r["segment"])
        if not log:
            continue
        ft = np.array([t for t, _ in log])
        i = int(np.argmin(abs(ft - r["t"])))
        n = len([1 for _, role, _, _ in log[i][1] if include_staff or role == "customer"])
        a = acc.setdefault(r["split"], {"n": 0, "exact": 0, "within1": 0, "presence": 0, "abs": 0, "sys": 0, "lab": 0})
        a["n"] += 1
        a["exact"] += n == r["count"]
        a["within1"] += abs(n - r["count"]) <= 1
        a["presence"] += (n > 0) == (r["count"] > 0)
        a["abs"] += abs(n - r["count"])
        a["sys"] += n
        a["lab"] += r["count"]
    return {s: {"frames": a["n"], "exact": round(a["exact"] / a["n"], 3), "within1": round(a["within1"] / a["n"], 3),
                "presence": round(a["presence"] / a["n"], 3), "mae": round(a["abs"] / a["n"], 2), "system_total": a["sys"], "labelled_total": a["lab"]}
            for s, a in acc.items()}
