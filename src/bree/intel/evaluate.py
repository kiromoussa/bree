"""Score door events against door-event labels.

Labels (REAL-MODEL-LABELLED, see docs/ultra/intel.md): {"segments": {"<segment stem>": {"split": "dev"|"heldout",
"events": [{"t": seconds, "dir": "in"|"out", "who": free text}], "checked": [[t0, t1], ...]}}}. `checked` are the windows a labeller
looked at; events outside them are assumed to be none. A system event matches a label of the same direction within `tol_s`
seconds, one-to-one (greedy by time distance). Reported per split: true positives, false positives, misses, precision, recall,
the count error (system entries/exits minus labelled), and the mean absolute timing error of the matches.
"""
from __future__ import annotations

import json
from pathlib import Path


def match(sys_events: list[dict], labels: list[dict], tol_s: float = 4.0):
    pairs = sorted(((abs(s["t"] - l["t"]), i, j) for i, s in enumerate(sys_events) for j, l in enumerate(labels)
                    if s["dir"] == l["dir"] and abs(s["t"] - l["t"]) <= tol_s), key=lambda p: p[0])
    used_s, used_l, tp = set(), set(), []
    for d, i, j in pairs:
        if i in used_s or j in used_l:
            continue
        used_s.add(i)
        used_l.add(j)
        tp.append((i, j, d))
    fp = [sys_events[i] for i in range(len(sys_events)) if i not in used_s]
    fn = [labels[j] for j in range(len(labels)) if j not in used_l]
    return tp, fp, fn


def score(records_by_seg: dict[str, list[dict]], labels: dict, tol_s: float = 4.0) -> dict:
    """records_by_seg: {segment stem: flow records of that segment}. Returns {split: {...}, "detail": {...}}."""
    out: dict = {"tol_s": tol_s, "detail": {}}
    acc: dict[str, dict] = {}
    for seg, lab in labels["segments"].items():
        if seg not in records_by_seg:
            continue
        sysd = [r for r in records_by_seg[seg] if r["type"] == "door"]
        tp, fp, fn = match(sysd, lab["events"], tol_s)
        a = acc.setdefault(lab["split"], {"tp": 0, "fp": 0, "fn": 0, "labelled_in": 0, "labelled_out": 0, "sys_in": 0, "sys_out": 0,
                                         "dt": [], "via": {}})
        a["tp"] += len(tp)
        a["fp"] += len(fp)
        a["fn"] += len(fn)
        a["labelled_in"] += sum(e["dir"] == "in" for e in lab["events"])
        a["labelled_out"] += sum(e["dir"] == "out" for e in lab["events"])
        a["sys_in"] += sum(e["dir"] == "in" for e in sysd)
        a["sys_out"] += sum(e["dir"] == "out" for e in sysd)
        a["dt"] += [d for _, _, d in tp]
        for i, _, _ in tp:
            v = sysd[i]["via"]
            a["via"][v] = a["via"].get(v, 0) + 1
        out["detail"][seg] = {"split": lab["split"], "tp": len(tp), "fp": [(e["dir"], e["t"], e["via"]) for e in fp],
                              "fn": [(e["dir"], e["t"], e.get("who", "")) for e in fn]}
    for split, a in acc.items():
        n_s, n_l = a["tp"] + a["fp"], a["tp"] + a["fn"]
        out[split] = {"true_pos": a["tp"], "false_pos": a["fp"], "missed": a["fn"],
                      "precision": round(a["tp"] / n_s, 3) if n_s else None, "recall": round(a["tp"] / n_l, 3) if n_l else None,
                      "labelled": {"in": a["labelled_in"], "out": a["labelled_out"]}, "system": {"in": a["sys_in"], "out": a["sys_out"]},
                      "mean_abs_time_err_s": round(sum(a["dt"]) / len(a["dt"]), 2) if a["dt"] else None, "matched_via": a["via"]}
    return out


def by_segment(records: list[dict]) -> dict[str, list[dict]]:
    """Split a flow.jsonl record list on its "segment" records (video file stem as key)."""
    out: dict[str, list[dict]] = {}
    cur = None
    for r in records:
        if r["type"] == "segment":
            cur = Path(r["video"]).stem
            out[cur] = []
        elif cur is not None:
            out[cur].append(r)
    return out


def load_labels(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
