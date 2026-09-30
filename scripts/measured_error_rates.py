"""Turn the real-data measurements into VisionNoise parameters -> results/measured_error_rates.json.

Only parameters with a real measurement get `measured: true`; the rest keep the assumed value and say why.
`make bench` re-runs the simulator with these ("measured" noise row) next to the assumed baseline.
"""
import json
from dataclasses import asdict
from pathlib import Path

from bree.sim.events_sim import VisionNoise

R = Path("results")
load = lambda n: json.loads((R / f"{n}.json").read_text()) if (R / f"{n}.json").exists() else None
merl, rs = load("merl_measure"), load("conceal_retails")
TH = "0.5"   # classifier operating threshold, fixed before looking at any evaluation set
assumed = asdict(VisionNoise())
out = {k: {"assumed": v, "value": v, "measured": False, "source": "not measured: no labelled real data for this yet"}
       for k, v in assumed.items()}


def put(k, value, source):
    out[k].update(value=float(value), measured=True, source=source)


if merl:
    put("p_pick_detected", merl["reach_recall"],
        f"UPPER BOUND. MERL Shopping test split ({merl['n_videos']} real videos, overhead camera): share of "
        f"{merl['reach_instances']} labelled Reach To Shelf + Hand In Shelf instances where our pose model put a wrist in the shelf zone. A pick also "
        f"needs the product detected, which MERL can't measure (its products aren't COCO classes).")
    splits = sum(1 for v in merl["per_video"] if v["person_track_ids"] > 1) / merl["n_videos"]
    put("p_id_switch", splits,
        f"MERL test split: share of single-shopper videos (~2 min each) whose shopper got more than one track id. "
        f"Overhead view, harder than an angled ceiling camera, so likely pessimistic for a real store.")
if rs:
    e = rs["staged"]["model"]["events"][TH]
    put("p_conceal_detected", e["pos_hit"] / max(e["pos"], 1),
        f"RetailS staged test (same store as PoseLift, clips never trained on, evaluation only): {e['pos_hit']}/{e['pos']} "
        f"shoplifting clips where the classifier (threshold {TH}) triggered on a labelled frame. Labels are per frame, "
        f"not per person, so any person's trigger counts. (PoseLift chain-grouped CV is in-distribution and not used here.)")
    n = rs["normal"]["model"]
    put("p_false_conceal", n["track_trigger_rate_at"][TH],
        f"RetailS normal footage ({rs['normal']['hours']:.1f} h, real shoppers, evaluation only): share of "
        f"{n['tracks_2s_plus']} person tracks (>= 2 s, split at frame gaps) with a classifier trigger at threshold {TH}. "
        f"A shopper can be several tracks, so per shopper it is likely higher; the simulator draws it per carried "
        f"item, so shoppers carrying nothing never false-alarm there. A stand-in, not a like-for-like rate.")

for k in ("p_putback_detected", "p_register_visit_detected", "p_exit_detected", "p_visible_at_exit",
          "p_category_confusion", "p_crowd_swap"):
    out[k]["source"] = "not measured: needs store footage with these events labelled (pilot shadow mode or staged session)"
for k in ("p_pos_dropped", "pos_jitter_s"):
    out[k]["source"] = "not measured: needs the operator's POS export"

R.mkdir(exist_ok=True)
(R / "measured_error_rates.json").write_text(json.dumps({"vision_noise": out}, indent=1))
print(json.dumps({k: v["value"] for k, v in out.items() if v["measured"]}, indent=1))
