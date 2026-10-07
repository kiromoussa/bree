"""The concealment cue through the ledger, on rendered clips with truth (SIMULATED), at shopper level.

  .venv/bin/python scripts/conceal/eval.py data/synth/conceal/heldout 4960 4961 ... --out out/conceal/runs/heldout \\
      [--stage out/bench/train --looks out/conceal/train] --json results/conceal_cue_heldout.json

Per clip: the shelf and people stages run once (or are copied from --stage), then the join and ledger run three times:
cue off, cue on, and with the true concealment times as cues (the most this ledger could do with a perfect cue; it
reads truth, so it is a bound and not a pipeline result). The records of "on" and "oracle" are also scored after
bree.concealment.tier.retier (on_tier, oracle_tier; the two *_any_identity drop the identity condition, to measure what it costs).
Scored with bree.sim.bench.score_clip.
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from bree.concealment.run import run  # noqa: E402
from bree.sim.bench import aggregate, score_clip  # noqa: E402
from rows import HIDE_BEFORE_S, truth_of  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def oracle(clip: Path):
    tracks, events, _ = truth_of(clip)

    def cues(acts, people, who):
        out = []
        for e in events:
            if e.get("tConceal") is None:
                continue
            xz = (tracks.get(round(e["tConceal"], 1)) or {}).get(e["shopper"])
            near = sorted((math.hypot(got[0][0] - xz[0], got[0][1] - xz[1]), p.id) for p in people if xz and not p.staff and (got := p.at(e["tConceal"])) is not None)
            if near and near[0][0] <= 1.0:
                out.append({"t": e["tConceal"] - HIDE_BEFORE_S, "person_id": near[0][1], "sku_id": None, "conf": 0.9, "source": "ORACLE"})
        return out
    return cues


def cue_level(clip: Path, out: Path) -> dict:
    """Who got a cue, whatever the ledger then did with it: {shopper: cues}, by the truth track under the cue's place."""
    from rows import shopper_at
    tracks, _, _ = truth_of(clip)
    got: dict = {}
    for line in (out / "pipeline/conceal_cues.jsonl").read_text().splitlines():
        c = json.loads(line)
        name = shopper_at(tracks, c["at"], c["t"] - 0.3)[0] if c.get("at") else None
        got[name] = got.get(name, 0) + 1
    return got


def why_not(clip_res: dict, out: Path) -> dict:
    """For each thief of one scored clip that did not reach alert tier: what the ledger said. {shopper: reason}"""
    reasons = {}
    for line in (out / "pipeline/alerts.jsonl").read_text().splitlines():
        a = json.loads(line)
        reasons[a["alert_id"]] = [r for r in a.get("reasons", []) if r.startswith("capped at review")]
    got = {}
    for name in {t["shopper"] for t in clip_res["thefts"]}:
        if any(t["alert"] for t in clip_res["thefts"] if t["shopper"] == name):
            continue
        rows = [a for a in clip_res["alerts"] if a["shopper"] == name]
        caps = sorted({r.split(" (")[0].replace("capped at review: ", "") for a in rows for r in reasons.get(a["alert_id"], [])})
        caps = ["a paid item did not match the basket" if "paid item" in c else c for c in caps]
        got[name] = "no record with an unpaid item under this shopper" if not rows else "; ".join(caps) if caps else "review: score under the alert bar, nothing marked concealed"
    return got


def derived(base: Path, out: Path, scores, bar: float, ignore_identity: bool = False) -> None:
    """A run folder whose records are those of `base` after bree.concealment.tier.retier; every other file is linked."""
    from bree.concealment.tier import retier
    (out / "pipeline").mkdir(parents=True, exist_ok=True)
    for f in (base / "pipeline").iterdir():
        if f.is_file() and f.name != "alerts.jsonl" and not (out / "pipeline" / f.name).exists():
            (out / "pipeline" / f.name).symlink_to(f.resolve())
    rows = [json.loads(x) for x in (base / "pipeline/alerts.jsonl").read_text().splitlines() if x]
    (out / "pipeline/alerts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in retier(rows, scores, bar, ignore_identity)))


def why_not_d(clip_res: dict, out: Path, scores, bar: float) -> dict:
    """For each thief not at alert tier under the tier rule: which of its three conditions failed. {shopper: reason}"""
    from bree.concealment.tier import identity_uncertain
    recs = {a["alert_id"]: a for a in (json.loads(x) for x in (out / "pipeline/alerts.jsonl").read_text().splitlines() if x)}
    got = {}
    for name in {t["shopper"] for t in clip_res["thefts"]}:
        if any(t["alert"] for t in clip_res["thefts"] if t["shopper"] == name):
            continue
        rows = [recs[a["alert_id"]] for a in clip_res["alerts"] if a["shopper"] == name and a["alert_id"] in recs]
        marked = [a for a in rows if any(i.get("concealed") for i in a["unpaid_items"])]
        got[name] = ("no record with an unpaid item under this shopper" if not rows else "unpaid, nothing marked concealed" if not marked
                     else "identity uncertain" if all(identity_uncertain(a) for a in marked) else "shopper score under the bar")
    return got


def shopper_level(clips: list[dict]) -> dict:
    thieves = {(c["clip"], t["shopper"]) for c in clips for t in c["thefts"]}
    honest = {(c["clip"], n) for c in clips for n in c["honest"]}
    alerted = {(c["clip"], t["shopper"]) for c in clips for t in c["thefts"] if t["alert"]}
    flagged = {(c["clip"], t["shopper"]) for c in clips for t in c["thefts"] if t["alert_or_review"]}
    h_alert = {(a["clip"], a["shopper"]) for c in clips for a in c["alerts"] if a["tier"] == "alert" and (a["clip"], a["shopper"]) in honest}
    h_review = {(a["clip"], a["shopper"]) for c in clips for a in c["alerts"] if (a["clip"], a["shopper"]) in honest}
    return {"thieves": len(thieves), "thieves_at_alert": len(alerted), "thieves_at_alert_ci95": wilson(len(alerted), len(thieves)),
            "thieves_at_alert_or_review": len(flagged), "honest_shoppers": len(honest), "honest_at_alert": len(h_alert), "honest_at_alert_ci95": wilson(len(h_alert), len(honest)),
            "honest_at_alert_or_review": len(h_review), "alerts_on_nobody": sum(1 for c in clips for a in c["alerts"] if a["tier"] == "alert" and a["shopper"] in (None, "clerk")),
            "who_alerted": sorted(f"{c}:{s}" for c, s in alerted), "honest_alerted": sorted(f"{c}:{s}" for c, s in h_alert),
            "thieves_not_at_alert_because": dict(sorted(__import__("collections").Counter(r for c in clips for r in c.get("why_not_alert", {}).values()).items())),
            "thieves_not_at_alert": {f"{c['clip']}:{n}": r for c in clips for n, r in sorted(c.get("why_not_alert", {}).items())}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", help="folder with clip_<seed> folders")
    ap.add_argument("seeds", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", help="folder with clip_<seed>/pipeline of an earlier run: its shelf events and person boxes are copied")
    ap.add_argument("--looks", help="folder with clip_<seed>/looks_*.jsonl of an earlier scan: copied")
    ap.add_argument("--json")
    ap.add_argument("--variants", default="off,on,oracle", help="on and oracle also give on_tier, oracle_tier and the two *_any_identity (bree.concealment.tier on the same records)")
    a = ap.parse_args()
    from bree.concealment.cue import ConcealConfig
    BAR = ConcealConfig().shopper_bar
    res = {v: [] for v in sorted(a.variants.split(","), key=lambda v: v != "on")}      # "on" first: its folder holds the stages the others copy
    extra = {"on": [("on_tier", False), ("on_tier_any_identity", True)], "oracle": [("oracle_tier", False), ("oracle_tier_any_identity", True)]}
    for v in list(res):
        res.update({name: [] for name, _ in extra.get(v, [])})
    for s in a.seeds:
        clip = Path(a.clips) / f"clip_{s}"
        base = Path(a.out) / "on" / f"clip_{s}"
        (base / "pipeline/conceal").mkdir(parents=True, exist_ok=True)
        if a.stage and not (base / "pipeline/shelf_events.jsonl").exists():
            for f in (Path(a.stage) / f"clip_{s}/pipeline").glob("*.jsonl"):
                if f.name.startswith(("people_", "shelf_events", "shelf_status")):
                    shutil.copy(f, base / "pipeline" / f.name)
        if a.looks:
            for f in (Path(a.looks) / f"clip_{s}").glob("looks_*.jsonl"):
                if not (base / "pipeline/conceal" / f.name).exists():
                    shutil.copy(f, base / "pipeline/conceal" / f.name)
        for v in [v for v in res if "_tier" not in v]:
            out = Path(a.out) / v / f"clip_{s}"
            if v != "on":       # the stored stages are shared: only the join and the ledger differ
                (out / "pipeline").mkdir(parents=True, exist_ok=True)
                for f in (base / "pipeline").glob("*.jsonl"):
                    if f.name.startswith(("people_", "shelf_events", "shelf_status")) and not (out / "pipeline" / f.name).exists():
                        shutil.copy(f, out / "pipeline" / f.name)
            info = run(clip, out, cues=None if v == "on" else oracle(clip) if v == "oracle" else (lambda *_: []))
            res[v].append(score_clip(clip, out))
            res[v][-1]["why_not_alert"] = why_not(res[v][-1], out)
            if v == "on":
                res[v][-1]["cued"] = cue_level(clip, out)
            print(f"clip {s} {v}: cues {info['conceal_cues']}, thefts at alert {sum(t['alert'] for t in res[v][-1]['thefts'])} of {len(res[v][-1]['thefts'])}", flush=True)
            for name, any_identity in extra.get(v, []):       # the same records under the tier rule of bree.concealment.tier
                scores = json.loads((out / "pipeline/conceal_scores.json").read_text()) if v == "on" else None
                d = Path(a.out) / name / f"clip_{s}"
                derived(out, d, scores, BAR, any_identity)
                res[name].append(score_clip(clip, d))
                res[name][-1]["why_not_alert"] = why_not_d(res[name][-1], d, scores, BAR)
                print(f"clip {s} {name}: thefts at alert {sum(t['alert'] for t in res[name][-1]['thefts'])} of {len(res[name][-1]['thefts'])}", flush=True)
    summary = {}
    for v, clips in res.items():
        agg = aggregate(clips)["scorecard"]
        if v == "on":      # before the ledger: which shoppers got a cue at all
            thieves = {(c["clip"], t["shopper"]) for c in clips for t in c["thefts"]}
            cued = {(c["clip"], n) for c in clips for n in c["cued"]}
            honest = {(c["clip"], n) for c in clips for n in c["honest"]}
            summary["cue_before_ledger"] = {"thieves": len(thieves), "thieves_with_a_cue": len(cued & thieves), "thieves_with_a_cue_ci95": wilson(len(cued & thieves), len(thieves)),
                                            "honest_shoppers": len(honest), "honest_with_a_cue": len(cued & honest), "honest_with_a_cue_ci95": wilson(len(cued & honest), len(honest)),
                                            "cues_on_no_shopper": sum(c["cued"].get(None, 0) for c in clips), "thieves_cued": sorted(f"{a}:{b}" for a, b in cued & thieves),
                                            "honest_cued": sorted(f"{a}:{b}" for a, b in cued & honest)}
            print("cue_before_ledger", json.dumps(summary["cue_before_ledger"]))
        summary[v] = {**shopper_level(clips), **{k: agg[k] for k in ("stolen_items", "thefts_alerted", "thefts_alerted_or_reviewed", "false_alerts_on_honest_shoppers", "false_alerts_on_nobody", "reviews_on_honest_shoppers", "alert_precision")}}
        print(v, json.dumps(summary[v]))
    if a.json:
        Path(a.json).write_text(json.dumps({"what": "SIMULATED clips. Concealment cue through the ledger: off, on, and with the true concealment times as cues (a bound that reads truth).",
                                            "clips": a.clips, "seeds": a.seeds, "summary": summary, "thefts": {v: [t for c in clips for t in c["thefts"]] for v, clips in res.items()},
                                            "alerts": {v: [x for c in clips for x in c["alerts"]] for v, clips in res.items()},
                                            "truth_thieves": sorted(f"{c['clip']}:{n}" for c in next(iter(res.values())) for n in {t["shopper"] for t in c["thefts"]})}, indent=1))
