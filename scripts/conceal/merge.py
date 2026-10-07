"""Add up eval.py results of clip sets that live in different folders (counts summed, intervals recomputed).
  python scripts/conceal/merge.py out/conceal/heldout_existing.json out/conceal/heldout_4961.json ... --json results/conceal_cue_heldout.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eval import wilson

CI = {"thieves_at_alert": "thieves", "honest_at_alert": "honest_shoppers", "thieves_with_a_cue": "thieves", "honest_with_a_cue": "honest_shoppers"}


def merge(parts: list[dict]) -> dict:
    out: dict = {}
    for v in parts[0]["summary"]:
        s: dict = {}
        for p in parts:
            for k, x in p["summary"][v].items():
                if k.endswith("_ci95") or k == "alert_precision":
                    continue
                if isinstance(x, dict) and k == "thieves_not_at_alert_because":
                    s[k] = {r: s.get(k, {}).get(r, 0) + n for r, n in {**dict.fromkeys(s.get(k, {}), 0), **x}.items()}
                else:
                    s[k] = x if k not in s else {**s[k], **x} if isinstance(x, dict) else s[k] + x
        for k, n in CI.items():
            if k in s:
                s[k + "_ci95"] = wilson(s[k], s[n])
        if "thefts_alerted" in s:
            fa = s["false_alerts_on_honest_shoppers"] + s["false_alerts_on_nobody"]
            s["alert_precision"] = round(s["thieves_at_alert"] / (s["thieves_at_alert"] + fa), 3) if s["thieves_at_alert"] + fa else None
        out[v] = s
    return {"what": parts[0]["what"] + " Several clip sets added up (scripts/conceal/merge.py); alert_precision here is per shopper.",
            "clips": [p["clips"] for p in parts], "seeds": [x for p in parts for x in p["seeds"]], "summary": out,
            "thefts": {v: [t for p in parts for t in p["thefts"][v]] for v in parts[0]["thefts"]}, "alerts": {v: [t for p in parts for t in p["alerts"][v]] for v in parts[0]["alerts"]},
            "truth_thieves": [t for p in parts for t in p["truth_thieves"]]}


if __name__ == "__main__":
    i = sys.argv.index("--json")
    res = merge([json.loads(Path(f).read_text()) for f in sys.argv[1:i]])
    Path(sys.argv[i + 1]).write_text(json.dumps(res, indent=1))
    for v, s in res["summary"].items():
        print(v, json.dumps(s))
