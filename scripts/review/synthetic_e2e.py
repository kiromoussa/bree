"""End to end run of the review feedback loop on SYNTHETIC alerts. Prints one results table.

  .venv/bin/python scripts/review/synthetic_e2e.py [--out out/review_synthetic] [--n 60] [--seed 0] [--serve]

SYNTHETIC: drawn frames, random-number "reviewers", a made-up theft mix. The numbers show that
the loop runs and adds up. They say nothing about how well the vision pipeline detects theft.
With --serve the reviewer page stays up on the synthetic store so you can click through it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bree.review.labels import export_labels, retrain_hook  # noqa: E402
from bree.review.metrics import metrics, owner_report  # noqa: E402
from bree.review.store import ReviewStore  # noqa: E402
from bree.review.synthetic import populate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/review_synthetic")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    store, now = populate(out, a.n, a.seed)
    m = metrics(store)
    man = export_labels(store, now=now)
    again = export_labels(store, now=now)
    hook = retrain_hook(store, min_new=1, command=f"{sys.executable} -c pass", now=now)
    hook2 = retrain_hook(store, min_new=1, now=now)
    weeks = [owner_report(store, d, out / "reports") for d in ("2026-09-14", "2026-09-21", "2026-09-28")]
    clips = len(list((out / "alerts").glob("*.mp4")))
    # Retention: reopen the same store ten days later with a 10 day retention period.
    late = ReviewStore(out, retention_days=10, now=now + 10 * 86400)
    purged = sum(1 for x in late.alerts() if x["purged_ts"])
    clips_after = len(list((out / "alerts").glob("*.mp4")))
    man_after = json.loads(Path(man["path"]).read_text())

    o, ag = m["overall"], m["agreement"]
    rows = [
        ("alerts recorded", m["alerts_recorded"]), ("  from staged tests (left out below)", m["staged_alerts"]),
        ("  withdrawn by a late receipt (left out below)", m["retracted_alerts"]),
        ("customer alerts", o["sent"]), ("alerts reviewed", o["reviewed"]), ("review rate", o["review_rate"]),
        ("confirmed theft", o["confirmed_theft"]), ("theft, wrong item", o["wrong_item"]), ("not theft", o["not_theft"]),
        ("unclear", o["unclear"]), ("reviewers disagree", o["disputed"]),
        ("alert precision", o["precision"]), ("item precision", o["item_precision"]),
        ("median event to decision (s)", o["median_event_to_decision_s"]), ("median view time (s)", o["median_view_s"]),
        *[(f"precision {w}", b["precision"]) for w, b in m["by_week"].items()],
        *[(f"precision zone {z}", b["precision"]) for z, b in m["by_zone"].items()],
        *[(f"precision camera {c}", b["precision"]) for c, b in m["by_camera"].items()],
        ("alerts seen by two reviewers", ag["alerts_with_two_reviewers"]), ("same decision", ag["same_decision"]),
        ("percent agreement", ag["percent_agreement"]), ("Cohen's kappa", ag["cohens_kappa"]),
        ("detector examples", man["counts"]["detector"]),
        ("  of which class corrected", sum(1 for e in man["examples"] if e.get("corrected"))),
        ("pick examples", man["counts"]["pick"]), ("conceal examples", man["counts"]["conceal"]),
        ("  detector examples with an unknown class", man["counts"]["unknown_classes"]),
        ("duplicates skipped at export", man["counts"]["duplicates_skipped"]),
        ("new examples on a second export", again["counts"]["new_this_export"]),
        ("retrain hook: new examples, first call", hook["new_total"]),
        ("retrain hook: command exit code", hook["returncode"]),
        ("retrain hook: new examples, second call", hook2["new_total"]),
        *[(f"owner report {w['week_start']}: sent / staged / customer / precision",
           f"{w['alerts_sent']} / {w['alerts_from_staged_tests']} / {w['customer_alerts']['sent']} / {w['customer_alerts']['precision']}")
          for w in weeks],
        ("staged tests", sum(w["staged_tests"]["staged"] for w in weeks)),
        ("staged tests caught", sum(w["staged_tests"]["caught"] for w in weeks)),
        ("staged tests missed", sum(w["staged_tests"]["missed"] for w in weeks)),
        ("clips on disk before retention", clips), ("alerts purged after retention", purged),
        ("clips on disk after retention", clips_after), ("dataset examples after retention", len(man_after["examples"])),
    ]
    print(f"SYNTHETIC review loop, n={a.n}, seed={a.seed}  (not a measure of detection accuracy)")
    print("| Measure | Value |\n|---|---|")
    for k, v in rows:
        print(f"| {k} | {v} |")
    print(f"\nskipped at export: {json.dumps(man['counts']['skipped'])}")
    print(f"store: {out / 'review.sqlite'}\nmanifest: {man['path']}\nowner reports: {out / 'reports'}")
    if a.serve:
        from bree.review.server import serve
        shutil.rmtree(out, ignore_errors=True)
        store, _ = populate(out, a.n, a.seed, retention_days=36500)   # synthetic dates are in the past: keep the clips
        httpd = serve(store, port=8090)
        print(f"review page on the SYNTHETIC store: http://127.0.0.1:{httpd.server_address[1]}/ (reviewer id: anything new)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            httpd.shutdown()


if __name__ == "__main__":
    main()
