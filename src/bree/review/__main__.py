"""python -m bree.review <command> --store <folder>

  ingest   add alerts from a pipeline alerts.jsonl or a shadow-mode would_be_alerts.jsonl
  serve    the local reviewer page (127.0.0.1)
  staged   record a staged test theft
  export   decisions -> training examples + manifest.json
  metrics  precision over time, review rate, time to decision, per zone / camera, agreement
  report   weekly owner report (markdown + JSON)
  retrain  the retrain hook: export, then run a training command if enough examples are new
  purge    delete media past the retention period now (also runs on every open)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

from bree.review.labels import export_labels, retrain_hook
from bree.review.metrics import metrics, owner_report
from bree.review.store import ReviewStore, frames_from_log


def _find_clip(clip: str, src: Path) -> str | None:
    """A relative clip path, tried against the alerts file's folder (shadow records), the current
    folder, then every folder above the file (pipeline alerts name clips relative to where the run
    was started)."""
    here = src.resolve().parent
    for base in (here, Path.cwd(), *here.parents):
        if (base / clip).is_file():
            return str((base / clip).resolve())
    return None


def ingest(store: ReviewStore, alerts_file: str, camera: str | None = None, frames_log: str | None = None) -> dict:
    """Add the records of one alerts.jsonl / would_be_alerts.jsonl. Returns counts and warnings."""
    src = Path(alerts_file)
    if not src.is_file():
        raise SystemExit(f"ingest: no alerts file at {src} (make demo writes out/demo/<clip>/alerts.jsonl, "
                         "for example out/demo/walkout/alerts.jsonl)")
    recs = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    run, ids, warnings = src.resolve().parent.name, {}, []
    stored = sorted(store.alerts(), key=lambda a: a["created_ts"])
    known = {a["alert_id"] for a in stored}
    for r in recs:
        # Plain pipeline ids (A00001) restart every run, and a run folder can be reused. The id gets the
        # folder name plus a hash of the record: the same record twice is one alert, a different one is new.
        if "id" not in r:
            h = hashlib.sha1(json.dumps(r, sort_keys=True).encode()).hexdigest()[:8]
            ids[r["alert_id"]] = f"{run}-{r['alert_id']}-{h}"
    new = missing = 0
    for r in recs:
        if "id" not in r:
            r["id"] = ids[r["alert_id"]]
            if r.get("retracts"):
                # The original is normally in this file. If not (the file was split, or only the tail was
                # ingested), take the newest stored alert of this run folder with that number.
                old = [a["alert_id"] for a in stored if a["alert_id"].startswith(f"{run}-{r['retracts']}-")]
                target = ids.get(r["retracts"]) or (old[-1] if old else None)
                if target is None:
                    warnings.append(f"{r['id']}: retracts {r['retracts']}, which is not in this file or the store. Skipped.")
                    r["skip"] = True
                r["retracts"] = target
                continue
            clash = [k for k in known if k.startswith(f"{run}-{r['alert_id']}-") and k != r["id"]]
            if clash:
                warnings.append(f"{r['id']}: the store already has {clash[0]} from a run folder with the same name. "
                                "Stored as a new alert. If the clip file was overwritten, the older alert now shows the new clip.")
        if r.get("retracts") or r.get("skip"):
            continue
        key = "clip_path" if r.get("clip_path") else "clip"
        if r.get(key) and not Path(r[key]).is_absolute():
            found = _find_clip(r[key], src)
            if found is None:
                missing += 1
                warnings.append(f"{r['id']}: clip {r[key]} not found; the page will show no clip for it")
            r[key] = found or str((src.resolve().parent / r[key]))
        frames = frames_from_log(frames_log, r) if frames_log else None
        new += store.add_alert(r, camera=camera, frames=frames) is not None
    retracted = 0
    for r in recs:
        if r.get("retracts"):
            if store.add_alert(r) is None:
                warnings.append(f"{r['id']}: retracts {r['retracts']}, which is not in the store. Skipped.")
            else:
                retracted += 1
    return {"new": new, "records": len(recs), "retractions": retracted, "clips_missing": missing, "warnings": warnings}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bree.review", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="out/review", help="review store folder (default out/review)")
    ap.add_argument("--retention-days", type=float, default=None, help="set the retention period (kept in the store)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("alerts", help="alerts.jsonl or would_be_alerts.jsonl")
    p.add_argument("--camera", default=None, help="camera name, if the records do not carry one")
    p.add_argument("--frames-log", default=None, help="the run's frames.jsonl: adds pose windows for pick / conceal labels")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8090)
    p = sub.add_parser("staged")
    p.add_argument("test_id")
    p.add_argument("--time", default=None, help="local time YYYY-MM-DDTHH:MM:SS (default: now)")
    p.add_argument("--camera", default=None)
    p.add_argument("--zone", default=None)
    p.add_argument("--item", default=None)
    p = sub.add_parser("export")
    p.add_argument("--out", default=None, help="dataset folder (default <store>/datasets/review)")
    sub.add_parser("metrics")
    p = sub.add_parser("report")
    p.add_argument("--week-start", default=None, help="YYYY-MM-DD (UTC); default: seven days ago, so the report "
                   "covers the seven full UTC days ending yesterday and leaves today out")
    p.add_argument("--out", default=None, help="folder for the .md and .json (default <store>/reports)")
    p = sub.add_parser("retrain")
    p.add_argument("--out", default=None)
    p.add_argument("--min-new", type=int, default=50, help="new examples needed before the command runs")
    p.add_argument("--command", default=None, help="training command; gets the manifest path as its last argument")
    sub.add_parser("purge")
    a = ap.parse_args(argv)

    store = ReviewStore(a.store, retention_days=a.retention_days)
    if a.cmd == "ingest":
        res = ingest(store, a.alerts, a.camera, a.frames_log)
        for w in res["warnings"]:
            print("WARNING:", w, file=sys.stderr)
        print(f"{res['new']} new alert(s) of {res['records']} record(s), {res['retractions']} retraction(s) applied, "
              f"{res['clips_missing']} clip(s) missing -> {store.dir / 'review.sqlite'}")
    elif a.cmd == "serve":
        from bree.review.server import serve
        httpd = serve(store, port=a.port)
        print(f"review page: http://127.0.0.1:{httpd.server_address[1]}/  (this machine only; Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            httpd.shutdown()
    elif a.cmd == "staged":
        ts = time.mktime(time.strptime(a.time, "%Y-%m-%dT%H:%M:%S")) if a.time else time.time()
        store.add_staged_test(a.test_id, ts, a.camera, a.zone, a.item)
        print(f"staged test {a.test_id} recorded")
    elif a.cmd == "export":
        m = export_labels(store, a.out)
        print(json.dumps({"manifest": m["path"], "classes": m["classes"], "counts": m["counts"]}, indent=2))
    elif a.cmd == "metrics":
        print(json.dumps(metrics(store), indent=2))
    elif a.cmd == "report":
        week = a.week_start or (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=7)).isoformat()
        out = a.out or store.dir / "reports"
        owner_report(store, week, out)
        print((Path(out) / f"owner_report_{week}.md").read_text())
    elif a.cmd == "retrain":
        print(json.dumps(retrain_hook(store, a.out, a.min_new, a.command), indent=2))
    elif a.cmd == "purge":
        print(json.dumps(store.purge()))


if __name__ == "__main__":
    main()
