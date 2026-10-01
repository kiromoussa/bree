"""Shadow mode for the pilot: run the full pipeline on the store cameras and tell nobody.

Every would-be alert (tier "alert" or "review") is appended to `would_be_alerts.jsonl` with a
head-pixelated clip, the basket, confidence, whether concealment was seen and the ledger's
audit log. Nothing reaches staff: no console alerts, no live dashboard, no annotated video.
Kiro and the operator label each record on the dashboard's /review page; labels go to
`labels.jsonl` (append-only, latest label per alert wins). That is the first real labelled
data from the pilot.

Layout of `output_dir`:
  would_be_alerts.jsonl      one record per would-be alert, plus a retraction record (`retracts` =
                             the earlier id) when a late POS receipt lowers one
  labels.jsonl               one line per label click: id, label, reviewer, note, ts
  <camera>/<session>/        pipeline outputs per camera run (alerts/<id>.json + .mp4 clip, events.jsonl);
                             with `multicam:` one store/<session>/ for all cameras (one ledger)
  raw/<camera>/*.mp4         ONLY if record_raw.enabled: rolling UNBLURRED segments (contain faces)
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

LABELS = ("real_theft", "false_alert", "unsure")


@dataclass
class Camera:
    name: str
    rtsp_url: str          # rtsp:// URL (or a video file / webcam index to replay)
    store: str             # store layout YAML with this camera's zones


@dataclass
class ShadowConfig:
    cameras: list[Camera]
    output_dir: str
    pos_export_dir: str | None = None
    backend: str = "yolo"
    runtime: str = "pytorch"                                 # pytorch | onnx (see bree.edge.ort)
    review_port: int = 8080                                  # 0 = don't serve the review page
    ledger: dict = field(default_factory=dict)               # LedgerConfig overrides, e.g. exit_grace_s
    record_raw: dict = field(default_factory=dict)           # enabled / segment_minutes / retention_hours
    # None: each camera has its own ledger. A dict (YAML `multicam: true` or handoff settings
    # max_dist_m / max_gap_s): all cameras are one store, one person id across cameras, ONE ledger.
    multicam: dict | None = None


def load_shadow_config(path: str | Path) -> ShadowConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cams = [Camera(c["name"], str(c["rtsp_url"]), c["store"]) for c in raw["cameras"]]
    if len({c.name for c in cams}) != len(cams):
        raise ValueError("camera names must be unique")
    mc = raw.get("multicam")
    return ShadowConfig(cameras=cams, output_dir=raw.get("output_dir", "out/shadow"),
                        pos_export_dir=raw.get("pos_export_dir"), backend=raw.get("backend", "yolo"),
                        runtime=raw.get("runtime", "pytorch"),
                        review_port=int(raw.get("review_port", 8080)), ledger=raw.get("ledger") or {},
                        record_raw=raw.get("record_raw") or {},
                        multicam=({} if mc is True else dict(mc)) if mc else None)


MULTICAM_NAME = "store"   # "camera" name of the fused feed in record ids and folders


def would_be_record(alert, camera: str, session: str, out_dir: str | Path) -> dict:
    clip = alert.clip_path
    if clip:
        clip = str(Path(clip).resolve().relative_to(Path(out_dir).resolve()))
    return {
        "id": f"{camera}-{session}-{alert.alert_id}",
        # A late receipt lowered an earlier would-be alert: this record points at it.
        "retracts": f"{camera}-{session}-{alert.retracts}" if alert.retracts else None,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "camera": camera, "person_id": alert.person_id, "group": alert.group,
        "tier": alert.tier, "confidence": alert.confidence,
        "concealment_seen": any(i.concealed for i in alert.unpaid_items),
        "basket": alert.basket,
        "unpaid": [asdict(i) for i in alert.unpaid_items],
        "paid": alert.paid_items, "visited_register": alert.visited_register,
        "t_exit": round(alert.t_exit, 2), "clip": clip,
        "reasons": alert.reasons, "audit_log": alert.audit_log,
    }


class ShadowLog:
    """would_be_alerts.jsonl + labels.jsonl in one folder. Thread-safe appends (one per camera)."""

    def __init__(self, out_dir: str | Path):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.alerts_path = self.out / "would_be_alerts.jsonl"
        self.labels_path = self.out / "labels.jsonl"
        self.lock = threading.Lock()

    def _append(self, path: Path, rec: dict) -> None:
        with self.lock, path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def add_alert(self, rec: dict) -> None:
        self._append(self.alerts_path, rec)

    def alerts(self) -> list[dict]:
        return self._read(self.alerts_path)

    def latest_labels(self) -> dict[str, dict]:
        return {lab["id"]: lab for lab in self._read(self.labels_path)}

    def add_label(self, alert_id: str, label: str, reviewer: str, note: str = "") -> dict:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if not reviewer.strip():
            raise ValueError("reviewer name is required")
        if alert_id not in {a["id"] for a in self.alerts()}:
            raise ValueError(f"unknown alert {alert_id!r}")
        rec = {"id": alert_id, "label": label, "reviewer": reviewer.strip(), "note": note.strip(),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        self._append(self.labels_path, rec)
        return rec

    def labelled(self) -> list[dict]:
        """Every would-be alert with its latest label (or None) and its latest retraction (or
        None). Retraction records are folded into the alert they retract, not listed on their own."""
        labels = self.latest_labels()
        rows = self.alerts()
        retracted = {a["retracts"]: a for a in rows if a.get("retracts")}
        return [{**a, "label": labels.get(a["id"]), "retraction": retracted.get(a["id"])}
                for a in rows if not a.get("retracts")]

    def clip_file(self, alert_id: str) -> Path | None:
        for a in self.alerts():
            if a["id"] == alert_id and a.get("clip"):
                p = (self.out / a["clip"]).resolve()
                # Only pixelated alert clips inside the output folder, never raw segments.
                if p.is_file() and self.out.resolve() in p.parents and "alerts" in p.parts:
                    return p
        return None

    def summary(self) -> dict:
        """Counts per tier and label. Precision = real / (real + false); "unsure" is left out.
        Tiers are as first decided (what staff would have been told); `retracted` counts how many
        a late receipt later lowered."""
        rows = self.labelled()
        out = {"would_be_alerts": len(rows), "labelled": sum(1 for r in rows if r["label"]),
               "retracted": sum(1 for r in rows if r["retraction"])}
        for tier in ("alert", "review", "all"):
            sel = [r for r in rows if tier == "all" or r["tier"] == tier]
            n = {lab: sum(1 for r in sel if r["label"] and r["label"]["label"] == lab) for lab in LABELS}
            decided = n["real_theft"] + n["false_alert"]
            out[tier] = {"total": len(sel), **n, "unlabelled": len(sel) - sum(n.values()),
                         "precision": round(n["real_theft"] / decided, 3) if decided else None}
        return out


class RawRecorder:
    """Rolling raw segments for re-running the pipeline later. UNBLURRED: contains faces.

    Off by default. Never served by the review page. Segments older than the retention
    window are deleted every time a new segment starts.
    """

    def __init__(self, folder: Path, fps: float, segment_minutes: float = 5, retention_hours: float = 24):
        self.folder, self.fps = Path(folder), fps
        self.folder.mkdir(parents=True, exist_ok=True)
        self.segment_s, self.retention_s = segment_minutes * 60, retention_hours * 3600
        self.vw, self.t_roll = None, 0.0

    def write(self, img) -> None:
        import cv2
        now = time.time()
        if self.vw is None or now >= self.t_roll:
            self.close()
            for f in self.folder.glob("*.mp4"):
                if now - f.stat().st_mtime > self.retention_s:
                    f.unlink()
            h, w = img.shape[:2]
            path = self.folder / time.strftime("%Y%m%dT%H%M%S.mp4")
            self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.t_roll = now + self.segment_s
        self.vw.write(img)

    def close(self) -> None:
        if self.vw is not None:
            self.vw.release()
            self.vw = None


def _run_cameras(name: str, cams: list[Camera], cfg: ShadowConfig, log: ShadowLog, once: bool,
                 stop: threading.Event) -> None:
    """One ledger for `cams`: a single camera, or (multicam) every camera of the store."""
    from bree.cli import make_backend
    from bree.events.zones import load_store_config, merge_stores
    from bree.ledger.payments import FolderPayments
    from bree.pipeline import CameraInput, run_store
    stores = [load_store_config(c.store) for c in cams]
    store = merge_stores(stores)
    backend = make_backend(cfg.backend, store, runtime=cfg.runtime)
    inputs = [CameraInput(c.name, c.rtsp_url, s) for c, s in zip(cams, stores)]
    raw_cfg = cfg.record_raw
    while not stop.is_set():
        session = time.strftime("%Y%m%dT%H%M%S")
        payments = None
        if cfg.pos_export_dir:
            # Live: receipts already in the folder predate this run. Replay (--once): read them all.
            payments = FolderPayments(cfg.pos_export_dir, stream_start_wall=time.time(),
                                      terminals=store.terminals, skip_existing=not once)
        recorders = ({c.name: RawRecorder(Path(cfg.output_dir) / "raw" / c.name, s.fps,
                                          raw_cfg.get("segment_minutes", 5), raw_cfg.get("retention_hours", 24))
                      for c, s in zip(cams, stores)} if raw_cfg.get("enabled") else {})
        try:
            s = run_store(inputs, backend, Path(cfg.output_dir) / name / session,
                          payments=payments, save_video=False, log_frames=False, verbose=False,
                          ledger_overrides=cfg.ledger, handoff=cfg.multicam,
                          on_alert=lambda a: log.add_alert(would_be_record(a, name, session, cfg.output_dir)),
                          on_frame=(lambda fr, *_: recorders[fr.camera or cams[0].name].write(fr.image))
                          if recorders else None)
            print(f"[{name}] stream ended after {s.frames} frames, {len(s.alerts)} would-be alert(s)")
        except IOError:
            print(f"[{name}] cannot open the stream")   # not the URL: it may hold credentials
        finally:
            for r in recorders.values():
                r.close()
        if once:
            break
        stop.wait(10)   # camera dropped: reconnect with a new session


def run_shadow(cfg: ShadowConfig, once: bool = False) -> ShadowLog:
    log = ShadowLog(cfg.output_dir)
    httpd = None
    if cfg.review_port:
        from bree.dashboard.server import serve
        httpd = serve(None, "127.0.0.1", cfg.review_port, review=log)
        print(f"review page: http://127.0.0.1:{httpd.server_address[1]}/review")
    if cfg.record_raw.get("enabled"):
        print("WARNING: record_raw is on. Raw segments are NOT head-pixelated and contain faces: "
              f"{Path(cfg.output_dir) / 'raw'} (kept {cfg.record_raw.get('retention_hours', 24)} h)")
    stop = threading.Event()
    groups = ([(MULTICAM_NAME, cfg.cameras)] if cfg.multicam is not None
              else [(c.name, [c]) for c in cfg.cameras])
    threads = [threading.Thread(target=_run_cameras, args=(name, cams, cfg, log, once, stop), daemon=True)
               for name, cams in groups]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
    if httpd is not None:
        httpd.shutdown()
    return log
