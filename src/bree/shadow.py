"""Shadow mode for the pilot: run the full pipeline on the store cameras and tell nobody.

Every would-be alert (tier "alert" or "review") is appended to `would_be_alerts.jsonl` with a
head-pixelated clip, the basket, confidence, whether concealment was seen and the ledger's
audit log. Nothing reaches staff: no console alerts, no live dashboard, no annotated video.
Kiro and the operator label each record on the dashboard's /review page; labels go to
`labels.jsonl` (append-only, latest label per alert wins). That is the first real labelled
data from the pilot.

Layout of `output_dir`:
  would_be_alerts.jsonl      one record per would-be alert
  labels.jsonl               one line per label click: id, label, reviewer, note, ts
  <camera>/<session>/        pipeline outputs per camera run (alerts/<id>.json + .mp4 clip, events.jsonl)
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


def load_shadow_config(path: str | Path) -> ShadowConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cams = [Camera(c["name"], str(c["rtsp_url"]), c["store"]) for c in raw["cameras"]]
    if len({c.name for c in cams}) != len(cams):
        raise ValueError("camera names must be unique")
    return ShadowConfig(cameras=cams, output_dir=raw.get("output_dir", "out/shadow"),
                        pos_export_dir=raw.get("pos_export_dir"), backend=raw.get("backend", "yolo"),
                        runtime=raw.get("runtime", "pytorch"),
                        review_port=int(raw.get("review_port", 8080)), ledger=raw.get("ledger") or {},
                        record_raw=raw.get("record_raw") or {})


def would_be_record(alert, camera: str, session: str, out_dir: str | Path) -> dict:
    clip = alert.clip_path
    if clip:
        clip = str(Path(clip).resolve().relative_to(Path(out_dir).resolve()))
    return {
        "id": f"{camera}-{session}-{alert.alert_id}",
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
        """Every would-be alert with its latest label (or None)."""
        labels = self.latest_labels()
        return [{**a, "label": labels.get(a["id"])} for a in self.alerts()]

    def clip_file(self, alert_id: str) -> Path | None:
        for a in self.alerts():
            if a["id"] == alert_id and a.get("clip"):
                p = (self.out / a["clip"]).resolve()
                # Only pixelated alert clips inside the output folder, never raw segments.
                if p.is_file() and self.out.resolve() in p.parents and "alerts" in p.parts:
                    return p
        return None

    def summary(self) -> dict:
        """Counts per tier and label. Precision = real / (real + false); "unsure" is left out."""
        rows = self.labelled()
        out = {"would_be_alerts": len(rows), "labelled": sum(1 for r in rows if r["label"])}
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


def _run_camera(cam: Camera, cfg: ShadowConfig, log: ShadowLog, once: bool, stop: threading.Event) -> None:
    from bree.cli import make_backend
    from bree.events.zones import load_store_config
    from bree.ledger.payments import FolderPayments
    from bree.pipeline import run_pipeline
    store = load_store_config(cam.store)
    backend = make_backend(cfg.backend, store, runtime=cfg.runtime)
    raw_cfg = cfg.record_raw
    while not stop.is_set():
        session = time.strftime("%Y%m%dT%H%M%S")
        payments = None
        if cfg.pos_export_dir:
            # Live: receipts already in the folder predate this run. Replay (--once): read them all.
            payments = FolderPayments(cfg.pos_export_dir, stream_start_wall=time.time(),
                                      terminals=store.terminals, skip_existing=not once)
        recorder = (RawRecorder(Path(cfg.output_dir) / "raw" / cam.name, store.fps,
                                raw_cfg.get("segment_minutes", 5), raw_cfg.get("retention_hours", 24))
                    if raw_cfg.get("enabled") else None)
        try:
            s = run_pipeline(cam.rtsp_url, store, backend, Path(cfg.output_dir) / cam.name / session,
                             payments=payments, save_video=False, log_frames=False, verbose=False,
                             ledger_overrides=cfg.ledger,
                             on_alert=lambda a: log.add_alert(would_be_record(a, cam.name, session, cfg.output_dir)),
                             on_frame=(lambda fr, *_: recorder.write(fr.image)) if recorder else None)
            print(f"[{cam.name}] stream ended after {s.frames} frames, {len(s.alerts)} would-be alert(s)")
        except IOError:
            print(f"[{cam.name}] cannot open the stream")   # not the URL: it may hold credentials
        finally:
            if recorder:
                recorder.close()
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
    threads = [threading.Thread(target=_run_camera, args=(c, cfg, log, once, stop), daemon=True)
               for c in cfg.cameras]
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
