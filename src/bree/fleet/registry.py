"""Fleet registry: which stations exist, what cameras each has, and whether they are alive.

State is plain files so it survives restarts and can be inspected by hand (no database, no daemon):

  <root>/stations/<id>/station.yaml        written by `bree onboard`: serial, data centre, channels + roles, clock offset
  <root>/stations/<id>/health/chNN.json    heartbeat written by the running source (bree.ingest.imou.HealthSink)
  <root>/stations/<id>/cameras/chNN.yaml   per-camera store config drafts (zones to be corrected by a human)

`root` defaults to $BREE_FLEET or ~/.local/share/bree/fleet. Footage never goes in here (it lives outside every
git repo under <data>/real/<station>/); station.yaml names the credential *file*, never the credentials.
Health is derived from heartbeat age, so a crashed process shows up as `stale` then `down` without any cleanup.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_ROOT = Path(os.environ.get("BREE_FLEET", Path.home() / ".local/share/bree/fleet"))
STALE_AFTER_S = 30.0          # no frame for this long: stale
DOWN_AFTER_S = 300.0          # no frame for this long: down
MIN_FREE_GB = 5.0
STATION_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")
ROLES = ("door", "register", "cooler", "aisle", "backroom", "outdoor", "obscured", "unknown")


@dataclass
class CameraEntry:
    channel: int
    role: str = "unknown"
    role_confidence: float = 0.0
    role_source: str = "suggested"       # suggested | human (a person corrected it: never overwrite)
    enabled: bool = True
    stream: str = "sub"
    online: bool | None = None


@dataclass
class Station:
    id: str
    serial: str
    name: str = ""
    dc: str = "or"
    credentials_file: str = "~/.config/bree/imou.env"    # a path, not a secret
    timezone_note: str = ""                              # free text: what we know about the DVR clock
    clock_offset_s: float | None = None                  # DVR overlay clock minus the edge box clock
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    cameras: list[CameraEntry] = field(default_factory=list)

    def camera(self, ch: int) -> CameraEntry | None:
        return next((c for c in self.cameras if c.channel == ch), None)

    def source_url(self, ch: int) -> str:
        c = self.camera(ch)
        q = f"stream={c.stream if c else 'sub'}&station={self.id}"
        if self.clock_offset_s is not None:
            q += f"&clock_offset={self.clock_offset_s:.0f}"
        return f"imou://{self.serial}/{ch}?{q}"


class Fleet:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else DEFAULT_ROOT

    def dir(self, station_id: str) -> Path:
        if not STATION_ID_RE.fullmatch(station_id):
            raise ValueError(f"station id must be lowercase letters, digits, - or _ (got {station_id!r})")
        return self.root / "stations" / station_id

    def health_path(self, station_id: str, ch: int) -> Path:
        return self.dir(station_id) / "health" / f"ch{ch:02d}.json"

    # -- stations
    def save(self, st: Station) -> Path:
        d = self.dir(st.id)
        d.mkdir(parents=True, exist_ok=True)
        raw = {k: v for k, v in st.__dict__.items() if k != "cameras"}
        raw["cameras"] = [dict(c.__dict__) for c in st.cameras]
        tmp = d / "station.yaml.tmp"
        tmp.write_text(yaml.safe_dump(raw, sort_keys=False))
        os.replace(tmp, d / "station.yaml")
        return d / "station.yaml"

    def load(self, station_id: str) -> Station:
        f = self.dir(station_id) / "station.yaml"
        if not f.exists():
            raise KeyError(f"no station {station_id!r} in {self.root}")
        raw = yaml.safe_load(f.read_text())
        cams = [CameraEntry(**c) for c in raw.pop("cameras", [])]
        return Station(**raw, cameras=cams)

    def stations(self) -> list[Station]:
        base = self.root / "stations"
        return [self.load(p.name) for p in sorted(base.iterdir()) if (p / "station.yaml").exists()] if base.exists() else []

    def merge_onboarding(self, st: Station) -> Station:
        """Save `st`, keeping camera entries a human already corrected (role_source == "human")."""
        try:
            old = self.load(st.id)
        except KeyError:
            old = None
        if old:
            for c in st.cameras:
                o = old.camera(c.channel)
                if o and o.role_source == "human":
                    c.role, c.role_confidence, c.role_source, c.enabled, c.stream = o.role, o.role_confidence, "human", o.enabled, o.stream
        self.save(st)
        return st

    def set_role(self, station_id: str, ch: int, role: str) -> Station:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        st = self.load(station_id)
        c = st.camera(ch)
        if c is None:
            raise KeyError(f"station {station_id} has no channel {ch}")
        c.role, c.role_source, c.role_confidence = role, "human", 1.0
        self.save(st)
        return st

    # -- health
    def read_heartbeat(self, station_id: str, ch: int) -> dict | None:
        p = self.health_path(station_id, ch)
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def camera_health(self, st: Station, c: CameraEntry, now: float | None = None) -> dict:
        now = now or time.time()
        hb = self.read_heartbeat(st.id, c.channel)
        out = {"channel": c.channel, "role": c.role, "enabled": c.enabled}
        if not c.enabled:
            return {**out, "state": "disabled"}
        if hb is None:
            return {**out, "state": "never_seen", "detail": "no heartbeat file: source never started"}
        last = hb.get("last_frame_wall")
        age = None if last is None else now - last
        state = hb.get("state", "unknown")
        # a source that wrote "ok" but then stopped writing is only stale/down by the clock
        if state in ("ok", "reconnecting", "starting", "stub"):
            if age is None or age > DOWN_AFTER_S:
                state = "down"
            elif age > STALE_AFTER_S:
                state = "stale"
        elif state == "stopped" and age is not None and age < STALE_AFTER_S:
            state = "stopped"
        return {**out, "state": state, "last_frame_age_s": None if age is None else round(age, 1), "frames": hb.get("frames", 0),
                "reconnects": hb.get("reconnects", 0), "stub_opens": hb.get("stub_opens", 0), "last_error": hb.get("last_error"),
                "resolution": [hb.get("width"), hb.get("height")], "fps": hb.get("fps"), "stream": hb.get("stream")}

    def station_health(self, st: Station, now: float | None = None) -> dict:
        cams = [self.camera_health(st, c, now) for c in st.cameras]
        active = [c for c in cams if c["state"] != "disabled"]
        ok = [c for c in active if c["state"] == "ok"]
        if not active:
            overall = "no_cameras"
        elif len(ok) == len(active):
            overall = "ok"
        elif ok:
            overall = "degraded"
        else:
            overall = "down"
        last = [h["last_frame_age_s"] for h in cams if h.get("last_frame_age_s") is not None]
        return {"station": st.id, "serial": st.serial, "overall": overall, "last_frame_age_s": min(last) if last else None,
                "cameras": cams}

    def check(self, st: Station, data_dir: str | Path | None = None, now: float | None = None) -> list[dict]:
        """Local checks (no network): each {name, ok, detail}. Fix hints are in `detail`."""
        res = []

        def add(name, ok, detail=""):
            res.append({"name": name, "ok": bool(ok), "detail": detail})
        add("tools: ffmpeg", shutil.which("ffmpeg"), "needed to record segments and cut stills")
        add("tools: tesseract", shutil.which("tesseract"), "needed to read the DVR overlay clock; without it the clock offset must be set by hand")
        cred = Path(st.credentials_file).expanduser()
        add("credentials file present", cred.exists() or bool(os.environ.get("IMOU_APP_ID")),
            f"{cred} (checked for existence only; contents are never read here)")
        if cred.exists():
            mode = cred.stat().st_mode & 0o077
            add("credentials file private", mode == 0, "chmod 600 it" if mode else "")
        add("clock offset known", st.clock_offset_s is not None, "run `bree clock-check` on a segment, or set it by hand" if st.clock_offset_s is None else f"{st.clock_offset_s:.0f} s")
        free = shutil.disk_usage(self.root if self.root.exists() else Path.home()).free / 1e9
        add("disk free", free >= MIN_FREE_GB, f"{free:.0f} GB free (need {MIN_FREE_GB:.0f})")
        if data_dir:
            fd = Path(data_dir).expanduser() / "real" / st.id
            add("footage dir private", not fd.exists() or (fd.stat().st_mode & 0o077) == 0, f"{fd}: chmod 700" if fd.exists() else "no footage yet")
        for c in st.cameras:
            if not c.enabled:
                continue
            cfg = self.dir(st.id) / "cameras" / f"ch{c.channel:02d}.yaml"
            add(f"ch{c.channel:02d} store config", cfg.exists(), str(cfg) if cfg.exists() else "run bree onboard to draft it")
            h = self.camera_health(st, c, now)
            add(f"ch{c.channel:02d} heartbeat", h["state"] in ("ok", "stopped", "disabled"), f"{h['state']}" + (f": {h['last_error']}" if h.get("last_error") else ""))
        return res
