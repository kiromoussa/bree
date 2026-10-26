"""`bree onboard`: from a serial (or a folder of recorded footage) to a station folder a human can correct.

Steps: list channels -> sample frames per channel -> statistics -> role suggestion -> still + pixelated contact
sheet -> draft per-camera store YAML with zone templates -> shadow config draft -> fleet registry entry.
Nothing here decides anything alone: roles carry their evidence and a `needs_review` flag, zones are templates,
and a role a human has already set (bree fleet set-role) is kept on re-runs.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2

from bree.fleet.registry import CameraEntry, Fleet, Station
from bree.onboard import drafts
from bree.onboard.roles import suggest_role
from bree.onboard.sheet import contact_sheet, pixelate_people
from bree.onboard.stats import compute_stats

LIMITS = [
    "Roles are suggestions from simple scene statistics (docs/ultra/platform.md); check each one against its still.",
    "Zones are templates or crude boxes from detections; they are not validated. Correct them with scripts/draw_zones.py.",
    "Stills are raw frames (people are visible). The contact sheet pixelates detected heads only. Keep both local.",
    "Backroom and obscured cameras are not reliably recognised (no real examples were available when this was built).",
]


def _private_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    os.chmod(p, 0o700)
    return p


def _pick_still(sample, stats):
    """The sampled frame with the fewest people (cleanest view for drawing zones); ties: the middle one."""
    best, best_n = 0, 10 ** 9
    for i, dets in enumerate(stats.detections):
        n = sum(1 for d in dets if d.name == "person" and d.conf >= 0.3)
        if n < best_n or (n == best_n and abs(i - len(sample.frames) / 2) < abs(best - len(sample.frames) / 2)):
            best, best_n = i, n
    return best


def onboard(discovery, scene, fleet: Fleet, station_id: str, serial: str, channels: list[int] | None = None,
            stream: str = "sub", zone_res: tuple[int, int] | None = None, clock_offset_s: float | None = None,
            dc: str = "or", log=print) -> dict:
    sdir = _private_dir(fleet.dir(station_id))
    stills = _private_dir(sdir / "stills")
    cams_dir = _private_dir(sdir / "cameras")
    avail = discovery.channels()
    chans = [c for c in avail if channels is None or c in channels]
    entries, tiles, report_cams, shadow_cams = [], [], [], []
    for ch in chans:
        t0 = time.time()
        sample = discovery.sample(ch)
        feats_stats = compute_stats(sample, scene)
        sug = suggest_role(ch, feats_stats.flat(), error=sample.error, online=sample.online)
        size = zone_res or sample.size
        entry = CameraEntry(ch, sug.role, round(sug.confidence, 2), "suggested", stream=stream, online=sample.online and not sample.error)
        entries.append(entry)
        rec = {"channel": ch, "source": sample.source, "stream_size": list(sample.size), "n_frames": len(sample.frames),
               "error": sample.error, "suggestion": sug.to_dict(), "stats": {k: (None if v is None else round(float(v), 4)) for k, v in feats_stats.flat().items()}}
        if sample.frames:
            i = _pick_still(sample, feats_stats)
            raw = sample.frames[i]
            cv2.imwrite(str(stills / f"ch{ch:02d}.jpg"), raw, [cv2.IMWRITE_JPEG_QUALITY, 90])
            people = [d.box for d in feats_stats.detections[i] if d.name == "person" and d.conf >= 0.2]
            tiles.append((f"ch{ch:02d}  {sug.role}{'?' if sug.needs_review else ''} ({sug.confidence:.2f})  {sample.size[0]}x{sample.size[1]}",
                          pixelate_people(raw, people)))
            zones, znotes = drafts.zones_for(sug.role, size, feats_stats.detections, sample.frames)
            ysrc = drafts.store_yaml(station_id, serial, ch, sug.role, size, zones, znotes, sug.evidence, sample.fps or 15.0, sug.needs_review)
            (cams_dir / f"ch{ch:02d}.yaml").write_text(ysrc)
            rec["zones"], rec["zone_notes"] = [z["name"] for z in zones], znotes
        else:
            tiles.append((f"ch{ch:02d}  no picture: {sample.error}", __import__("numpy").zeros((90, 160, 3), "uint8")))
        report_cams.append(rec)
        log(f"ch{ch:02d}: {sug.role} ({sug.confidence:.2f}){' needs review' if sug.needs_review else ''}  [{time.time() - t0:.0f}s]")
    st = Station(station_id, serial, dc=dc, clock_offset_s=clock_offset_s, cameras=entries,
                 timezone_note="DVR overlay clock minus edge-box clock; the DVR's own time zone is not known" if clock_offset_s is not None else "")
    st = fleet.merge_onboarding(st)
    for c in st.cameras:   # roles a human fixed earlier come back here: reflect them in the report
        for r in report_cams:
            if r["channel"] == c.channel and c.role_source == "human":
                r["human_role"] = c.role
    if tiles:
        cv2.imwrite(str(sdir / "contact_sheet.jpg"), contact_sheet(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for c in st.cameras:
        f = cams_dir / f"ch{c.channel:02d}.yaml"
        if c.enabled and f.exists() and c.role not in ("outdoor", "obscured", "backroom", "unknown"):
            shadow_cams.append((c.channel, st.source_url(c.channel), str(f)))
    (sdir / "shadow.yaml").write_text(drafts.shadow_yaml(station_id, shadow_cams, f"out/shadow_{station_id}"))
    report = {"station": station_id, "serial": serial, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "limits": LIMITS,
              "clock_offset_s": clock_offset_s, "cameras": report_cams}
    (sdir / "onboard_report.json").write_text(json.dumps(report, indent=1))
    return report
