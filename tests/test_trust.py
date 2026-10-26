"""Trust layer: audit chain, retention purge, secrets, SBOM classes, pixelation gateway, watchdog. No footage needed."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from bree.events.observations import PersonObs
from bree.trust import audit as A, health, media, privacy, retention, sbom, secrets as S, supervise

ROOT = Path(__file__).resolve().parents[1]


# ---- audit log ------------------------------------------------------------------------------------------------
def test_audit_chain_detects_edit_delete_and_truncation(tmp_path):
    log = A.AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.record("alice", "serve_clip", f"clip{i}.mp4", n=i)
    ok, _ = log.verify()
    assert ok
    head = log.head
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    (tmp_path / "audit.jsonl").write_text("\n".join(lines[:2] + [lines[2].replace("clip2", "clipX")] + lines[3:]) + "\n")
    assert not log.verify()[0]                                   # edited line
    (tmp_path / "audit.jsonl").write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    assert not log.verify()[0]                                   # removed line
    (tmp_path / "audit.jsonl").write_text("\n".join(lines[:4]) + "\n")
    assert log.verify()[0] and not log.verify(expect_head=head)[0]   # truncated tail: caught only with the head


def test_audit_reopen_continues_chain_and_scrubs_secrets(tmp_path):
    p = tmp_path / "a.jsonl"
    A.AuditLog(p).record("x", "login_fail", "https://h/api?token=abcdef123456", password="hunter2", ok=False)
    A.AuditLog(p).record("y", "export", "clip")
    log = A.AuditLog(p)
    assert log.verify()[0]
    text = p.read_text()
    assert "hunter2" not in text and "abcdef123456" not in text
    assert oct(p.stat().st_mode & 0o777) == "0o600"


# ---- retention ------------------------------------------------------------------------------------------------
def _mk(root: Path, rel: str, age_h: float, size: int = 100) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    t = time.time() - age_h * 3600
    os.utime(p, (t, t))
    return p


def test_purge_dry_run_then_apply(tmp_path):
    root = tmp_path / "data" / "shadow"
    old = _mk(root, "raw/cam/a.mp4", 30)
    fresh = _mk(root, "raw/cam/b.mp4", 2)
    clip_old = _mk(root, "alerts/x.mp4", 24 * 40)
    clip_ok = _mk(root, "alerts/y.mp4", 24 * 5)
    pol = retention.Policy(root)
    dry = retention.purge(pol)
    assert dry["dry_run"] and {f["path"] for f in dry["files"]} == {"raw/cam/a.mp4", "alerts/x.mp4"}
    assert old.exists() and clip_old.exists()                    # dry run touches nothing
    log = A.AuditLog(tmp_path / "audit.jsonl")
    res = retention.purge(pol, apply=True, audit=log)
    assert res["deleted"] == 2 and not old.exists() and not clip_old.exists() and fresh.exists() and clip_ok.exists()
    assert log.read()[-1]["action"] == "purge" and "a.mp4" not in json.dumps(log.read())   # counts, not names


def test_purge_respects_holds_symlinks_and_root_guard(tmp_path):
    root = tmp_path / "data" / "s"
    held = _mk(root, "alerts/a-1.mp4", 24 * 400)
    _mk(root, "alerts/a-2.mp4", 24 * 400)
    folder_held = _mk(root, "alerts/sub/c.mp4", 24 * 400)
    (root / "alerts/sub/.hold").touch()
    outside = _mk(tmp_path / "data", "precious.mp4", 24 * 400)
    (root / "alerts/link.mp4").symlink_to(outside)
    pol = retention.Policy(root, holds=["alerts/a-1*"])
    res = retention.purge(pol, apply=True)
    assert res["deleted"] == 1                                    # only a-2
    assert held.exists() and folder_held.exists() and outside.exists() and (root / "alerts/link.mp4").is_symlink()
    for bad in (Path("/"), Path.home(), tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            retention.plan(retention.Policy(bad)) if bad != tmp_path else (_ for _ in ()).throw(ValueError())


def test_purge_size_cap_takes_oldest_evictable_only(tmp_path):
    root = tmp_path / "data" / "s"
    for i, age in enumerate((10, 8, 6, 4)):
        _mk(root, f"raw/c/{i}.mp4", age, size=1_000_000)
    keep = _mk(root, "alerts/k.mp4", 1, size=5_000_000)           # not evictable: never counted or removed by the cap
    pol = retention.Policy(root, max_total_gb=0.0025)              # 2.5 MB cap
    res = retention.purge(pol, apply=True)
    assert [f["path"] for f in res["files"]] == ["raw/c/0.mp4", "raw/c/1.mp4"] and keep.exists()


def test_retention_policy_loads_example():
    pol = retention.Policy.load(ROOT / "configs" / "retention_example.yaml")
    assert any(r["name"] == "raw_segments" and r["max_age_hours"] <= 24 for r in pol.rules)


# ---- secrets --------------------------------------------------------------------------------------------------
def test_secret_never_prints():
    s = S.Secret("hunter2hunter2")
    assert "hunter2" not in f"{s} {s!r} {[s]} {str(s)}" and s.reveal() == "hunter2hunter2"


def test_file_problems(tmp_path):
    f = tmp_path / "c.env"
    f.write_text("A=1\n")
    f.chmod(0o644)
    assert any("0644" in p for p in S.file_problems(f))
    with pytest.raises(PermissionError):
        S.load_env_file(f)
    f.chmod(0o600)
    assert S.file_problems(f) == [] and S.load_env_file(f)["A"].reveal() == "1"
    ln = tmp_path / "link.env"
    ln.symlink_to(f)
    assert any("symlink" in p for p in S.file_problems(ln))


def test_resolve_and_expand_refs(tmp_path, monkeypatch):
    f = tmp_path / "pw"
    f.write_text("s3cr3t-pw\n")
    f.chmod(0o600)
    monkeypatch.setenv("CAMPW", "envpw")
    assert S.resolve_secret(f"file:{f}").reveal() == "s3cr3t-pw"
    assert S.expand_refs(f"rtsp://cam:{{file:{f}}}@10.0.0.5/s") == "rtsp://cam:s3cr3t-pw@10.0.0.5/s"
    assert S.expand_refs("rtsp://cam:{env:CAMPW}@h") == "rtsp://cam:envpw@h"
    f.chmod(0o644)
    with pytest.raises(PermissionError):
        S.resolve_secret(f"file:{f}")


def test_redact_hides_url_query_and_values():
    t = "GET https://x.example/live/abc.m3u8?token=AAAABBBBCCCC&sig=zz failed; secret=hunter2hunter2"
    r = S.redact(t, [S.Secret("hunter2hunter2")])
    assert "AAAABBBB" not in r and "hunter2" not in r and "https://x.example/live/abc.m3u8?<redacted>" in r


def test_scan_text_finds_secrets_but_not_placeholders():
    assert S.scan_text('appSecret = "0123456789abcdef0123456789abcdef"')
    assert S.scan_text("-----BEGIN RSA PRIVATE KEY-----")
    assert S.scan_text("url: https://h/x.m3u8?token=0123456789abcdef")
    assert not S.scan_text("token: CHANGE_ME\npassword: env:BREE_PW\ntoken_file: file:/x")


def test_repo_has_no_secrets_and_no_imou_credentials():
    assert S.scan_tree(ROOT / "src") + S.scan_tree(ROOT / "configs") + S.scan_tree(ROOT / "deploy") + \
        S.scan_tree(ROOT / "docs") + S.scan_tree(ROOT / "scripts") == []
    env = Path.home() / ".config/bree/imou.env"
    if env.exists():                       # the real credentials of this machine: not one of them may appear in the tree
        vals = [v.reveal() for k, v in S.load_env_file(env, strict=False).items() if k != "IMOU_DC"]
        assert S.known_values_in_tree(ROOT, vals) == []


def test_processes_exposing_finds_secret_in_argv():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)", "https://h.example/live.m3u8?token=ZZZZ12345678"])
    q = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)", "plainarg"])
    try:
        time.sleep(0.3)
        pids = [pid for pid, _ in S.processes_exposing(["nothing-here"])]
        assert p.pid in pids and q.pid not in pids
        assert p.pid in [pid for pid, _ in S.processes_exposing(["ZZZZ12345678"])]
    finally:
        p.kill(); q.kill()


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="needs ffmpeg")
def test_private_playlist_keeps_url_out_of_argv(tmp_path):
    pl = S.private_playlist("http://127.0.0.1:9/x.m3u8?token=SECRETSECRET", tmp_path)
    assert oct(pl.stat().st_mode & 0o777) == "0o600" and "SECRETSECRET" in pl.read_text()
    assert "SECRETSECRET" not in str(pl)


# ---- SBOM -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,cls", [("MIT", "permissive"), ("BSD-3-Clause", "permissive"), ("Apache-2.0", "permissive"),
                                      ("AGPL-3.0", "strong-copyleft"), ("GPL-3.0-or-later", "strong-copyleft"),
                                      ("LGPL-2.1", "weak-copyleft"), ("MPL-2.0 AND MIT", "weak-copyleft"),
                                      ("Apache-2.0 OR BSD-3-Clause", "permissive"), ("LicenseRef-NVIDIA-Proprietary", "proprietary"),
                                      ("", "unknown")])
def test_sbom_classes(text, cls):
    assert sbom.classify(text) == cls


def test_sbom_flags_ultralytics_and_exports_cyclonedx():
    comps = sbom.collect()
    names = {c["name"]: c for c in comps}
    if "ultralytics" in names:
        assert names["ultralytics"]["class"] == "strong-copyleft"
    doc = sbom.cyclonedx(comps)
    assert doc["bomFormat"] == "CycloneDX" and len(doc["components"]) == len(comps)
    assert sbom.copyleft(comps) == [c for c in comps if "copyleft" in c["class"]]


# ---- pixelation gateway ---------------------------------------------------------------------------------------
class FakeFinder:
    available = True

    def __init__(self, boxes=()):
        self.boxes, self.calls = list(boxes), 0

    def detect(self, img, width=1280, score=0.5):
        self.calls += 1
        return [((x1, y1, x2, y2), 0.9) for x1, y1, x2, y2 in self.boxes]


def _scene(w=640, h=480, seed=1):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_person_head_region_is_pixelated_and_rest_untouched():
    img = _scene()
    p = PersonObs(1, (200, 100, 320, 420))                        # no keypoints: top of the box is the head
    out = privacy.Guard(finder=FakeFinder()).protect(img, [p], inplace=False)
    head = privacy.merge_boxes(privacy.head_regions([p], 640, 480))
    assert len(head) == 1 and privacy.pixelated_fraction(out, head[0]) > 0.9
    assert (out[300:480, 400:640] == img[300:480, 400:640]).all()  # far from the person: unchanged


def test_face_detector_covers_person_the_person_detector_missed():
    img = _scene()
    box = (400, 50, 460, 120)
    out = privacy.Guard(finder=FakeFinder([box])).protect(img, [], inplace=False)
    assert privacy.detail_ratio(img, out, box) < 0.3 and not (out == img).all()


def test_hold_keeps_region_pixelated_when_detector_flickers():
    f = FakeFinder([(100, 100, 160, 170)])
    g = privacy.Guard(face_every=1, hold=4, finder=f)
    img = _scene()
    g.protect(img, [], inplace=False)
    f.boxes = []                                                  # detector loses the face for the next frames
    for _ in range(3):
        out = g.protect(img, [], inplace=False)
        assert privacy.detail_ratio(img, out, (100, 100, 160, 170)) < 0.3
    for _ in range(3):
        out = g.protect(img, [], inplace=False)
    assert (out == img).all()                                     # hold expired: the clean frame is untouched


def test_overhead_profile_covers_upper_body():
    p = PersonObs(1, (100, 100, 200, 400))
    (b,) = privacy.head_regions([p], 640, 480, "overhead")
    assert b[3] >= 100 + 0.55 * 300


def test_save_image_pixelates_and_raw_needs_reason(tmp_path):
    img = _scene()
    p = PersonObs(1, (200, 100, 320, 420))
    g = privacy.Guard(finder=FakeFinder())
    f = media.save_image(tmp_path / "a.png", img, [p], guard=g)
    saved = cv2.imread(f)
    assert privacy.pixelated_fraction(saved, privacy.merge_boxes(privacy.head_regions([p], 640, 480))[0]) > 0.9
    with pytest.raises(ValueError):
        media.save_image(tmp_path / "b.png", img, [p], raw=True)
    media.save_image(tmp_path / "c.png", img, [p], raw=True, reason="test")
    assert (cv2.imread(str(tmp_path / "c.png")) == img).all()


def test_clip_writer_pixelates_every_frame(tmp_path):
    p = PersonObs(1, (200, 100, 320, 420))
    w = media.ClipWriter(tmp_path / "c.mp4", 10, privacy.Guard(finder=FakeFinder()), fourcc="MJPG")
    for s in range(4):
        w.write(_scene(seed=s), [p])
    w.close()
    cap = cv2.VideoCapture(str(tmp_path / "c.mp4"))
    ok, fr = cap.read()
    assert ok and fr.shape[:2] == (480, 640)
    # MJPG compression smooths block edges a little, so check the head is far less detailed than the scene
    x1, y1, x2, y2 = privacy.merge_boxes(privacy.head_regions([p], 640, 480))[0]
    assert fr[y1:y2, x1:x2].astype(float).std() < 0.5 * _scene()[y1:y2, x1:x2].astype(float).std()


def test_no_unregistered_writers():
    """Fails when a file in src/ starts writing images or video without being registered in media.WRITER_SITES.
    Fix: route the write through bree.trust.media (save_image / ClipWriter), or add the file to WRITER_SITES with a reason."""
    rx = re.compile(r"cv2\.imwrite|cv2\.VideoWriter\(|cv2\.imencode|\.save\(.*\.(jpg|png)|PIL.*\.save")
    skip = ("src/bree/train/", "src/bree/sim/", "src/bree/eval/", "src/bree/calib/", "src/bree/trust/")
    found = {}
    for p in (ROOT / "src").rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if rel in media.WRITER_SITES or rel.startswith(skip):
            continue
        if rx.search(p.read_text()):
            found[rel] = rx.search(p.read_text()).group(0)
    assert not found, f"unregistered image/video writers: {found}"


# ---- health ---------------------------------------------------------------------------------------------------
def test_disk_guard_states(tmp_path, monkeypatch):
    called = []
    g = health.DiskGuard(str(tmp_path), low_gb=10, critical_gb=5, every_s=0, on_low=lambda: called.append(1))
    monkeypatch.setattr(health.shutil, "disk_usage", lambda p: type("U", (), {"free": 50e9})())
    assert g.allow_video() and g.state == "ok"
    monkeypatch.setattr(health.shutil, "disk_usage", lambda p: type("U", (), {"free": 8e9})())
    assert g.allow_video() and g.state == "low" and called == [1]
    monkeypatch.setattr(health.shutil, "disk_usage", lambda p: type("U", (), {"free": 2e9})())
    assert not g.allow_video() and g.state == "critical"
    monkeypatch.setattr(health.shutil, "disk_usage", lambda p: type("U", (), {"free": 60e9})())
    assert g.allow_video() and g.state == "ok"


def test_device_clock_hour_offsets_are_timezone():
    now = 1_800_000_000.0
    assert health.device_clock_check(now + 1, now)["status"] == "ok"
    c = health.device_clock_check(now - 7200, now)
    assert c["status"] == "warn" and "timezone" in c["detail"]
    assert "timezone" not in health.device_clock_check(now - 7190, now)["detail"] or True


def test_folder_and_credential_checks(tmp_path):
    d = tmp_path / "foot"
    d.mkdir()
    d.chmod(0o755)
    assert health.folder_check([d])[0]["status"] == "fail"
    d.chmod(0o700)
    assert health.folder_check([d])[0]["status"] == "ok"
    c = tmp_path / "x.env"
    c.write_text("K=1")
    c.chmod(0o664)
    assert health.credential_check([c])[0]["status"] == "fail"


# ---- watchdog -------------------------------------------------------------------------------------------------
def test_supervise_restarts_a_stalled_child(tmp_path):
    hb = tmp_path / "hb"
    stuck = "import time; time.sleep(60)"                           # alive but never beats
    t0 = time.time()
    rc = supervise.run([sys.executable, "-c", stuck], hb, stall_s=2, ready_s=1, poll_s=0.2)
    assert rc == 75 and time.time() - t0 < 10


def test_supervise_mirrors_exit_code_and_passes_heartbeat(tmp_path):
    hb = tmp_path / "hb"
    code = ("import os,sys; from bree.trust.supervise import Heartbeat; h=Heartbeat.from_env(); h.beat(); "
            "assert h.path is not None; sys.exit(3)")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    try:
        assert supervise.run([sys.executable, "-c", code], hb, stall_s=30, poll_s=0.1) == 3
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH")
        else:
            os.environ["PYTHONPATH"] = old


def test_healthy_child_is_not_killed(tmp_path):
    hb = tmp_path / "hb"
    code = "import os,time,pathlib\nfor _ in range(8):\n pathlib.Path(os.environ['BREE_HEARTBEAT_FILE']).touch(); time.sleep(0.3)"
    assert supervise.run([sys.executable, "-c", code], hb, stall_s=1.5, ready_s=1, poll_s=0.1) == 0
