"""DVR overlay clock: OCR on synthetic frames, offset maths, file check, and live calibration through the mock."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from bree.ingest import overlay
from bree.ingest.imou import ImouClient, ImouCredentials, ImouSource
from bree.ingest.imou_mock import APP_ID, APP_SECRET, MockImou

needs_tesseract = pytest.mark.skipif(not overlay.have_tesseract(), reason="tesseract not installed")


def burn(h: int, m: int, s: int, size=(1280, 720)) -> np.ndarray:
    im = np.full((size[1], size[0], 3), 90, np.uint8)
    x = int(size[0] * 0.75)
    t = f"10-25-2026 {h:02d}:{m:02d}:{s:02d}"
    cv2.putText(im, t, (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(im, t, (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def test_parsers():
    assert overlay.parse_time_of_day(["x", "10-25-2026 16:29:12", "10-25-2026 16:29:12", "6:29:13"]) == 16 * 3600 + 29 * 60 + 12
    assert overlay.parse_time_of_day(["99:99:99", "abc", ""]) is None
    assert overlay.parse_datetime(["10-25-2026 16:29:12"]).year == 2026
    assert overlay.parse_datetime(["10-25-2099x"]) is None
    assert overlay.file_start_from_name("/a/b/20261025_182851.mp4").minute == 28
    assert overlay.file_start_from_name("clip.mp4") is None


def test_offset_wraps_midnight_and_drops_misreads():
    # DVR is 2 h behind; reference just after midnight so the DVR time of day is 22:xx the day before
    ref = [100.0 + 10 * i for i in range(8)]
    got = [(r - 7200) % 86400 for r in ref]
    got[3] = 12345                                     # a misread
    est = overlay.estimate_offset(list(zip(ref, got)))
    assert est.ok and est.offset_s == -7200 and est.n_read == 7
    assert not overlay.estimate_offset([(1.0, None)] * 4).ok
    assert not overlay.estimate_offset([(1.0, 5)]).ok              # one read is not enough


@needs_tesseract
def test_read_clock_on_a_synthetic_overlay():
    assert overlay.read_clock(burn(16, 29, 12)) == 16 * 3600 + 29 * 60 + 12
    assert overlay.read_clock(np.full((720, 1280, 3), 90, np.uint8)) is None      # no overlay: no invented time


def _clip(path, start_tod, seconds=20, fps=5, offset_name=None):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    for i in range(seconds * fps):
        t = int(start_tod + i / fps)
        w.write(burn(t // 3600 % 24, t // 60 % 60, t % 60))
    w.release()


@needs_tesseract
def test_check_file_clock_measures_a_two_hour_offset(tmp_path):
    f = tmp_path / "20261025_182851.mp4"             # recorder says 18:28:51 ...
    _clip(f, 16 * 3600 + 28 * 60 + 51)               # ... the DVR overlay says 16:28:51
    r = overlay.check_file_clock(f, n=5)
    assert r["estimate"]["ok"] and abs(r["estimate"]["offset_s"] + 7200) <= 2
    assert abs(r["overlay_seconds_per_file_second"] - 1.0) < 0.15


@needs_tesseract
def test_live_calibration_sets_the_station_clock():
    lt = time.localtime()
    tod_now = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        clip = pathlib.Path(d) / "c.mp4"
        _clip(clip, tod_now - 7200 - 40, seconds=40, fps=5)    # overlay 2 h behind, 40 s of video already "elapsed"
        with MockImou(channels={0: "ok"}, video=str(clip), size="1280x720", rate=5) as m:
            c = ImouClient(ImouCredentials(APP_ID, APP_SECRET, "or"), base_url=m.base_url)
            src = ImouSource(f"imou://{m.serial}/0", client=c, max_frames=2)
            # the mock plays a clip faster than real time, so sample consecutive frames (a real stream is paced by the camera)
            est = src.calibrate_clock(n=4, step_s=0.0)
            assert est.ok and src.clock.source == "overlay" and abs(src.clock.offset_s - (-7200 - 40)) <= 8
