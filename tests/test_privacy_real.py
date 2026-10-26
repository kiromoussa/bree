"""Leak test on real footage kept in the data dir (never in git). Skipped when the footage or weights are absent.

Fails if, after pixelation, a face detector (several scales, score >= 0.5) still finds a face centred in the head area of
a detected person. The detector family is the same one the guard uses, so this is a regression test, not an accuracy
measure: the by-eye check is in docs/ultra/trust.md (REAL-MODEL-LABELLED).
"""
import glob
from pathlib import Path

import cv2
import pytest

pytestmark = pytest.mark.vision
DATA = Path.home() / "Projects/bree-data/real/trumanrd"
SEGS = sorted(glob.glob(str(DATA / "ch07/*.mp4")) + glob.glob(str(DATA / "ch12/*.mp4")))[:6:2]


@pytest.mark.skipif(not SEGS, reason="no real footage in the data dir")
def test_no_face_survives_in_person_head_areas():
    from bree.trust.privacy import FACE_MODEL, scan_faces, still_guard
    from bree.trust.redact import make_person_backend, persons_of
    if not FACE_MODEL.exists():
        pytest.skip("face model missing")
    be = make_person_backend()
    checked = leaks = 0
    for seg in SEGS:
        cap = cv2.VideoCapture(seg)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in range(4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * n / 4))
            ok, fr = cap.read()
            if not ok:
                continue
            persons = persons_of(be, fr)
            out = still_guard().protect(fr, persons, inplace=False)
            for (x1, y1, x2, y2), s, _ in scan_faces(out, score=0.5, widths=(1280, 2560)):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                for p in persons:
                    bx1, by1, bx2, by2 = p.bbox
                    if bx1 <= cx <= bx2 and by1 <= cy <= by1 + 0.35 * (by2 - by1):
                        leaks += 1
            checked += len(persons)
    assert leaks == 0, f"{leaks} face detection(s) survive in head areas ({checked} persons checked)"
