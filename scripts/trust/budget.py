#!/usr/bin/env python3
"""Resource budget of the trust layer on THIS machine (REAL-UNLABELLED timings on real 2560x1440 frames)."""
import json, resource, sys, time, glob, os, tempfile
from pathlib import Path
import cv2, numpy as np
from bree.trust.privacy import Guard, still_guard, FaceFinder, pixelate
from bree.trust.audit import AuditLog
from bree.trust.retention import Policy, plan
from bree.trust.redact import make_person_backend, persons_of

seg = sorted(glob.glob(str(Path.home() / "Projects/bree-data/real/trumanrd/ch07/*.mp4")))[0]
cap = cv2.VideoCapture(seg)
frames = []
for _ in range(60):
    for _ in range(5):
        ok, fr = cap.read()
    frames.append(fr)
be = make_person_backend()
persons = [persons_of(be, f) for f in frames[:10]]
def t(fn, n):
    t0 = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - t0) / n * 1000
out = {}
f = FaceFinder()
out["face_detect_ms_640"] = t(lambda i: f.detect(frames[i], 640), 20)
out["face_detect_ms_1280"] = t(lambda i: f.detect(frames[i], 1280), 20)
out["face_detect_ms_2560"] = t(lambda i: f.detect(frames[i], 2560), 20)
out["person_detect_pose_ms_gpu"] = t(lambda i: persons_of(be, frames[i]), 20)
g = Guard(face_every=2)
out["guard_per_frame_ms_face_every2_w1280"] = t(lambda i: g.protect(frames[i], persons[i % 10], inplace=False), 40)
g1 = Guard(face_every=1000)
out["guard_per_frame_ms_person_only"] = t(lambda i: g1.protect(frames[i], persons[i % 10], inplace=False), 40)
out["still_guard_ms_w2560"] = t(lambda i: still_guard().protect(frames[i], persons[i % 10], inplace=False), 10)
with tempfile.TemporaryDirectory() as d:
    a = AuditLog(Path(d) / "a.jsonl")
    out["audit_record_us"] = t(lambda i: a.record("x", "serve_clip", f"c{i}", n=i), 2000) * 1000
    t0 = time.time(); ok, msg = a.verify(); out["audit_verify_2000_records_ms"] = (time.time() - t0) * 1000
    for i in range(20000):
        (Path(d) / f"f{i}.mp4").write_bytes(b"x")
    t0 = time.time(); plan(Policy(Path(d), rules=[{"name": "a", "glob": "*.mp4", "max_age_hours": 0.0001}])); out["purge_plan_20000_files_ms"] = (time.time() - t0) * 1000
out["python_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
out["yunet_model_kb"] = os.path.getsize(Path(__file__).resolve().parents[2] / "weights/face_detection_yunet_2023mar.onnx") / 1024
print(json.dumps({k: round(v, 2) for k, v in out.items()}, indent=1))
