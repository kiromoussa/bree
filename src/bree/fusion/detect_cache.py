"""Person detections of real recorded segments, cached once so the fusion experiments can be repeated without the model.

    python -m bree.fusion.detect_cache <real/station dir> <out dir> --cameras 07,29,12 [--stride 6] [--imgsz 1280]

Reads <station>/chNN/YYYYmmdd_HHMMSS.mp4 (finished segments only, never .part) and writes
<out>/chNN/YYYYmmdd_HHMMSS.jsonl: one line per sampled frame
  {"f": frame index, "t": seconds since segment start (frame index / fps), "persons": [{"bbox": [x0, y0, x1, y1] in
   source pixels, "conf": c, "app": 104 clothing-colour numbers (bree.track.reid, shoulders down, never the head)}]}
Every line of a file is preceded by a header line {"segment": name, "fps": f, "size": [w, h], "stride": s}.
Frames are shrunk to <= 1280 px wide before the detector (the 2560x1440 main stream is not needed to find people and
decodes slower than it detects); boxes are scaled back to source pixels. Local data only: the output holds
boxes, no images. Footage is never copied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SEGMENT_FMT = "%Y%m%d_%H%M%S"


def segment_start(path: Path) -> datetime:
    """Recorder clock (machine local time) at the first frame, from the file name."""
    return datetime.strptime(path.stem, SEGMENT_FMT)


def segments(station: Path, cam: str) -> list[Path]:
    return sorted(p for p in (station / f"ch{cam}").glob("*.mp4") if p.stat().st_size > 1_000_000)


def run_segment(detect, path: Path, dst: Path, stride: int, max_w: int = 1280, max_frames: int | None = None) -> int:
    import cv2
    from bree.track.reid import part_colors
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.3
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, max_w / w)
    n = f = 0
    tmp = dst.with_suffix(".jsonl.part")
    with tmp.open("w") as fh:
        fh.write(json.dumps({"segment": path.name, "fps": round(fps, 3), "size": [w, h], "stride": stride}) + "\n")
        while True:
            ok = cap.grab()
            if not ok:
                break
            if f % stride == 0:
                ok, im = cap.retrieve()
                if not ok:
                    break
                small = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else im
                persons = []
                for d in detect(small):
                    b = [round(v / s, 1) for v in d["bbox"]]
                    app = [round(float(v), 3) for v in part_colors(small, d["bbox"], None)[:2].ravel()]
                    persons.append({"bbox": b, "conf": d["conf"], "app": app})
                fh.write(json.dumps({"f": f, "t": round(f / fps, 3), "persons": persons}) + "\n")
                n += 1
                if max_frames and n >= max_frames:
                    break
            f += 1
    cap.release()
    tmp.rename(dst)
    return n


def person_detector(imgsz: int = 1280, conf: float = 0.25):
    """Plain person boxes (class 0) of the repo's detector weights on the GPU when there is one."""
    from ultralytics import YOLO
    from bree.cli import _weights
    from bree.hw import detect_hardware
    hw = detect_hardware()
    model = YOLO(str(_weights(hw.detect_model)))

    def detect(image: np.ndarray) -> list[dict]:
        r = model.predict(image, imgsz=imgsz, conf=conf, classes=[0], device=hw.device, verbose=False)[0]
        return [{"bbox": b.tolist(), "conf": round(float(c), 3)} for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return detect


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("station")
    ap.add_argument("out")
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args(argv)
    detect = person_detector(args.imgsz, args.conf)
    for cam in args.cameras.split(","):
        out = Path(args.out) / f"ch{cam}"
        out.mkdir(parents=True, exist_ok=True)
        for seg in segments(Path(args.station), cam):
            dst = out / f"{seg.stem}.jsonl"
            if dst.exists():
                continue
            t0 = time.time()
            n = run_segment(detect, seg, dst, args.stride, max_frames=args.max_frames)
            print(f"ch{cam} {seg.name}: {n} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
