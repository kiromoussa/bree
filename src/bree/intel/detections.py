"""Person detections of one video segment at a reduced frame rate, cached as JSONL so the flow analytics can be re-run
(tuned, evaluated) without running the model again. Flow metrics (counts, dwell, queues) do not need 30 fps: a stride
of about 5 (6 fps at 30 fps source) keeps a walking person moving a few pixels per step and cuts model time 5x.

    python -m bree.intel.detections <video> <out.jsonl> [--stride 5] [--imgsz 960] [--conf 0.25]

One line per sampled frame: {"frame": n, "t": seconds, "persons": [{"bbox": [x0,y0,x1,y1], "conf": c, "kpts": [[x,y,s]*17]}]}
and a first line {"meta": {...}} with the video size and fps. No images are written.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2


def detect_segment(video: str | Path, out: str | Path, stride: int = 5, imgsz: int | None = None,
                   conf: float = 0.25, max_frames: int | None = None, detector=None) -> int:
    """Run the person detector on every `stride`-th frame. Returns the number of sampled frames written."""
    if detector is None:
        from bree.track.people import person_detector
        detector = person_detector(imgsz=imgsz, conf=conf)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    n = i = 0
    with tmp.open("w") as fh:
        fh.write(json.dumps({"meta": {"video": Path(video).name, "fps": fps, "width": w, "height": h, "stride": stride,
                                      "imgsz": imgsz, "conf": conf}}) + "\n")
        while max_frames is None or i < max_frames:
            if not cap.grab():
                break
            if i % stride == 0:
                ok, im = cap.retrieve()
                if not ok:
                    break
                fh.write(json.dumps({"frame": i, "t": round(i / fps, 3), "persons": detector(im)}) + "\n")
                n += 1
            i += 1
    cap.release()
    tmp.rename(out)
    return n


def read_detections(path: str | Path) -> tuple[dict, list[dict]]:
    """(meta, frames) of a detections JSONL."""
    meta, frames = {}, []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        if "meta" in d:
            meta = d["meta"]
        else:
            frames.append(d)
    return meta, frames


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)
    print(detect_segment(a.video, a.out, a.stride, a.imgsz, a.conf, a.max_frames), "frames", flush=True)


if __name__ == "__main__":
    main()
