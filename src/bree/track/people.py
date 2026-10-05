"""Person boxes with pose keypoints for the people cameras (overhead, entrance), from the repo's own person path
(bree.detect.yolo: one detector pass, then top-down pose on each person crop). No tracker: bree.track.floor does
the tracking on the floor plan. It places a person by the hips and shoulders, which a plain box cannot give when a
gondola hides the legs.

    python -m bree.track.people <clip folder> <out folder> [--cameras TRACK-1,TRACK-2] [--max-frames 100]
writes <out>/pipeline/frames_<camera>.jsonl (the frame log format bree.sim.bench reads), so a tracker run can be
repeated without running the model again. A full frame pose model (one pass, no crops) was tried first and found
fewer of the small overhead figures (DEV clip 7001, TRACK-1: 450 boxes on a shopper against 1230), so it is not used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PEOPLE_KINDS = ("overhead", "entrance")


def person_detector(imgsz: int | None = None, conf: float = 0.3):
    """image -> [{"bbox": [x0, y0, x1, y1], "conf": c, "kpts": [[x, y, conf] * 17]}]."""
    from bree.cli import _weights
    from bree.detect.yolo import YoloBackend
    from bree.hw import detect_hardware
    hw = detect_hardware()
    backend = YoloBackend(_weights(hw.pose_model), _weights(hw.detect_model), {}, device=hw.device, imgsz=imgsz or hw.imgsz, products=False, person_conf=conf)

    def detect(image: np.ndarray) -> list[dict]:
        p, _ = backend(image)
        k = p.keypoints if p.keypoints is not None else [None] * len(p)
        return [{"bbox": [round(float(v), 1) for v in b], "conf": round(float(c), 3), **({"kpts": [[round(float(x), 1), round(float(y), 1), round(float(s), 2)] for x, y, s in kk]} if kk is not None else {})}
                for b, c, kk in zip(p.boxes, p.confs, k)]
    return detect


def appearance(image: np.ndarray, det: dict) -> list[float]:
    """Clothing colour of one person box: hue and saturation histograms of the upper and the lower body (bree.track.reid,
    shoulders down, never the head), 104 numbers. bree.track.floor uses it to keep two people apart who pass each other."""
    from bree.track.reid import part_colors
    kp = np.asarray(det["kpts"], float) if det.get("kpts") is not None else None
    return [round(float(v), 3) for v in part_colors(image, det["bbox"], kp)[:2].ravel()]


def frames(clip: Path, cam: str):
    """Frames of one camera: <clip>/<cam>.mp4, or a folder of rgb_0001.png (the older clip format)."""
    import cv2
    if (clip / f"{cam}.mp4").exists():
        cap = cv2.VideoCapture(str(clip / f"{cam}.mp4"))
        while True:
            ok, im = cap.read()
            if not ok:
                break
            yield im
        cap.release()
    else:
        for p in sorted((clip / cam).glob("rgb_*.png")):
            yield cv2.imread(str(p))


def people_cameras(clip: Path) -> list[str]:
    meta = json.loads((clip / "clip.json").read_text())
    if (clip / "calibration.json").exists():
        kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
        return [c for c in meta["cameras"] if kind[c] in PEOPLE_KINDS]
    return [c for c in meta["cameras"] if c.startswith(("TRACK", "ENTRANCE"))]


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--cameras")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args(argv)
    clip, out = Path(args.clip), Path(args.out) / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    fps = float(json.loads((clip / "clip.json").read_text())["fps"])
    detect = person_detector(imgsz=args.imgsz, conf=args.conf)
    for cam in (args.cameras.split(",") if args.cameras else people_cameras(clip)):
        dst = out / f"frames_{cam}.jsonl"
        if dst.exists():
            print(f"{cam}: already there", flush=True)
            continue
        n = boxes = 0
        with (out / f"frames_{cam}.jsonl.part").open("w") as fh:
            for f, im in enumerate(frames(clip, cam)):
                if args.max_frames and f >= args.max_frames:
                    break
                ps = [{**p, "app": appearance(im, p)} for p in detect(im)]
                fh.write(json.dumps({"frame": f, "t": round(f / fps, 3), "persons": [{"id": i, **p} for i, p in enumerate(ps)], "products": []}) + "\n")
                n, boxes = n + 1, boxes + len(ps)
        (out / f"frames_{cam}.jsonl.part").rename(dst)
        print(f"{cam}: {n} frames, {boxes} person boxes", flush=True)


if __name__ == "__main__":
    main()
