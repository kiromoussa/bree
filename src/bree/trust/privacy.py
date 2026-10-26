"""Head and face pixelation for every image and clip BREE stores.

Three layers, any one of which is enough to hide a face the other two missed:
  1. person boxes: the head from pose keypoints, else the top of the person box (`head_regions`);
  2. a face detector (YuNet, weights/face_detection_yunet_2023mar.onnx, MIT) run on the whole frame, so a person the
     person detector missed (edge of frame, behind glass, half hidden) is still covered;
  3. a short hold: a region stays for a few frames after it was last seen, so a flickering detection does not
     let one clear frame through.
Regions are pixelated to a 5x5 grid, the same strength as bree.alerts.annotate.blur_heads. Faces on posters, masks on
a wall or a mannequin are also found by the detector; over-blurring is the safe side, so they are pixelated too.

`scan_faces` is the independent check used by tests/test_privacy.py and `bree trust scan`: it runs the face detector at
several scales and a low threshold on a stored image and reports what it still finds.

This is pixelation, not anonymisation in a legal sense: build, clothing, gait and the store itself remain visible.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bree.events.observations import HEAD_KPTS, PersonObs

FACE_MODEL = Path(__file__).resolve().parents[3] / "weights" / "face_detection_yunet_2023mar.onnx"
GRID = 5                      # blocks per side inside a pixelated region

Box = tuple[int, int, int, int]


def _expand(b: Box, pad: float, w: int, h: int) -> Box:
    x1, y1, x2, y2 = b
    dx, dy = (x2 - x1) * pad / 2, (y2 - y1) * pad / 2
    return max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(w, int(x2 + dx + 1)), min(h, int(y2 + dy + 1))


def merge_boxes(boxes: list[Box]) -> list[Box]:
    """Union of boxes that overlap or touch, so each cluster is one clean mosaic instead of mosaics cut by each other."""
    boxes = [tuple(b) for b in boxes if b[2] > b[0] and b[3] > b[1]]
    merged = True
    while merged:
        merged = False
        out: list[Box] = []
        for b in boxes:
            for i, o in enumerate(out):
                if b[0] <= o[2] and o[0] <= b[2] and b[1] <= o[3] and o[1] <= b[3]:
                    out[i] = (min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3]))
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def pixelate(img: np.ndarray, boxes: list[Box], grid: int = GRID) -> int:
    """Pixelate each box in place to a grid x grid mosaic. Returns how many boxes were changed."""
    h, w = img.shape[:2]
    n = 0
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        small = cv2.resize(img[y1:y2, x1:x2], (grid, grid), interpolation=cv2.INTER_AREA)
        img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        n += 1
    return n


def head_regions(persons: list[PersonObs], w: int, h: int, profile: str = "standard", kpt_conf: float = 0.25) -> list[Box]:
    """Where heads are, from person boxes and pose. profile: 'standard' (head from keypoints or the top of the box),
    'overhead' (ceiling camera: the head is not at the top of the box, so the upper 55% of the box is covered)."""
    out: list[Box] = []
    for p in persons:
        x1, y1, x2, y2 = p.bbox
        bw, bh = x2 - x1, y2 - y1
        if profile == "overhead":
            out.append(_expand((int(x1), int(y1), int(x2), int(y1 + 0.55 * bh)), 0.2, w, h))
            continue
        pts = [q for q in (p.kpt(i, kpt_conf) for i in HEAD_KPTS) if q is not None]
        # The box-top fallback also covers keypoint heads: pose can place a nose a little off a turned head.
        side = max(0.30 * bh, 0.0)
        cx = (x1 + x2) / 2
        fb = (int(cx - min(side, bw) / 2 - 4), int(y1 - 4), int(cx + min(side, bw) / 2 + 4), int(y1 + side * 0.9))
        if len(pts) >= 2:
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            r = max(max(xs) - min(xs), max(ys) - min(ys), 0.12 * bw) * 1.3 + 6
            kx, ky = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            out.append(_expand((int(kx - r), int(ky - r), int(kx + r), int(ky + r)), 0.0, w, h))
            # a standing, upright person: also keep the box-top estimate when it overlaps the keypoint head
            out.append(_expand(fb, 0.0, w, h))
        else:
            out.append(_expand(fb, 0.1, w, h))
    return out


class FaceFinder:
    """YuNet face detector (OpenCV FaceDetectorYN). One shared instance is fine: calls are serialised."""

    def __init__(self, model: str | Path = FACE_MODEL):
        self.model = Path(model)
        self.available = self.model.exists()
        self._det = None
        self._lock = threading.Lock()

    def detect(self, img: np.ndarray, width: int = 1280, score: float = 0.5) -> list[tuple[Box, float]]:
        if not self.available:
            return []
        h, w = img.shape[:2]
        s = min(1.0, width / w)
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s)))) if s < 1.0 else img
        with self._lock:
            if self._det is None:
                self._det = cv2.FaceDetectorYN.create(str(self.model), "", (320, 320), score, 0.3, 5000)
            self._det.setScoreThreshold(score)
            self._det.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self._det.detect(small)
        out = []
        for r in (faces if faces is not None else []):
            x, y, fw, fh = (float(v) / s for v in r[:4])
            out.append(((int(x), int(y), int(x + fw), int(y + fh)), float(r[-1])))
        return out


_FINDER: FaceFinder | None = None


def default_finder() -> FaceFinder:
    global _FINDER
    if _FINDER is None:
        _FINDER = FaceFinder()
    return _FINDER


@dataclass
class _Held:
    box: Box
    ttl: int


class Guard:
    """Per-stream pixelation. Call `protect(frame, persons)` on every frame you are about to store; it returns the frame
    with heads and faces pixelated (in place unless inplace=False). One Guard per camera: it remembers regions."""

    def __init__(self, face_every: int = 2, face_width: int = 1280, face_score: float = 0.5, hold: int = 6,
                 profile: str = "standard", finder: FaceFinder | None = None):
        self.face_every, self.face_width, self.face_score = max(1, face_every), face_width, face_score
        self.hold, self.profile = hold, profile
        self.finder = finder or default_finder()
        self._held: list[_Held] = []
        self._n = 0
        self.stats = {"frames": 0, "person_regions": 0, "face_regions": 0, "held_regions": 0}

    def regions(self, img: np.ndarray, persons: list[PersonObs] | None) -> list[Box]:
        h, w = img.shape[:2]
        boxes = head_regions(persons or [], w, h, self.profile)
        self.stats["person_regions"] += len(boxes)
        for hd in self._held:
            hd.ttl -= 1
        self._held = [x for x in self._held if x.ttl > 0]
        held = [_expand(x.box, 0.4, w, h) for x in self._held]
        self.stats["held_regions"] += len(held)
        boxes += held
        if self._n % self.face_every == 0:
            faces = [_expand(b, 0.5, w, h) for b, _ in self.finder.detect(img, self.face_width, self.face_score)]
            self.stats["face_regions"] += len(faces)
            boxes += faces
            self._held += [_Held(b, self.hold) for b in faces]
        self._n += 1
        return boxes

    def protect(self, img: np.ndarray, persons: list[PersonObs] | None = None, inplace: bool = True) -> np.ndarray:
        out = img if inplace else img.copy()
        pixelate(out, merge_boxes(self.regions(out, persons)))
        self.stats["frames"] += 1
        return out


def still_guard(profile: str = "standard") -> Guard:
    """A guard for a single image (an evidence still, a shelf crop): face detector on this very frame, full width."""
    return Guard(face_every=1, face_width=2560, face_score=0.4, hold=1, profile=profile)


def scan_faces(img: np.ndarray, finder: FaceFinder | None = None, score: float = 0.35,
               widths: tuple[int, ...] = (640, 960, 1280, 1920, 2560)) -> list[tuple[Box, float, int]]:
    """Independent audit of a stored image: faces a detector can still find at several scales (box, score, width)."""
    finder = finder or default_finder()
    hits = []
    for wd in sorted({min(wd, img.shape[1]) for wd in widths}):   # widths above the image's own are one scan
        hits += [(b, s, wd) for b, s in finder.detect(img, wd, score)]
    return hits


def detail_ratio(before: np.ndarray, after: np.ndarray, box: Box) -> float:
    """Pixel standard deviation inside box after / before: a clean mosaic of a detailed region is far below 1."""
    x1, y1, x2, y2 = box
    b, a = before[y1:y2, x1:x2].astype(float), after[y1:y2, x1:x2].astype(float)
    return float(a.std() / max(b.std(), 1e-6))


def pixelated_fraction(img: np.ndarray, box: Box, grid: int = GRID) -> float:
    """1.0 when a region is a clean grid x grid mosaic (at most grid*grid distinct colours, lossless images only),
    falling towards 0 as it carries more detail. Used by tests; a JPEG's ringing breaks the exact count."""
    x1, y1, x2, y2 = box
    roi = img[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return 1.0
    colours = len(np.unique(roi.reshape(-1, roi.shape[-1]), axis=0))
    return min(1.0, grid * grid / colours)
