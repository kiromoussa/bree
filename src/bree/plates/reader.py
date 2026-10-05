"""Plate detector + plate text reader behind one small interface.

    reader = make_reader()                    # both plate detectors + open text reader, CPU
    reads = reader.read(frame)                # list[PlateRead], best first
    reads = reader.read(frame, roi=car_box)   # only look inside a vehicle box

Parts (each replaceable, see the two Protocols):
- detector "open":      open-image-models, YOLOv9-t plate detector, ONNX. Package licence MIT.
- detector "classical": OpenCV only (black-hat + gradient + contour shape). No model weights at all.
- ocr "open":           fast-plate-ocr, CCT model, ONNX. Package licence MIT.
Licence notes and what counsel should check are in README.md.

The model packages are optional: `uv pip install --no-deps fast-plate-ocr==1.1.0 open-image-models==0.6.0`
plus `rich tqdm`. Weights download on first use from the projects' GitHub releases (no sign-up) to
~/.cache/fast-plate-ocr and ~/.cache/open-image-models.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

Box = tuple[int, int, int, int]
CPU = ["CPUExecutionProvider"]     # CoreML crashes at interpreter exit with these graphs on this Mac


@dataclass
class PlateRead:
    text: str                      # A-Z 0-9, no spaces; "" when nothing readable
    conf: float                    # mean character probability, 0 to 1
    char_conf: list[float]
    box: Box                       # plate box in the frame passed to read()
    det_conf: float = 1.0
    region: str | None = None      # the OCR model's country guess, when it has one


class PlateDetector(Protocol):
    def detect(self, img: np.ndarray) -> list[tuple[Box, float]]: ...


class PlateOcr(Protocol):
    def recognise(self, crop: np.ndarray) -> tuple[str, list[float], str | None]: ...


class OpenDetector:
    def __init__(self, model: str = "yolo-v9-t-384-license-plate-end2end", conf: float = 0.3):
        from open_image_models import create_detector
        self.det = create_detector(model, conf_thresh=conf, providers=CPU)

    def detect(self, img):
        return [((r.bounding_box.x1, r.bounding_box.y1, r.bounding_box.x2, r.bounding_box.y2), float(r.confidence))
                for r in self.det.predict(img)]


class ClassicalDetector:
    """Plate localiser with OpenCV only.

    A plate is a small region that is (a) rectangular with the 2:1 plate shape and (b) dense in
    vertical strokes. Shape comes from closed contours of the edge map, strokes from the Sobel-x
    gradient. Two candidate sources are merged: quadrilateral contours, and blobs of the closed
    gradient image (the textbook black-hat pipeline).
    """
    # ponytail: fixed kernel sizes tuned for plates 60 to 250 px wide at up to 25 degrees of yaw;
    # a trained detector (OpenDetector) is the upgrade path for anything harder.

    def __init__(self, min_w: int = 40, max_w: int = 420, max_out: int = 3):
        self.min_w, self.max_w, self.max_out = min_w, max_w, max_out

    def detect(self, img):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        grey = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grey)      # night frames
        smooth = cv2.bilateralFilter(grey, 5, 40, 5)
        gx = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3))
        gx = gx / (gx.max() + 1e-6)
        boxes: list[Box] = []
        edges = cv2.Canny(smooth, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            boxes.append(cv2.boundingRect(c))
        bh = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7)))
        th = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7)))
        for hat in (bh, th):                                 # dark text on light, light text on dark
            g = np.abs(cv2.Sobel(hat, cv2.CV_32F, 1, 0, ksize=3))
            g = (255 * g / (g.max() + 1e-6)).astype(np.uint8)
            g = cv2.morphologyEx(cv2.GaussianBlur(g, (5, 5), 0), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)))
            _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                boxes.append(cv2.boundingRect(c))
        integral = cv2.integral(gx)
        scored = []
        for x, y, w, h in boxes:
            if not (self.min_w <= w <= self.max_w and 1.4 <= w / max(h, 1) <= 4.5 and h >= 12):
                continue
            dens = (integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]) / (w * h)
            # Strokes should fill the middle band (the serial), and the box should be near 2:1.
            shape = 1.0 - min(1.0, abs(w / h - 2.0) / 2.5)
            scored.append((float(dens) * (0.5 + 0.5 * shape), (x, y, x + w, y + h)))
        scored.sort(reverse=True)
        out: list[tuple[Box, float]] = []
        for s, b in scored:
            if all(_iou(b, o) < 0.3 for o, _ in out):
                out.append((b, min(1.0, s * 6)))
            if len(out) >= self.max_out:
                break
        return out


class BothDetectors:
    """Candidates from the open detector and the OpenCV localiser together; the pipeline keeps the
    candidate the text reader is most sure about. Costs one extra text read per extra candidate."""

    def __init__(self, *detectors: PlateDetector):
        self.detectors = detectors or (OpenDetector(), ClassicalDetector())

    def detect(self, img):
        return [d for det in self.detectors for d in det.detect(img)]


class OpenOcr:
    def __init__(self, model: str = "cct-s-v2-global-model"):
        from fast_plate_ocr import LicensePlateRecognizer
        self.m = LicensePlateRecognizer(model, providers=CPU)
        self.pad = self.m.config.pad_char

    def recognise(self, crop):
        p = self.m.run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if self.m.config.image_color_mode == "rgb"
                       else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), return_confidence=True)[0]
        keep = [(c, float(q)) for c, q in zip(p.plate, p.char_probs) if c != self.pad]
        return "".join(c for c, _ in keep), [q for _, q in keep], getattr(p, "region", None)


class PlatePipeline:
    def __init__(self, detector: PlateDetector, ocr: PlateOcr, pad: float = 0.06, min_plate_w: int = 24):
        self.detector, self.ocr, self.pad, self.min_plate_w = detector, ocr, pad, min_plate_w

    def read_crop(self, crop: np.ndarray, box: Box = (0, 0, 0, 0), det_conf: float = 1.0) -> PlateRead:
        text, cc, region = self.ocr.recognise(crop)
        return PlateRead(text, float(np.mean(cc)) if cc else 0.0, cc, box, det_conf, region)

    def read(self, frame: np.ndarray, roi: Box | None = None) -> list[PlateRead]:
        H, W = frame.shape[:2]
        ox, oy, rx2, ry2 = (0, 0, W, H) if roi is None else (max(0, int(roi[0])), max(0, int(roi[1])),
                                                              min(W, int(roi[2])), min(H, int(roi[3])))
        view = frame[oy:ry2, ox:rx2]
        if view.size == 0:
            return []
        out = []
        for (x1, y1, x2, y2), dc in self.detector.detect(view):
            if x2 - x1 < self.min_plate_w:
                continue
            px, py = int(self.pad * (x2 - x1)), int(self.pad * (y2 - y1))
            crop = view[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
            if crop.size:
                out.append(self.read_crop(crop, (x1 + ox, y1 + oy, x2 + ox, y2 + oy), dc))
        out.sort(key=lambda r: r.conf * r.det_conf * (len(r.text) >= 4), reverse=True)
        return out


def make_reader(detector: str = "both", ocr: str = "open", **kw) -> PlatePipeline:
    """detector: "open" | "classical" | "both"; ocr: "open". Raises ImportError with the install line when
    an open model package is missing."""
    try:
        det = {"open": OpenDetector, "classical": ClassicalDetector, "both": BothDetectors}[detector]()
        return PlatePipeline(det, OpenOcr() if ocr == "open" else ocr, **kw)
    except ImportError as e:
        raise ImportError(f"{e}. Install: uv pip install --no-deps fast-plate-ocr==1.1.0 "
                          "open-image-models==0.6.0 && uv pip install rich tqdm") from e


def vote(reads: list[PlateRead], min_conf: float = 0.5) -> tuple[str, float]:
    """One plate text from several frames of the same vehicle: reads of the most common length,
    then a confidence-weighted vote per character. Returns (text, confidence) or ("", 0.0)."""
    reads = [r for r in reads if r.text and r.conf >= min_conf]
    if not reads:
        return "", 0.0
    n = Counter(len(r.text) for r in reads).most_common(1)[0][0]
    same = [r for r in reads if len(r.text) == n]
    text, confs = "", []
    for i in range(n):
        w: dict[str, float] = {}
        for r in same:
            w[r.text[i]] = w.get(r.text[i], 0.0) + r.char_conf[i]
        ch = max(w, key=w.get)
        text += ch
        confs.append(w[ch] / sum(w.values()) * max(r.char_conf[i] for r in same if r.text[i] == ch))
    return text, float(np.mean(confs))


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _iou(a: Box, b: Box) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
