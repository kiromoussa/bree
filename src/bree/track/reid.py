"""Person re-identification from the body, never the face.

Used to keep one identity on a shopper across occlusions (engine stitching), across cameras (floor-plane
handoff) and from shelf to register to door, which is what basket minus paid depends on.

Cues, all computed on the body below the shoulder line (the head is cut off AND any head keypoint inside
the crop is greyed out before anything is computed, so no face pixel reaches a feature):
  emb     appearance embedding: DINOv2 ViT-S/14 (Apache-2.0 code and weights,
          github.com/facebookresearch/dinov2), run from ONNX on the body crop (224x112)
  colors  clothing colour per body part from the pose keypoints: upper (shoulders to hips), lower (hips to
          ankles), shoes. Hue x saturation histogram for coloured pixels + 4 brightness bins for grey/black/white
          ones, so a lighting change moves mass between neighbouring bins instead of changing the colour
  shape   shoulder/hip width ratio and torso/leg length ratio (scale free; weak, view dependent)
  bag     a backpack / handbag / suitcase box overlapping the person (COCO classes of the detector)
  speed   walking speed in body heights per second (weak gait cue)
One logistic score fuses them (`match_prob`, weights fitted by scripts/reid_bench.py, leave-one-sequence-out
on MOT16). Position and time are NOT in the score: callers gate on them (stitch distance / walking speed).

Privacy: features exist only in memory (PersonObs.reid, per-track and per-person galleries) and are dropped
when the visit ends or after `max_age_s` (default 2 h). With closed-world identity a lost shopper's gallery is
kept until they exit or have not been seen for `closed_world_timeout_s` (default 60 min), then cleared.
Nothing here writes to disk or the network, and nothing links one visit to another. See DECISIONS.md,
"Re-identification without the face" and "Closed-world identity".
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bree.events.observations import HEAD_KPTS, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

L_ANKLE, R_ANKLE = 15, 16
CROP_HW = (224, 112)
ONNX_PATH = Path(__file__).resolve().parents[3] / "models" / "dinov2_vits14_reid.onnx"


@dataclass
class ReidFeatures:
    t: float
    emb: np.ndarray | None = None          # L2-normalised appearance embedding
    colors: np.ndarray | None = None       # (3, 52): upper, lower, shoes; a zero row = part not visible
    shape: np.ndarray | None = None        # [log shoulder/hip width, log torso/leg length]
    bag: bool | None = None                # None = unknown (no carried-object detections)
    speed: float | None = None             # body heights / s


# ------------------------------------------------------------------ crops


def _kp_y(kp, idx, conf):
    ys = [kp[i, 1] for i in idx if kp is not None and kp[i, 2] >= conf]
    return float(np.mean(ys)) if ys else None


def body_parts(bbox, kp, img_hw, conf: float = 0.3) -> dict[str, tuple[int, int, int, int]]:
    """Pixel boxes for body (shoulders down), upper, lower and shoes. Fixed fractions of the box when the
    keypoints are missing. The body box never starts above the shoulder line."""
    H, W = img_hw
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1.0)
    sh = _kp_y(kp, (L_SHOULDER, R_SHOULDER), conf)
    hip = _kp_y(kp, (L_HIP, R_HIP), conf)
    ank = _kp_y(kp, (L_ANKLE, R_ANKLE), conf)
    sh = sh if sh is not None and y1 <= sh < y2 else y1 + 0.2 * h
    hip = hip if hip is not None and sh < hip < y2 else sh + 0.4 * h
    ank = ank if ank is not None and hip < ank <= y2 + 0.05 * h else y1 + 0.93 * h
    cw = 0.2 * (x2 - x1)                    # central 60% of the width: less background
    def box(a, b, inner=True):
        xa, xb = (x1 + cw, x2 - cw) if inner else (x1, x2)
        return (int(max(0, xa)), int(max(0, a)), int(min(W, xb)), int(min(H, b)))
    return {"body": box(sh, y2, inner=False), "upper": box(sh, hip), "lower": box(hip, ank - 0.03 * h),
            "shoes": box(ank - 0.06 * h, min(y2, ank + 0.04 * h) if ank < y2 else y2)}


def body_crop(image: np.ndarray, bbox, kp, conf: float = 0.3) -> np.ndarray | None:
    """Shoulders-down crop with head keypoints greyed out (a bowed head can dip below the shoulder line)."""
    x1, y1, x2, y2 = body_parts(bbox, kp, image.shape[:2], conf)["body"]
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2].copy()
    if kp is not None:
        head = [(kp[i, 0], kp[i, 1]) for i in HEAD_KPTS if kp[i, 2] >= conf]
        if head:
            sw = abs(kp[L_SHOULDER, 0] - kp[R_SHOULDER, 0]) if min(kp[L_SHOULDER, 2], kp[R_SHOULDER, 2]) >= conf else 0
            r = max(0.6 * sw, 0.12 * (bbox[3] - bbox[1]), 6)
            hx = [p[0] for p in head]; hy = [p[1] for p in head]
            a, b = int(min(hx) - r - x1), int(max(hx) + r - x1)
            c, d = int(min(hy) - r - y1), int(max(hy) + r - y1)
            crop[max(0, c):max(0, d), max(0, a):max(0, b)] = 127
    return crop


# ------------------------------------------------------------------ colour / shape


def hsv_hist(bgr: np.ndarray) -> np.ndarray:
    """52 bins: 12 hue x 4 saturation for coloured pixels, 4 brightness bins for grey ones. Sums to 1."""
    import cv2
    if bgr.size == 0:
        return np.zeros(52, np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    col = (s >= 40) & (v >= 40)
    out = np.zeros(52, np.float32)
    np.add.at(out, (h[col] * 12 // 180) * 4 + np.minimum((s[col] - 40) * 4 // 216, 3), 1)
    np.add.at(out, 48 + np.minimum(v[~col] * 4 // 256, 3), 1)
    return out / max(out.sum(), 1)


def part_colors(image, bbox, kp, conf: float = 0.3) -> np.ndarray:
    parts = body_parts(bbox, kp, image.shape[:2], conf)
    rows = []
    for name in ("upper", "lower", "shoes"):
        x1, y1, x2, y2 = parts[name]
        ok = x2 - x1 >= 2 and y2 - y1 >= 2
        rows.append(hsv_hist(image[y1:y2, x1:x2]) if ok else np.zeros(52, np.float32))
    return np.stack(rows)


def body_shape(kp, conf: float = 0.3) -> np.ndarray | None:
    if kp is None or kp[[5, 6, 11, 12, 15, 16], 2].min() < conf:
        return None
    sw = abs(kp[5, 0] - kp[6, 0]); hw = abs(kp[11, 0] - kp[12, 0])
    torso = (kp[11, 1] + kp[12, 1] - kp[5, 1] - kp[6, 1]) / 2
    leg = (kp[15, 1] + kp[16, 1] - kp[11, 1] - kp[12, 1]) / 2
    if min(sw, hw, torso, leg) < 2:
        return None                         # side view / bent over: widths or lengths collapse
    return np.log(np.array([sw / hw, torso / leg], np.float32))


def carries_bag(bbox, bags: np.ndarray | None) -> bool | None:
    if bags is None:
        return None
    x1, y1, x2, y2 = bbox
    for b in bags:
        ix = max(0, min(x2, b[2]) - max(x1, b[0])) * max(0, min(y2, b[3]) - max(y1, b[1]))
        if ix >= 0.5 * (b[2] - b[0]) * (b[3] - b[1]):
            return True
    return False


# ------------------------------------------------------------------ appearance embedding


class Embedder:
    """DINOv2 ViT-S/14 (Apache-2.0) from ONNX on CPU (or the best ONNX Runtime provider). Exported once from
    torch.hub (github.com/facebookresearch/dinov2) to models/dinov2_vits14_reid.onnx if missing."""
    name = "dinov2_vits14"
    MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
    STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

    def __init__(self, path: str | Path = ONNX_PATH, providers: list | None = None, threads: int = 0):
        import onnxruntime as ort
        path = Path(path)
        if not path.exists():
            export_dinov2(path)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        import cv2
        if not crops:
            return np.zeros((0, 384), np.float32)
        x = np.stack([cv2.resize(c, CROP_HW[::-1], interpolation=cv2.INTER_AREA)[:, :, ::-1] for c in crops])
        x = ((x.astype(np.float32) - self.MEAN) / self.STD).transpose(0, 3, 1, 2)
        e = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-6)


def export_dinov2(dst: Path) -> None:
    import torch
    class Cls(torch.nn.Module):     # forward(x) only (the hub model also takes an optional mask input)
        def __init__(self):
            super().__init__()
            self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

        def forward(self, x):
            return self.m(x)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Cls().eval(), torch.zeros(1, 3, *CROP_HW), str(dst), input_names=["crops"], output_names=["emb"],
                      dynamic_axes={"crops": {0: "n"}, "emb": {0: "n"}}, opset_version=17, dynamo=False)


# ------------------------------------------------------------------ fusion

# Pair features: [embedding cosine, upper / lower / shoes colour similarity, shape difference, bag mismatch,
# speed difference]. NEUTRAL fills a feature that one side lacks. WEIGHTS/BIAS: logistic regression fitted on
# MOT16 stitch-candidate pairs by scripts/reid_bench.py (results/reid_bench.json, "calibration").
FEATURES = ("emb_cos", "upper", "lower", "shoes", "shape_diff", "bag_mismatch", "speed_diff")
NEUTRAL = np.array([0.5, 0.5, 0.5, 0.5, 0.3, 0.0, 0.7], np.float32)
# The speed weight fits to 0: a brand-new track has no motion history at the moment it must be matched, so
# walking speed only acts through the callers' "could they have walked there" gate.
WEIGHTS = np.array([6.2671, 3.4329, 3.1373, 1.141, -0.1538, -0.7199, 0.0], np.float32)
BIAS = -13.0131


def _bhatt(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.sum() < 0.5 or b.sum() < 0.5:
        return None
    return float(np.sqrt(a * b).sum())


def summarize(gallery: list[ReidFeatures]) -> ReidFeatures | None:
    """One descriptor for a person from their recent features (mean embedding / histograms)."""
    if not gallery:
        return None
    embs = [g.emb for g in gallery if g.emb is not None]
    emb = None
    if embs:
        emb = np.mean(embs, axis=0)
        emb = emb / max(np.linalg.norm(emb), 1e-6)
    cols = [g.colors for g in gallery if g.colors is not None]
    colors = None
    if cols:
        c = np.stack(cols)                                   # (n, 3, 52); average only visible parts
        vis = c.sum(2, keepdims=True) > 0.5
        colors = (c * vis).sum(0) / np.maximum(vis.sum(0), 1)
    shapes = [g.shape for g in gallery if g.shape is not None]
    bags = [g.bag for g in gallery if g.bag is not None]
    speeds = [g.speed for g in gallery if g.speed is not None]
    return ReidFeatures(gallery[-1].t, emb, colors, np.median(shapes, axis=0) if shapes else None,
                        (sum(bags) * 2 > len(bags)) if bags else None, float(np.median(speeds)) if speeds else None)


def pair_features(a: ReidFeatures, b: ReidFeatures) -> np.ndarray:
    x = NEUTRAL.copy()
    if a.emb is not None and b.emb is not None:
        x[0] = float(a.emb @ b.emb)
    if a.colors is not None and b.colors is not None:
        for i in range(3):
            s = _bhatt(a.colors[i], b.colors[i])
            if s is not None:
                x[1 + i] = s
    if a.shape is not None and b.shape is not None:
        x[4] = float(np.abs(a.shape - b.shape).sum())
    if a.bag is not None and b.bag is not None:
        x[5] = float(a.bag != b.bag)
    if a.speed is not None and b.speed is not None:
        x[6] = abs(math.log((a.speed + 0.1) / (b.speed + 0.1)))
    return x


def match_prob(gallery: list[ReidFeatures], f: ReidFeatures | list[ReidFeatures] | None,
               weights: np.ndarray | None = None, bias: float | None = None) -> float | None:
    """P(same person) from appearance only. None when either side has no features."""
    b = summarize(f if isinstance(f, list) else [f]) if f is not None else None
    a = summarize(gallery)
    if a is None or b is None:
        return None
    w = WEIGHTS if weights is None else weights
    z = float(pair_features(a, b) @ w + (BIAS if bias is None else bias))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def remember(gallery: list[ReidFeatures], f: ReidFeatures | None, now: float, max_age_s: float,
             keep: int = 8) -> None:
    """Add f (if new) and drop features older than max_age_s; keeps the last `keep`."""
    if f is not None and (not gallery or gallery[-1] is not f):
        gallery.append(f)
    gallery[:] = [g for g in gallery if now - g.t <= max_age_s][-keep:]


# ------------------------------------------------------------------ per-camera extractor


@dataclass
class _TrackState:
    t_last: float
    t_feat: float = -1e9
    feat: ReidFeatures | None = None
    feet: list[tuple[float, float, float, float]] = field(default_factory=list)   # (t, x, y, height)


class ReidExtractor:
    """Attaches `.reid` features to each PersonObs. A track gets features on its first frame and then every
    `every_s` seconds (embedding is the expensive part; colours and shape are cheap). Per-track state is
    dropped `ttl_s` after the track was last seen."""

    def __init__(self, embedder: Embedder | None = None, every_s: float = 0.5, ttl_s: float = 10.0,
                 use_embedding: bool = True, kpt_conf: float = 0.3):
        self.embedder = embedder if embedder is not None or not use_embedding else Embedder()
        self.every_s, self.ttl_s, self.kpt_conf = every_s, ttl_s, kpt_conf
        self.tracks: dict[int, _TrackState] = {}
        self.ms: list[float] = []          # wall ms per extracted crop (all cues), for reporting

    def update(self, image: np.ndarray, persons, t: float, bags: np.ndarray | None = None) -> None:
        due, crops = [], []
        for p in persons:
            st = self.tracks.setdefault(p.track_id, _TrackState(t))
            st.t_last = t
            x1, y1, x2, y2 = p.bbox
            st.feet = [q for q in st.feet if t - q[0] <= 1.0] + [(t, (x1 + x2) / 2, y2, max(y2 - y1, 1.0))]
            if t - st.t_feat >= self.every_s:
                crop = body_crop(image, p.bbox, p.keypoints, self.kpt_conf)
                if crop is not None:
                    due.append((p, st)); crops.append(crop)
        if due:
            t0 = time.perf_counter()
            embs = self.embedder(crops) if self.embedder is not None else [None] * len(crops)
            for (p, st), e in zip(due, embs):
                speed = None
                if len(st.feet) >= 2 and st.feet[-1][0] - st.feet[0][0] >= 0.5:
                    a, b = st.feet[0], st.feet[-1]
                    speed = math.hypot(b[1] - a[1], b[2] - a[2]) / b[3] / (b[0] - a[0])
                st.feat = ReidFeatures(t, e, part_colors(image, p.bbox, p.keypoints, self.kpt_conf),
                                       body_shape(p.keypoints, self.kpt_conf), carries_bag(p.bbox, bags), speed)
                st.t_feat = t
            self.ms.append(1000 * (time.perf_counter() - t0) / len(due))
        for p in persons:
            p.reid = self.tracks[p.track_id].feat
        self.tracks = {k: s for k, s in self.tracks.items() if t - s.t_last <= self.ttl_s}
