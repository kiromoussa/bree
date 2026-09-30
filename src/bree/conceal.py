"""Pose-sequence concealment / shoplifting classifier, trained and evaluated on PoseLift.

PoseLift (TeCSAR-UNCC, WACV 2025, Apache-2.0): real retail CCTV, pose only.
  Pickle_files/{Train,Test}/<cam>_<vid>.pkl : dict[frame] -> dict[person_id] -> [bbox_xyxy, kps(17,3) as (y, x, conf)]
  Pickle_files/GT/<cam>_<vid>.npy          : per-frame 0/1 shoplifting label (Test only; Train is all normal)
Labels are per frame, not per person: in a labelled frame every visible person counts as positive.

Two scorers, same output (a score per person per frame, causal: only past frames are used):
  rule   the event engine's torso-box geometry without the product signal: fraction of the last
         WINDOW frames with a wrist inside the shoulders-to-hips box (PersonObs.torso_box).
  model  a small temporal CNN over bbox-normalised keypoints + velocities.
A frame's score is the max over the people in it (frame-level protocol of the PoseLift paper).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bree.events.observations import L_WRIST, R_WRIST, PersonObs

WINDOW = 24          # frames (1.6 s at PoseLift's 15 fps)
FEAT = 17 * 3 + 17 * 2


@dataclass
class Track:
    frames: np.ndarray   # (T,) frame indices, sorted
    kps: np.ndarray      # (T, 17, 3) x, y, conf
    bbox: np.ndarray     # (T, 4) x1, y1, x2, y2


def clean_kps(kps: np.ndarray) -> np.ndarray:
    """PoseLift ships some NaN keypoints: treat them as not detected (conf 0)."""
    bad = np.isnan(kps).any(-1)
    kps = np.nan_to_num(kps)
    kps[bad] = 0.0
    return kps


@dataclass
class Video:
    name: str            # e.g. "3_262"
    cam: str
    split: str           # "train" (all normal) or "test" (labelled)
    n_frames: int
    tracks: dict[int, Track]
    labels: np.ndarray   # (n_frames,) 0/1


def load_poselift(root: Path) -> list[Video]:
    root = Path(root)
    out = []
    for split, sub in (("train", "Train"), ("test", "Test")):
        for f in sorted((root / sub).glob("*.pkl")):
            d = pickle.load(open(f, "rb"))
            per: dict[int, list] = {}
            for fr, people in d.items():
                for pid, (bb, kp) in people.items():
                    per.setdefault(int(pid), []).append((int(fr), bb, kp))
            tracks = {}
            for pid, rows in per.items():
                rows.sort(key=lambda r: r[0])
                kps = clean_kps(np.asarray([r[2] for r in rows], dtype=np.float32)[:, :, [1, 0, 2]])   # (y,x,c) -> (x,y,c)
                tracks[pid] = Track(np.array([r[0] for r in rows]), kps,
                                    np.asarray([r[1] for r in rows], dtype=np.float32))
            n = max(d) + 1 if d else 0
            if split == "test":
                g = np.load(root / "GT" / f"{f.stem}.npy").astype(np.int8)
                labels = np.zeros(max(n, len(g)), np.int8)
                labels[:len(g)] = g
                n = len(labels)
            else:
                labels = np.zeros(n, np.int8)
            out.append(Video(f.stem, f.stem.split("_")[0], split, n, tracks, labels))
    return out


def load_retails(json_dir: Path, gt_dir: Path | None = None, split: str = "test", limit: int | None = None) -> list[Video]:
    """RetailS (TeCSAR-UNCC): one JSON per video, {person_id: {frame: {"keypoints": [51 floats as (y, x, c)]}}},
    no boxes. Labels in gt_dir/<name>.npy for the test sets. Same cameras/format as PoseLift."""
    import json
    out = []
    for f in sorted(Path(json_dir).glob("*.json"))[:limit]:
        d = json.load(open(f))
        tracks, n = {}, 0
        for pid, frames in d.items():
            fr = sorted(frames, key=int)
            if not fr:
                continue
            kps = clean_kps(np.asarray([frames[k]["keypoints"] for k in fr], np.float32).reshape(-1, 17, 3)[:, :, [1, 0, 2]])
            idx = np.array([int(k) for k in fr])
            tracks[int(pid)] = Track(idx, kps, kpt_box(kps))
            n = max(n, idx.max() + 1)
        labels = np.zeros(n, np.int8)
        if gt_dir is not None:
            g = np.load(Path(gt_dir) / f"{f.stem}.npy").astype(np.int8)
            labels = np.zeros(max(n, len(g)), np.int8)
            labels[:len(g)] = g
        out.append(Video(f.stem, f.stem.split("_")[0].lstrip("0"), split, len(labels), tracks, labels))
    return out


# ---------- features ----------

def kpt_box(kps: np.ndarray, min_conf: float = 0.1) -> np.ndarray:
    """(T, 4) box around the confident keypoints (all keypoints if none are confident). Datasets differ
    in whether they ship detector boxes (RetailS doesn't), so features never depend on them."""
    ok = kps[:, :, 2] >= min_conf
    ok[~ok.any(1)] = True
    x = np.where(ok, kps[:, :, 0], np.nan)
    y = np.where(ok, kps[:, :, 1], np.nan)
    return np.stack([np.nanmin(x, 1), np.nanmin(y, 1), np.nanmax(x, 1), np.nanmax(y, 1)], 1).astype(np.float32)


def track_features(t: Track) -> np.ndarray:
    """(T, FEAT): keypoints relative to the keypoint-box centre / height, conf, and per-frame velocity."""
    x1, y1, x2, y2 = kpt_box(t.kps).T
    h = np.maximum(y2 - y1, 1.0)[:, None]
    cx, cy = ((x1 + x2) / 2)[:, None], ((y1 + y2) / 2)[:, None]
    nx, ny = (t.kps[:, :, 0] - cx) / h, (t.kps[:, :, 1] - cy) / h
    conf = t.kps[:, :, 2]
    pos = np.stack([nx, ny], -1)
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    return np.concatenate([pos.reshape(len(pos), -1), conf, vel.reshape(len(pos), -1)], 1).astype(np.float32)


def windows(feat: np.ndarray, win: int = WINDOW) -> np.ndarray:
    """Causal windows ending at every frame, left-padded by repeating the first frame: (T, win, FEAT)."""
    pad = np.concatenate([np.repeat(feat[:1], win - 1, 0), feat], 0)
    idx = np.arange(len(feat))[:, None] + np.arange(win)[None, :]
    return pad[idx]


def rule_scores(t: Track, win: int = WINDOW, margin: float = 0.25, kpt_conf: float = 0.3) -> np.ndarray:
    """Fraction of the last `win` frames with a wrist inside the torso box (engine geometry, no products)."""
    inside = np.zeros(len(t.frames), np.float32)
    for i in range(len(t.frames)):
        p = PersonObs(0, tuple(t.bbox[i]), keypoints=t.kps[i])
        bx1, by1, bx2, by2 = p.torso_box(margin, kpt_conf)
        for w in (L_WRIST, R_WRIST):
            q = p.kpt(w, kpt_conf)
            if q and bx1 <= q[0] <= bx2 and by1 <= q[1] <= by2:
                inside[i] = 1
    c = np.concatenate([[0], np.cumsum(inside)])
    lo = np.maximum(np.arange(len(inside)) - win + 1, 0)
    return (c[1:] - c[lo]) / win


def frame_scores(v: Video, per_track) -> np.ndarray:
    """Max over people of per_track(track) -> (T,) scores, placed on the video's frame axis."""
    s = np.zeros(v.n_frames, np.float32)
    for t in v.tracks.values():
        sc = per_track(t)
        ok = t.frames < v.n_frames
        np.maximum.at(s, t.frames[ok], sc[ok])
    return s


# ---------- model ----------

def build_model():
    import torch.nn as nn

    class ConcealNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(FEAT, 64, 5, padding=2), nn.ReLU(), nn.Dropout(0.2),
                nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(64, 1))

        def forward(self, x):                    # x: (B, win, FEAT)
            return self.net(x.transpose(1, 2)).squeeze(1)

    return ConcealNet()


def training_windows(videos: list[Video], stride: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Windows ending every `stride` frames (neighbouring windows are near-duplicates)."""
    X, y = [], []
    for v in videos:
        for t in v.tracks.values():
            ok = np.flatnonzero(t.frames < v.n_frames)[::stride]
            X.append(windows(track_features(t))[ok])
            y.append(v.labels[t.frames[ok]])
    return np.concatenate(X), np.concatenate(y).astype(np.float32)


def train_model(videos: list[Video], epochs: int = 20, seed: int = 0, device: str = "cpu", batch: int = 512):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, y = training_windows(videos)
    mu, sd = X.reshape(-1, FEAT).mean(0), X.reshape(-1, FEAT).std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd)
    Y = torch.tensor(y)
    m = build_model().to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    pos_w = torch.tensor(max((1 - y.mean()) / max(y.mean(), 1e-3), 1.0), device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(len(X)), max(len(X) // batch, 1)):
            opt.zero_grad()
            loss = lossf(m(Xn[b].to(device)), Y[b].to(device))
            loss.backward()
            opt.step()
    m.eval()
    return {"model": m.cpu(), "mu": mu, "sd": sd}


def model_track_scorer(bundle):
    import torch

    def score(t: Track) -> np.ndarray:
        w = (windows(track_features(t)) - bundle["mu"]) / bundle["sd"]
        with torch.no_grad():
            return torch.sigmoid(bundle["model"](torch.tensor(w, dtype=torch.float32))).numpy()
    return score


def save_bundle(bundle, path: Path, meta: dict):
    import torch
    torch.save({"state": bundle["model"].state_dict(), "mu": bundle["mu"], "sd": bundle["sd"], "meta": meta}, path)


def load_bundle(path: Path):
    import torch
    d = torch.load(path, weights_only=False)
    m = build_model()
    m.load_state_dict(d["state"])
    m.eval()
    return {"model": m, "mu": d["mu"], "sd": d["sd"], "meta": d["meta"]}


# ---------- triggers (what an operator would see) ----------

THRESH = {"model": (0.5, 0.7, 0.9), "rule": (0.25, 0.5, 0.75)}   # fixed up front, never tuned on eval data
MIN_RUN, MERGE_S = 8, 10.0                                       # 0.5 s over threshold at 15 fps; merge within 10 s


def trigger_frames(scores: np.ndarray, frames: np.ndarray, th: float, fps: float = 15.0) -> list[int]:
    """Frames where one track's score has stayed >= th for MIN_RUN frames; triggers closer than MERGE_S merge."""
    out, run, last_t = [], 0, -1e9
    for s, f in zip(scores, frames):
        run = run + 1 if s >= th else 0
        if run == MIN_RUN:
            if f / fps - last_t > MERGE_S:
                out.append(int(f))
            last_t = f / fps
    return out


def video_triggers(v: Video, per_track, th: float) -> list[int]:
    return sorted(f for t in v.tracks.values() if len(t.frames) for f in trigger_frames(per_track(t), t.frames, th))


def event_hit(v: Video, per_track, th: float, pad: int = 15) -> bool:
    """Shoplifting video: did any trigger land inside a labelled interval (+-pad frames)?"""
    pos = np.flatnonzero(v.labels)
    return any(((pos >= f - pad) & (pos <= f + pad)).any() for f in video_triggers(v, per_track, th))


# ---------- metrics ----------

def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling."""
    from scipy.stats import rankdata   # scipy ships with ultralytics' deps
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision; tied scores form one threshold step (order-independent)."""
    o = np.argsort(-s, kind="stable")
    y, s = y[o].astype(bool), s[o]
    last = np.r_[s[1:] != s[:-1], True]             # last index of each tie group
    tp = np.cumsum(y)[last]
    n = np.flatnonzero(last) + 1
    new_tp = np.diff(np.r_[0, tp])
    return float((new_tp * tp / n).sum() / max(y.sum(), 1))


def eer(y: np.ndarray, s: np.ndarray) -> float:
    ths = np.unique(s)
    best, gap = (1.0, 1.0), np.inf
    for th in ths:
        fpr = ((s >= th) & (y == 0)).sum() / max((y == 0).sum(), 1)
        fnr = ((s < th) & (y == 1)).sum() / max((y == 1).sum(), 1)
        if abs(fpr - fnr) < gap:
            best, gap = (fpr, fnr), abs(fpr - fnr)
    return float((best[0] + best[1]) / 2)


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {"auc_roc": roc_auc(y, s), "auc_pr": pr_auc(y, s), "eer": eer(y, s),
            "n_frames": int(len(y)), "n_pos": int(y.sum())}
