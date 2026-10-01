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
FEAT_V2 = FEAT + 12  # + wrist-to-hip/torso geometry (see body_features)
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # COCO-17 left <-> right


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
    fps: float = 15.0


def split_gaps(tracks: dict[int, Track], max_gap: int = 2) -> dict[int, Track]:
    """Split a track wherever frames are missing for more than max_gap frames (PoseLift tracks jump up to
    740 frames), so windows, velocities and trigger runs never span a gap."""
    out = {}
    for pid, t in tracks.items():
        cuts = np.flatnonzero(np.diff(t.frames) > max_gap + 1) + 1
        for k, idx in enumerate(np.split(np.arange(len(t.frames)), cuts)):
            out[pid * 1000 + k] = Track(t.frames[idx], t.kps[idx], t.bbox[idx])
    return out


def occupied(v: Video) -> np.ndarray:
    """(n_frames,) True where at least one pose exists. Empty frames score 0 for free, so AUCs are reported
    on occupied frames (and on all frames for reference)."""
    m = np.zeros(v.n_frames, bool)
    for t in v.tracks.values():
        m[t.frames[t.frames < v.n_frames]] = True
    return m


def incident_chains(videos: list[Video], max_px: float = 20.0) -> dict[str, int]:
    """PoseLift clips are consecutive cuts of continuous recordings: clip N's last pose and clip N+1's first
    pose (same camera, next id) are the same person at the same moment. Join such neighbours into one
    chain id so an incident never sits on both sides of a train/test split."""
    def ends(v):
        firsts = [(t.frames[0], t.kps[0]) for t in v.tracks.values() if len(t.frames)]
        lasts = [(t.frames[-1], t.kps[-1]) for t in v.tracks.values() if len(t.frames)]
        return ([k for f, k in firsts if f == min(f for f, _ in firsts)] if firsts else [],
                [k for f, k in lasts if f == max(f for f, _ in lasts)] if lasts else [])
    by_cam: dict[str, list[Video]] = {}
    for v in videos:
        by_cam.setdefault(v.cam, []).append(v)
    chain, cid = {}, 0
    for cam, vs in by_cam.items():
        vs.sort(key=lambda v: int(v.name.split("_")[1]))
        for a, b in zip([None] + vs[:-1], vs):
            joined = False
            if a is not None and int(b.name.split("_")[1]) == int(a.name.split("_")[1]) + 1:
                _, la = ends(a)
                fb, _ = ends(b)
                for ka in la:
                    for kb in fb:
                        ok = (ka[:, 2] > 0) & (kb[:, 2] > 0)
                        if ok.sum() >= 5 and np.median(np.linalg.norm(ka[ok, :2] - kb[ok, :2], axis=1)) < max_px:
                            joined = True
            if not joined:
                cid += 1
            chain[b.name] = cid
    return chain


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
            tracks = split_gaps(tracks)
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
        tracks = split_gaps(tracks)
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


def track_features(t: Track, version: int = 1) -> np.ndarray:
    """(T, FEAT): keypoints relative to the keypoint-box centre / height, conf, and per-frame velocity."""
    x1, y1, x2, y2 = kpt_box(t.kps).T
    h = np.maximum(y2 - y1, 1.0)[:, None]
    cx, cy = ((x1 + x2) / 2)[:, None], ((y1 + y2) / 2)[:, None]
    nx, ny = (t.kps[:, :, 0] - cx) / h, (t.kps[:, :, 1] - cy) / h
    conf = t.kps[:, :, 2]
    pos = np.stack([nx, ny], -1)
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    base = np.concatenate([pos.reshape(len(pos), -1), conf, vel.reshape(len(pos), -1)], 1).astype(np.float32)
    return np.concatenate([base, body_features(t.kps)], 1) if version == 2 else base


def body_features(kps: np.ndarray) -> np.ndarray:
    """(T, 12) camera-independent hand geometry: each wrist's offset (dx, dy) from the hip centre and from the
    shoulder centre, its distance to the nearer hip, and its speed, all in torso lengths. Concealment is a hand
    going to the pocket / waistband / under the shirt, which these measure directly."""
    xy, c = kps[:, :, :2], kps[:, :, 2:]
    sh, hip = (xy[:, 5] + xy[:, 6]) / 2, (xy[:, 11] + xy[:, 12]) / 2
    torso = np.maximum(np.linalg.norm(sh - hip, axis=1, keepdims=True), 1.0)
    out = []
    for w in (9, 10):
        ok = (c[:, w] > 0.1).astype(np.float32)
        d_hip, d_sh = (xy[:, w] - hip) / torso * ok, (xy[:, w] - sh) / torso * ok
        near = np.minimum(np.linalg.norm(xy[:, w] - xy[:, 11], axis=1), np.linalg.norm(xy[:, w] - xy[:, 12], axis=1))
        speed = np.r_[0, np.linalg.norm(np.diff(xy[:, w], axis=0), axis=1)]
        out += [d_hip, d_sh, (near / torso[:, 0] * ok[:, 0])[:, None], (speed / torso[:, 0] * ok[:, 0])[:, None]]
    return np.concatenate(out, 1).astype(np.float32)


def augment_track(t: Track, rng) -> Track:
    """Training-only: mirror left/right, change speed 0.8-1.25x, drop keypoints, jitter by ~2% of height."""
    kps, frames = t.kps.copy(), t.frames
    if rng.random() < 0.5:
        x1, _, x2, _ = kpt_box(kps).T
        kps = kps[:, FLIP]
        kps[:, :, 0] = (x1 + x2)[:, None] - kps[:, :, 0]
    r = rng.uniform(0.8, 1.25)
    idx = np.clip(np.round(np.arange(0, len(kps), r)).astype(int), 0, len(kps) - 1)
    kps, frames = kps[idx], np.arange(len(idx)) + frames[0]
    h = np.maximum(kpt_box(kps)[:, 3] - kpt_box(kps)[:, 1], 1.0)[:, None, None]
    kps[:, :, :2] += rng.normal(0, 0.02, kps[:, :, :2].shape) * h
    kps[rng.random(kps.shape[:2]) < 0.1, 2] = 0.0
    return Track(frames, kps.astype(np.float32), kpt_box(kps))


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

def build_model(n_feat: int = FEAT):
    import torch.nn as nn

    class ConcealNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_feat, 64, 5, padding=2), nn.ReLU(), nn.Dropout(0.2),
                nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(64, 1))

        def forward(self, x):                    # x: (B, win, FEAT)
            return self.net(x.transpose(1, 2)).squeeze(1)

    return ConcealNet()


def training_windows(videos: list[Video], stride: int = 3, version: int = 1, augment: int = 0,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Windows ending every `stride` frames (neighbouring windows are near-duplicates); `augment` extra
    randomly augmented copies of every track (labels follow the augmented frame mapping)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for v in videos:
        for t in v.tracks.values():
            for k in range(1 + augment):
                if k == 0:
                    tt, lab = t, v.labels[np.minimum(t.frames, v.n_frames - 1)]
                else:
                    tt = augment_track(t, rng)
                    src = t.frames[np.clip(np.round(np.linspace(0, len(t.frames) - 1, len(tt.frames))).astype(int), 0, len(t.frames) - 1)]
                    lab = v.labels[np.minimum(src, v.n_frames - 1)]
                ok = np.arange(len(tt.frames))[::stride]
                X.append(windows(track_features(tt, version))[ok])
                y.append(lab[ok])
    return np.concatenate(X), np.concatenate(y).astype(np.float32)


def train_model(videos: list[Video], epochs: int = 20, seed: int = 0, device: str = "cpu", batch: int = 512,
                version: int = 1, augment: int = 0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, y = training_windows(videos, version=version, augment=augment, seed=seed)
    nf = X.shape[-1]
    mu, sd = X.reshape(-1, nf).mean(0), X.reshape(-1, nf).std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd)
    Y = torch.tensor(y)
    m = build_model(nf).to(device)
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
    return {"model": m.cpu(), "mu": mu, "sd": sd, "version": version}


def model_track_scorer(bundle):
    import torch

    def score(t: Track) -> np.ndarray:
        w = (windows(track_features(t, bundle.get("version", 1))) - bundle["mu"]) / bundle["sd"]
        with torch.no_grad():
            return torch.sigmoid(bundle["model"](torch.tensor(w, dtype=torch.float32))).numpy()
    return score


def save_bundle(bundle, path: Path, meta: dict):
    import torch
    torch.save({"state": bundle["model"].state_dict(), "mu": bundle["mu"], "sd": bundle["sd"],
                "version": bundle.get("version", 1), "meta": meta}, path)


def load_bundle(path: Path):
    import torch
    d = torch.load(path, weights_only=False)
    m = build_model(len(d["mu"]))
    m.load_state_dict(d["state"])
    m.eval()
    return {"model": m, "mu": d["mu"], "sd": d["sd"], "version": d.get("version", 1), "meta": d["meta"]}


# ---------- triggers (what an operator would see) ----------

THRESH = {"model": (0.5, 0.7, 0.9), "rule": (0.25, 0.5, 0.75)}   # fixed up front, never tuned on eval data
MIN_RUN_S, MERGE_S = 0.5, 10.0                                   # 0.5 s over threshold on one track; merge within 10 s


def trigger_frames(scores: np.ndarray, frames: np.ndarray, th: float, fps: float = 15.0) -> list[int]:
    """Frames where one track's score has stayed >= th for MIN_RUN_S; triggers closer than MERGE_S merge."""
    need = max(int(round(MIN_RUN_S * fps)), 1)
    out, run, last_t = [], 0, -1e9
    for s, f in zip(scores, frames):
        run = run + 1 if s >= th else 0
        if run == need:
            if f / fps - last_t > MERGE_S:
                out.append(int(f))
            last_t = f / fps
    return out


def video_triggers(v: Video, per_track, th: float) -> list[int]:
    return sorted(f for t in v.tracks.values() if len(t.frames)
                  for f in trigger_frames(per_track(t), t.frames, th, v.fps))


def sustained_frames(scores: np.ndarray, frames: np.ndarray, th: float, fps: float = 15.0) -> np.ndarray:
    """Frames at which one track's score has been >= th for at least MIN_RUN_S (no merging)."""
    need = max(int(round(MIN_RUN_S * fps)), 1)
    run = np.zeros(len(scores), int)
    for i, s in enumerate(scores):
        run[i] = run[i - 1] + 1 if s >= th and i else int(s >= th)
    return frames[run >= need]


def event_hit(v: Video, per_track, th: float) -> bool:
    """Shoplifting video: was some person's score sustained over th (>= MIN_RUN_S) on a labelled frame?
    Monotone in th (no merging). Labels are per frame, not per person, so any person counts; read it next
    to the clean-clip trigger rate."""
    lab = set(np.flatnonzero(v.labels).tolist())
    return any(lab.intersection(sustained_frames(per_track(t), t.frames, th, v.fps).tolist())
               for t in v.tracks.values() if len(t.frames))


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


def metrics(y: np.ndarray, s: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Headline numbers on frames with at least one pose (mask); `all_frames` keeps the unmasked AUC."""
    out = {"all_frames_auc_roc": roc_auc(y, s)}
    if mask is not None:
        y, s = y[mask], s[mask]
    return {"auc_roc": roc_auc(y, s), "auc_pr": pr_auc(y, s), "eer": eer(y, s),
            "n_frames": int(len(y)), "n_pos": int(y.sum()), **out}
