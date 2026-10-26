"""Load the agent-made people-count labels for real footage into one shape.

The five label files under <data>/real/labels were written by different agents with different layouts. They are
REAL-MODEL-LABELLED (an agent looked at downsampled frames): useful for before/after comparisons, never ground truth.
Positions are box centres normalised to the frame (+-0.03 to +-0.1). Splits: "dev" may be tuned on, "heldout" may
only be scored, and only once per frozen configuration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path.home() / "Projects" / "bree-data"
LABEL_KIND = "REAL-MODEL-LABELLED"


@dataclass
class LabelledFrame:
    channel: int
    segment: str                  # file stem, e.g. 20261025_182851
    t_s: float
    split: str                    # dev | heldout
    centres: list[tuple[float, float]]
    uncertain: bool = False
    note: str = ""
    outside: list[tuple[float, float]] = field(default_factory=list)   # seen through glass; not counted

    @property
    def count(self) -> int:
        return len(self.centres)

    @property
    def key(self) -> str:
        return f"ch{self.channel:02d}/{self.segment}@{self.t_s:g}"


def _stem(s: str) -> str:
    return Path(s).stem


def _ch07(d: dict) -> list[LabelledFrame]:
    out = []
    for split, name in (("dev", "dev"), ("heldout", "heldout")):
        seg = _stem(d[name]["segment"])
        for f in d[name]["frames"]:
            ins = [p for p in f["people"] if p.get("role") != "outside_or_at_door"]
            out.append(LabelledFrame(7, seg, f["t_s"], split, [(p["x"], p["y"]) for p in ins],
                                     any(p.get("conf") == "low" for p in ins), f.get("note", ""),
                                     [(p["x"], p["y"]) for p in f["people"] if p.get("role") == "outside_or_at_door"]))
    return out


def _ch10(d: dict) -> list[LabelledFrame]:
    out = []
    for name, s in d["sets"].items():
        split = "heldout" if "held" in name else "dev"
        for f in s["frames"]:
            out.append(LabelledFrame(10, _stem(s["segment"]), f["t_s"], split, [(p["cx"], p["cy"]) for p in f["people"]]))
    return out


def _ch12(d: dict) -> list[LabelledFrame]:
    return [LabelledFrame(12, f["segment"], f["t_s"], f["split"], [tuple(c) for c in f["centres_xy"]],
                          f.get("uncertain", False), f.get("note", "")) for f in d["frames"]]


def _ch14(d: dict) -> list[LabelledFrame]:
    return [LabelledFrame(14, _stem(f["segment"]), f["t_s"], f["split"], [(p["cx"], p["cy"]) for p in f["people"]],
                          note=f.get("note", "")) for f in d["frames"]]


def _ch15(d: dict) -> list[LabelledFrame]:
    held = _stem(d["held_out"]["segment"])
    return [LabelledFrame(15, f["segment"], f["t_s"], "heldout" if f["segment"] == held else "dev",
                          [tuple(c) for c in f["person_centres_xy_norm"]], note=f.get("note", ""),
                          outside=[tuple(c) for c in f.get("people_outside_through_window", [])]) for f in d["frames"]]


def _ch29(d: dict) -> list[LabelledFrame]:
    return [LabelledFrame(29, f["segment"], f["t_s"], f["split"], [tuple(c) for c in f["centres_xy"]],
                          note=f.get("note", "")) for f in d["frames"]]


_LOADERS = {7: _ch07, 10: _ch10, 12: _ch12, 14: _ch14, 15: _ch15, 29: _ch29}


def load_labels(channel: int | None = None, data_dir: Path = DATA_DIR, station: str = "trumanrd") -> list[LabelledFrame]:
    """All labelled frames (or one channel's), every layout normalised."""
    out: list[LabelledFrame] = []
    for ch, fn in _LOADERS.items():
        if channel is not None and ch != channel:
            continue
        p = data_dir / "real" / "labels" / f"{station}_ch{ch:02d}_people.json"
        if p.exists():
            out += fn(json.loads(p.read_text()))
    return out


def segment_path(lf: LabelledFrame, data_dir: Path = DATA_DIR, station: str = "trumanrd") -> Path:
    return data_dir / "real" / station / f"ch{lf.channel:02d}" / f"{lf.segment}.mp4"
