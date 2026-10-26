"""Top-down picture of the plan for looking at it: camera positions and their view direction, the floor each camera
places people on, regions, and optional floor tracks. Written as a PNG to a local path (it holds no faces or footage)."""
from __future__ import annotations

import numpy as np

from bree.fusion.plan import Plan, footprint

COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def render(plan: Plan, path, tracks: dict[str, list] | None = None, extent=None, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8), dpi=110)
    for name, r in plan.regions.items():
        p = np.asarray(r["poly"], float)
        ax.fill(p[:, 0], p[:, 1], alpha=0.12, label=None)
        ax.plot(*np.vstack([p, p[:1]]).T, lw=1, color="0.4")
        ax.text(*p.mean(0), name, fontsize=8, ha="center", color="0.25")
    for i, (cid, pc) in enumerate(plan.cameras.items()):
        col = COLOURS[i % len(COLOURS)]
        fp = footprint(pc.camera, pc.floor_px, pc.max_range_m, step_px=40)
        if len(fp):
            ax.scatter(fp[:, 0], fp[:, 1], s=3, color=col, alpha=0.25)
        C = pc.camera.C
        ax.scatter([C[0]], [C[2]], marker="^", s=80, color=col, zorder=5)
        fwd = pc.camera.R.T @ np.array([0, 0, 1.0])
        ax.arrow(C[0], C[2], fwd[0] * 1.2, fwd[2] * 1.2, color=col, width=0.05, zorder=5)
        ax.text(C[0] + 0.15, C[2] + 0.15, f"ch{cid}", color=col, fontsize=10, weight="bold")
    for i, (name, pts) in enumerate((tracks or {}).items()):
        p = np.asarray(pts, float)
        if len(p):
            ax.plot(p[:, 0], p[:, 1], lw=1.5, color=COLOURS[i % len(COLOURS)])
            ax.scatter(p[0:1, 0], p[0:1, 1], s=20, color=COLOURS[i % len(COLOURS)])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if extent:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title or plan.station)
    fig.savefig(path)
    plt.close(fig)
