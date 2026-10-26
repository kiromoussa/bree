"""Fill the perception cache for real footage: python scripts/theft_cache.py [--stride 2] [--channels 7,29] ...

Reads ~/Projects/bree-data/real/<station>/chNN/*.mp4 (finished segments only, >= 1 MB), writes
~/Projects/bree-data/theft/cache/<station>/chNN/<segment>/{frames.jsonl.gz,health.npz,info.json}.
Footage and cache stay outside git. Run under nice -n 10 with OMP_NUM_THREADS=4.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = Path.home() / "Projects" / "bree-data"


def segments(station: str, channels: list[str] | None):
    base = DATA / "real" / station
    for ch in sorted(base.glob("ch*")):
        if channels and ch.name not in channels:
            continue
        for mp4 in sorted(ch.glob("*.mp4")):
            if mp4.stat().st_size >= 1_000_000:       # smaller = metering stub, not footage
                yield ch.name, mp4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="trumanrd")
    ap.add_argument("--channels", default=None, help="comma list, e.g. 7,29 (default all)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--health-only", action="store_true")
    ap.add_argument("--out", default=str(DATA / "theft" / "cache"))
    a = ap.parse_args()
    from bree.cli import make_backend
    from bree.events.zones import load_store_config
    from bree.theft.cache import cache_video, is_cached
    chans = [f"ch{int(c):02d}" for c in a.channels.split(",")] if a.channels else None
    store = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")   # only for the COCO -> category map
    backend = None if a.health_only else make_backend("yolo", store)
    for ch, mp4 in segments(a.station, chans):
        out = Path(a.out) / a.station / ch / mp4.stem
        if is_cached(out):
            continue
        t = time.time()
        info = cache_video(mp4, out, backend, stride=a.stride, health_only=a.health_only,
                           log=lambda s: print(s, flush=True))
        print(f"{ch}/{mp4.stem}: {info.frames} frames in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
