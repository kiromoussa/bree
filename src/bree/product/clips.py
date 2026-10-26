"""Make evidence clips playable in a browser. OpenCV builds without H.264 write MPEG-4 part 2, which
Firefox and Chrome will not play; such clips are re-encoded once (ffmpeg, local) into a cache folder."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

BROWSER_CODECS = {"h264", "vp8", "vp9", "av1"}


def codec(path: Path) -> str | None:
    if not shutil.which("ffprobe"):
        return None
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=20)
    return r.stdout.strip() or None


def playable(path: Path, cache_dir: Path) -> Path:
    """`path` itself when a browser can play it (or nothing can be done), else the cached H.264 copy."""
    c = codec(path)
    if c is None or c in BROWSER_CODECS or not shutil.which("ffmpeg"):
        return path
    out = cache_dir / (path.stem + ".h264.mp4")
    if out.is_file() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-movflags", "+faststart", str(tmp)], capture_output=True, timeout=120)
    if r.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return path
    tmp.replace(out)
    return out
