"""Fetch a public Google Drive folder file by file (gdown --folder gets rate-limited on big folders).
Usage: python scripts/drive_fetch.py <folder_id> <out_dir>. Skips files that already exist."""
import html, re, subprocess, sys, time, urllib.request
from pathlib import Path

def ls(fid):
    req = urllib.request.Request(f"https://drive.google.com/embeddedfolderview?id={fid}", headers={"User-Agent": "Mozilla/5.0"})
    s = urllib.request.urlopen(req, timeout=60).read().decode()
    for m in re.finditer(r'<div class="flip-entry" id="entry-([^"]+)".*?<div class="flip-entry-title">([^<]+)</div>', s, re.S):
        yield m.group(1), html.unescape(m.group(2)), "/folders/" in s[m.start():m.start() + 600]

def fetch(fid, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for cid, name, is_dir in ls(fid):
        dst = out / name
        if is_dir:
            fetch(cid, dst)
        elif not dst.exists():
            for attempt in range(4):
                url = f"https://drive.usercontent.google.com/download?id={cid}&export=download&confirm=t"
                tmp = dst.with_suffix(dst.suffix + ".part")
                r = subprocess.run(["curl", "-sSfL", "-A", "Mozilla/5.0", "-o", str(tmp), url])
                if r.returncode == 0 and not tmp.read_bytes()[:15].lower().startswith(b"<!doctype html"):
                    tmp.rename(dst)
                    break
                time.sleep(20 * (attempt + 1))
            print("ok" if dst.exists() else "FAILED", dst, flush=True)

if __name__ == "__main__":
    fetch(sys.argv[1], Path(sys.argv[2]))
