"""Secrets handling: where credentials may live, how they are read, and how to check nothing leaks.

Rules (docs/ultra/security_imou.md has the reasoning):
  - a credential lives in a file readable only by its owner (0600, owner = the running user, not a symlink), or in an
    environment variable set by systemd's EnvironmentFile / LoadCredential. Never in argv (ps shows it to every
    user), never in a YAML config that is committed, never in a log line.
  - config values may be a reference: "file:/path", "env:NAME" or a literal (literals only for tests and CHANGE_ME).
  - `Secret` wraps a value so print(), repr(), f-strings and logging show "<secret>"; `.reveal()` is the one way out.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path


class Secret:
    __slots__ = ("_v",)

    def __init__(self, value: str):
        self._v = value

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "<secret>"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._v)

    def __format__(self, spec: str) -> str:
        return "<secret>"


def file_problems(path: str | Path) -> list[str]:
    """Why a credential file is not acceptable (empty list = fine): wrong owner, group/other access, symlink, or a
    parent folder other users can write to (they could swap the file)."""
    p = Path(path).expanduser()
    if not p.exists() and not p.is_symlink():
        return [f"{p} does not exist"]
    probs = []
    if p.is_symlink():
        probs.append("is a symlink (use the real file so its permissions are the ones checked)")
    st = p.stat()
    if st.st_uid != os.getuid():
        probs.append(f"owned by uid {st.st_uid}, not the running user")
    if st.st_mode & 0o077:
        probs.append(f"mode {stat.S_IMODE(st.st_mode):04o} lets group/others read or write it (want 0600)")
    pst = p.parent.stat()
    if pst.st_mode & 0o002 and not pst.st_mode & stat.S_ISVTX:
        probs.append(f"parent folder {p.parent} is world-writable")
    return probs


def parse_env_text(text: str) -> dict[str, str]:
    out = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def load_env_file(path: str | Path, strict: bool = True) -> dict[str, Secret]:
    """KEY=VALUE file -> {key: Secret}. strict: refuse a file with bad permissions (raises PermissionError)."""
    probs = file_problems(path)
    if probs and strict:
        raise PermissionError(f"{path}: " + "; ".join(probs))
    return {k: Secret(v) for k, v in parse_env_text(Path(path).expanduser().read_text()).items()}


def resolve_secret(ref: str, strict: bool = True) -> Secret:
    """'file:/path' (first line, permissions checked), 'env:NAME', or the literal itself."""
    if ref.startswith("file:"):
        path = Path(ref[5:]).expanduser()
        probs = file_problems(path)
        if probs and strict:
            raise PermissionError(f"{path}: " + "; ".join(probs))
        return Secret(path.read_text().splitlines()[0].strip() if path.read_text().strip() else "")
    if ref.startswith("env:"):
        return Secret(os.environ.get(ref[4:], ""))
    return Secret(ref)


_URL = re.compile(r"\b(\w+://[^\s\"'<>?#]+)(\?[^\s\"'<>]*)?")


def redact(text: str, secrets=()) -> str:
    """Remove every known secret and the query part of every URL (live addresses carry tokens in the query)."""
    text = _URL.sub(lambda m: m.group(1) + ("?<redacted>" if m.group(2) else ""), str(text))
    for s in secrets:
        v = s.reveal() if isinstance(s, Secret) else s
        if v:
            text = text.replace(v, "<secret>")
    return text


# ---- scanners ---------------------------------------------------------------------------------------------------

_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer literal": re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{24,}"),
    "assigned secret": re.compile(r"""(?ix)\b(app_?secret|secret|passw(?:or)?d|api_?key|access_?token|token)\b\s*[:=]\s*['"]?
                                      (?!CHANGE_ME|\$|<|\{|None|null|""|''|env:|file:|\s)([A-Za-z0-9_\-+/=.]{16,})['"]?"""),
    "url with token": re.compile(r"(?i)https?://[^\s\"']+[?&](?:token|sign|key|secret)=[^\s\"'&]{8,}"),
}


def scan_text(text: str) -> list[tuple[str, int]]:
    """(pattern name, line number) for secret-looking content. Values are never returned."""
    hits = []
    for i, ln in enumerate(text.splitlines(), 1):
        for name, rx in _PATTERNS.items():
            if rx.search(ln):
                hits.append((name, i))
    return hits


def scan_tree(root: str | Path, globs=("*.py", "*.yaml", "*.yml", "*.md", "*.sh", "*.service", "*.json", "*.txt", "*.toml", "*.env*"),
              skip=("weights", ".venv", "data", "models", "results", ".git")) -> list[tuple[str, str, int]]:
    root = Path(root)
    hits = []
    for g in globs:
        for p in root.rglob(g):
            if any(part in skip for part in p.relative_to(root).parts) or not p.is_file() or p.stat().st_size > 2_000_000:
                continue
            try:
                for name, ln in scan_text(p.read_text(errors="ignore")):
                    hits.append((p.relative_to(root).as_posix(), name, ln))
            except OSError:
                pass
    return hits


def known_values_in_tree(root: str | Path, values: list[str], skip=(".venv", ".git", "models", "weights")) -> list[str]:
    """Files under root that contain any of the given secret values verbatim (the real credentials of this machine).
    Returns file names only. Used by the test that proves the repo has none of the Imou credentials in it."""
    values = [v for v in values if v and len(v) >= 6]
    root = Path(root)
    found = []
    for p in root.rglob("*"):
        if not p.is_file() or any(part in skip for part in p.relative_to(root).parts) or p.stat().st_size > 20_000_000:
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if any(v.encode() in data for v in values):
            found.append(p.relative_to(root).as_posix())
    return found


def processes_exposing(values: list[str]) -> list[tuple[int, str]]:
    """(pid, command name) of processes whose command line contains a secret value or a URL with a token query.
    /proc/<pid>/cmdline is world-readable on a default Linux, so anything here is visible to every local user."""
    values = [v for v in values if v and len(v) >= 6]
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().decode(errors="ignore").replace("\0", " ")
        except OSError:
            continue
        if any(v in cmd for v in values) or re.search(r"(?i)://\S+[?&](?:token|sign|key|secret|auth)\w*=", cmd):
            out.append((int(d.name), cmd.split(" ", 1)[0].rsplit("/", 1)[-1]))
    return out


def private_playlist(url: str, dest_dir: str | Path) -> Path:
    """Write a one-line HLS master playlist (mode 0600) that points at `url`, so ffmpeg is started with the PLAYLIST
    path in argv and the tokenised address never appears in `ps`. Run ffmpeg with
    `-protocol_whitelist file,crypto,tcp,tls,https,http -allowed_extensions ALL -i <playlist>`; delete the file after."""
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"live_{os.getpid()}_{os.urandom(4).hex()}.m3u8"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n{url}\n")
    return p


_REF = re.compile(r"\{(env|file):([^}]+)\}")


def expand_refs(text: str) -> str:
    """Fill {env:NAME} and {file:/path} placeholders (e.g. in an rtsp:// URL: rtsp://cam:{file:~/.config/bree/cam.pw}@10.0.0.5)
    so a config file can be committed or shared without the password in it."""
    def sub(m):
        return resolve_secret(f"{m.group(1)}:{m.group(2)}").reveal()
    return _REF.sub(sub, text)
