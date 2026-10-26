"""Configurable retention with a purge job. Dry-run is the default: nothing is deleted without apply=True / --apply.

A policy is a list of rules, each a glob under one root folder plus a maximum age, e.g.

    root: out/shadow
    rules:
      - {name: raw_segments, glob: "raw/**/*.mp4",    max_age_hours: 24,  evictable: true}
      - {name: alert_clips,  glob: "alerts/*.mp4",    max_age_days: 30}
    max_total_gb: 50          # emergency cap over the evictable rules only, oldest first
    holds: ["alerts/a-0042*"] # globs (relative to root) that are never deleted (e.g. handed to the police)

Safety: only regular files strictly inside root are ever touched (symlinks are skipped, not followed), root must
exist and cannot be / or a home folder, a file named in a hold or next to a `.hold` file is kept, and every
purge writes one audit record (bree.trust.audit) listing counts and bytes, never file contents.
"""
from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_RULES = [   # what the pipeline writes (see docs/ultra/data_flow.md); ages are conservative defaults
    {"name": "raw_segments", "glob": "raw/**/*.mp4", "max_age_hours": 24, "evictable": True},      # UNBLURRED, faces
    {"name": "node_bursts", "glob": "bursts/**/*", "max_age_hours": 48, "evictable": True},         # UNBLURRED
    {"name": "annotated_video", "glob": "**/annotated.mp4", "max_age_days": 7, "evictable": True},
    {"name": "alert_clips", "glob": "alerts/*.mp4", "max_age_days": 30},
    {"name": "alert_records", "glob": "alerts/*.json", "max_age_days": 90},
    {"name": "review_frames", "glob": "**/frames/*.jpg", "max_age_days": 30},
    {"name": "shelf_evidence", "glob": "**/evidence/*.jpg", "max_age_days": 30},
    {"name": "frame_logs", "glob": "**/frames.jsonl", "max_age_days": 30},
]


@dataclass
class Policy:
    root: Path
    rules: list[dict] = field(default_factory=lambda: [dict(r) for r in DEFAULT_RULES])
    max_total_gb: float | None = None
    holds: list[str] = field(default_factory=list)

    @staticmethod
    def load(path: str | Path) -> "Policy":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw = raw.get("retention", raw)
        return Policy(Path(raw["root"]).expanduser(), raw.get("rules") or [dict(r) for r in DEFAULT_RULES],
                      raw.get("max_total_gb"), raw.get("holds") or [])


def _max_age_s(rule: dict) -> float:
    if "max_age_hours" in rule:
        return float(rule["max_age_hours"]) * 3600
    if "max_age_days" in rule:
        return float(rule["max_age_days"]) * 86400
    raise ValueError(f"rule {rule.get('name')!r} needs max_age_hours or max_age_days")


def _check_root(root: Path) -> Path:
    r = root.expanduser().resolve()
    if not r.is_dir():
        raise FileNotFoundError(f"retention root {r} does not exist")
    if r == Path(r.anchor) or r == Path.home().resolve() or len(r.parts) < 3:
        raise ValueError(f"refusing to purge inside {r}: point the policy at a dedicated data folder")
    return r


def _held(rel: str, path: Path, holds: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, h) for h in holds) or (path.parent / ".hold").exists()


def plan(policy: Policy, now: float | None = None) -> list[dict]:
    """Files that would be deleted: [{path, bytes, rule, age_h, reason}]. Touches nothing."""
    now = time.time() if now is None else now
    root = _check_root(policy.root)
    chosen: dict[Path, dict] = {}
    for rule in policy.rules:
        limit = _max_age_s(rule)
        for p in root.glob(rule["glob"]):
            if p.is_symlink() or not p.is_file() or p.name == ".hold" or p in chosen:
                continue
            rel = p.relative_to(root).as_posix()
            if _held(rel, p, policy.holds):
                continue
            st = p.stat()
            if now - st.st_mtime > limit:
                chosen[p] = {"path": rel, "bytes": st.st_size, "rule": rule["name"],
                             "age_h": round((now - st.st_mtime) / 3600, 1), "reason": "older than limit"}
    if policy.max_total_gb is not None:      # emergency cap: oldest evictable files first until under the cap
        evictable = [r["name"] for r in policy.rules if r.get("evictable")]
        files = []
        for rule in policy.rules:
            if rule["name"] in evictable:
                files += [(q.stat().st_mtime, q, rule["name"]) for q in root.glob(rule["glob"])
                          if q.is_file() and not q.is_symlink() and q.name != ".hold"]
        total = sum(q.stat().st_size for _, q, _ in files)
        total -= sum(v["bytes"] for k, v in chosen.items() if v["rule"] in evictable)
        for mt, q, name in sorted(files, key=lambda f: f[0]):
            if total <= policy.max_total_gb * 1e9:
                break
            rel = q.relative_to(root).as_posix()
            if q in chosen or _held(rel, q, policy.holds):
                continue
            sz = q.stat().st_size
            chosen[q] = {"path": rel, "bytes": sz, "rule": name, "age_h": round((now - mt) / 3600, 1),
                         "reason": f"over max_total_gb {policy.max_total_gb}"}
            total -= sz
    return sorted(chosen.values(), key=lambda d: d["path"])


def purge(policy: Policy, apply: bool = False, now: float | None = None, audit=None, actor: str = "purge-job") -> dict:
    """Run the policy. apply=False (default) only reports. Returns a summary; `files` lists what was/would be deleted."""
    items = plan(policy, now)
    root = _check_root(policy.root)
    deleted = failed = 0
    freed = 0
    if apply:
        for it in items:
            try:
                os.unlink(root / it["path"])
                deleted += 1
                freed += it["bytes"]
            except OSError:
                failed += 1
        for d in sorted((p for p in root.rglob("*") if p.is_dir() and not p.is_symlink()), key=lambda p: -len(p.parts)):
            try:
                d.rmdir()       # only empty folders go; rmdir refuses a non-empty one
            except OSError:
                pass
    by_rule: dict[str, dict] = {}
    for it in items:
        r = by_rule.setdefault(it["rule"], {"files": 0, "bytes": 0})
        r["files"] += 1
        r["bytes"] += it["bytes"]
    out = {"dry_run": not apply, "root": str(root), "files": items, "by_rule": by_rule, "deleted": deleted,
           "failed": failed, "bytes_freed": freed, "bytes_planned": sum(i["bytes"] for i in items)}
    if audit is not None:
        audit.record(actor, "purge" if apply else "purge_dry_run", str(root),
                     files=len(items), deleted=deleted, failed=failed, bytes=freed if apply else out["bytes_planned"],
                     by_rule=by_rule)
    return out
