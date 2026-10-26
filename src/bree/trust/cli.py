"""`python -m bree.trust <command>` (or `bree-trust`): the trust layer from the command line.

  purge      apply the retention policy; DRY RUN unless --apply
  audit      verify the audit log chain, show the head hash or the last records
  doctor     clock, disk, credential files, folder permissions, audit chain; exit 1 on any fail
  scan       independent face scan of stored images / video frames: lists residual detections (the leak test)
  redact     write a head-pixelated copy of a video or image
  sbom       software bill of materials + licence classes (flags copyleft); --json writes CycloneDX
  secrets    scan the repo and running processes for credentials (never prints values)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _audit(path):
    from bree.trust.audit import AuditLog
    return AuditLog(path) if path else None


def cmd_purge(a):
    from bree.trust.retention import Policy, purge
    audit = _audit(a.audit)
    res = purge(Policy.load(a.policy), apply=a.apply, audit=audit)
    print(("DRY RUN: nothing deleted. " if res["dry_run"] else "") +
          f"{len(res['files'])} file(s), {res['bytes_planned'] / 1e6:.1f} MB past their limit under {res['root']}")
    for rule, v in res["by_rule"].items():
        print(f"  {rule:18s} {v['files']:6d} files {v['bytes'] / 1e6:10.1f} MB")
    if a.list:
        for it in res["files"]:
            print(f"  {'delete' if a.apply else 'would delete'} {it['path']}  ({it['rule']}, {it['age_h']} h, {it['reason']})")
    if a.apply:
        print(f"deleted {res['deleted']}, failed {res['failed']}, freed {res['bytes_freed'] / 1e6:.1f} MB")


def cmd_audit(a):
    log = _audit(a.log)
    ok, msg = log.verify(a.expect_head)
    print(("OK: " if ok else "TAMPERED: ") + msg)
    if a.head:
        print(log.head)
    for r in log.read()[-a.tail:] if a.tail else []:
        print(r["time"], r["actor"], r["action"], r["target"], json.dumps(r["detail"]))
    sys.exit(0 if ok else 1)


def cmd_doctor(a):
    from bree.trust.health import doctor, worst
    checks = doctor(a.data, a.credential or None, a.audit, a.policy)
    for c in checks:
        print(f"{c['status'].upper():5s} {c['name']:28s} {c['detail']}")
    sys.exit(1 if worst(checks) == "fail" else 0)


def cmd_scan(a):
    import cv2
    from bree.trust.privacy import scan_faces
    files = [Path(p) for p in a.paths]
    bad = n = 0
    for f in files:
        if f.is_dir():
            files += sorted(p for p in f.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".avi"))
            continue
        frames = []
        if f.suffix.lower() in (".mp4", ".avi"):
            cap = cv2.VideoCapture(str(f))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for k in range(a.frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * total / a.frames))
                ok, fr = cap.read()
                if ok:
                    frames.append(fr)
        else:
            im = cv2.imread(str(f))
            frames = [] if im is None else [im]
        for i, fr in enumerate(frames):
            n += 1
            hits = scan_faces(fr, score=a.score)
            if hits:
                bad += 1
                print(f"LEAK? {f} frame {i}: {len(hits)} detection(s), best score {max(s for _, s, _ in hits):.2f}")
    print(f"scanned {n} image(s)/frame(s): {bad} with a possible face (score >= {a.score})")
    sys.exit(1 if bad else 0)


def cmd_redact(a):
    from bree.trust.redact import make_person_backend, redact_image, redact_video
    import cv2
    be = None if a.no_persons else make_person_backend()
    src = Path(a.src)
    if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
        cv2.imwrite(a.dst, redact_image(cv2.imread(str(src)), be))
        print("wrote", a.dst)
    else:
        print(json.dumps(redact_video(src, a.dst, be, stride=a.stride, max_frames=a.max_frames, person_every=a.person_every), indent=1))


def cmd_sbom(a):
    from bree.trust import sbom
    comps = sbom.collect()
    if a.json:
        Path(a.json).write_text(json.dumps(sbom.cyclonedx(comps), indent=1))
        print("wrote", a.json)
    print(sbom.markdown(comps))
    if a.weights:
        for w in sbom.weights_manifest(a.weights):
            print(f"{w['file']:44s} {w['bytes']:>11,d} {w['sha256']}  {w['license']}")
    bad = [c for c in comps if c["class"] == "strong-copyleft"]
    if a.fail_on_copyleft and bad:
        sys.exit(1)


def cmd_secrets(a):
    from bree.trust import secrets as S
    hits = S.scan_tree(a.root)
    for rel, name, ln in hits:
        print(f"{rel}:{ln}: {name}")
    print(f"{len(hits)} secret-looking line(s) in {a.root}")
    if a.env:
        env = S.load_env_file(a.env, strict=False)
        vals = [v.reveal() for k, v in env.items() if any(w in k.upper() for w in ("SECRET", "PASS", "TOKEN", "KEY"))]
        files = S.known_values_in_tree(a.root, vals)
        procs = S.processes_exposing(vals)
        print(f"files under {a.root} containing a value from {a.env}: {files or 'none'}")
        print(f"processes with such a value or a tokenised URL in argv: {[f'{n}({p})' for p, n in procs] or 'none'}")
        sys.exit(1 if hits or files or procs else 0)
    sys.exit(1 if hits else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bree-trust", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("purge", help="retention purge (dry run unless --apply)")
    p.add_argument("--policy", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--list", action="store_true", help="print every file")
    p.add_argument("--audit", default=None)
    p.set_defaults(func=cmd_purge)
    p = sub.add_parser("audit")
    p.add_argument("--log", required=True)
    p.add_argument("--tail", type=int, default=0)
    p.add_argument("--head", action="store_true")
    p.add_argument("--expect-head", default=None)
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("doctor")
    p.add_argument("--data", default=None)
    p.add_argument("--credential", action="append")
    p.add_argument("--audit", default=None)
    p.add_argument("--policy", default=None)
    p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("scan", help="independent face scan of stored images / clips")
    p.add_argument("paths", nargs="+")
    p.add_argument("--score", type=float, default=0.5)
    p.add_argument("--frames", type=int, default=12, help="frames sampled per video")
    p.set_defaults(func=cmd_scan)
    p = sub.add_parser("redact")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--person-every", type=int, default=1)
    p.add_argument("--no-persons", action="store_true", help="face detector only (no person detector)")
    p.set_defaults(func=cmd_redact)
    p = sub.add_parser("sbom")
    p.add_argument("--json", default=None)
    p.add_argument("--weights", default=None, help="repo root: also list model files")
    p.add_argument("--fail-on-copyleft", action="store_true")
    p.set_defaults(func=cmd_sbom)
    p = sub.add_parser("secrets")
    p.add_argument("--root", default=".")
    p.add_argument("--env", default=None, help="a credential env file: check no value from it is in the tree or in argv")
    p.set_defaults(func=cmd_secrets)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
