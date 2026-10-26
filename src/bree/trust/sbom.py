"""Software bill of materials and licence check for the installed environment.

`collect()` lists every installed distribution with its declared licence (License-Expression, License field or Trove
classifier), `classify()` sorts the licence text into permissive / weak copyleft / strong copyleft / unknown, and
`cyclonedx()` writes a CycloneDX 1.5 JSON document. Model weights are listed separately (`weights_manifest`) with a
sha256 and the licence of the project that published them.

A licence string in package metadata is what the author declared, not a legal conclusion. Copyleft flags are a prompt
to ask a lawyer before shipping BREE as a product, not a verdict. Not legal advice.
"""
from __future__ import annotations

import hashlib
import json
import re
from importlib import metadata
from pathlib import Path

STRONG = re.compile(r"(?i)(agpl|affero|\bgpl|gnu general public|sspl|eupl|cc-by-sa)")
WEAK = re.compile(r"(?i)(lgpl|lesser general public|library general public|\bmpl|mozilla public|\bepl|eclipse public|cddl)")
PERMISSIVE = re.compile(r"(?i)(\bmit\b|bsd|apache|\bisc\b|psf|python software foundation|zlib|unlicense|public domain|cc0|"
                        r"hpnd|0bsd|\bbsl|boost|blue oak|\bpil\b|historical permission|\bofl|zpl|dual license)")
PROPRIETARY = re.compile(r"(?i)(proprietary|nvidia)")

# What metadata says is wrong or missing for packages in this project's tree (checked against each project's own repo).
OVERRIDES = {
    "ultralytics": "AGPL-3.0",     # also sold under an Enterprise licence; YOLO weights from Ultralytics follow it
    "lap": "BSD-2-Clause",
    "python-dateutil": "Apache-2.0 OR BSD-3-Clause",   # metadata says "Dual License"
}


def classify(license_text: str) -> str:
    """permissive | weak-copyleft | strong-copyleft | proprietary | unknown. The worst term named wins, except that an
    `OR` expression with a permissive alternative is permissive (the user may pick it)."""
    t = license_text or ""
    if not t.strip():
        return "unknown"
    if PROPRIETARY.search(t):
        return "proprietary"
    if " OR " in t and PERMISSIVE.search(t):
        return "permissive"
    if WEAK.search(t) and not STRONG.search(re.sub(r"(?i)lgpl", "", t)):
        return "weak-copyleft"
    if STRONG.search(re.sub(r"(?i)lgpl", "", t)):
        return "strong-copyleft"
    if WEAK.search(t):
        return "weak-copyleft"
    return "permissive" if PERMISSIVE.search(t) else "unknown"


def _license_of(md) -> str:
    name = md.get("Name", "")
    if name.lower() in OVERRIDES:
        return OVERRIDES[name.lower()]
    expr = md.get("License-Expression")
    if expr:
        return expr
    lic = (md.get("License") or "").strip()
    cls = [c.split("::")[-1].strip() for c in (md.get_all("Classifier") or []) if c.startswith("License ::")]
    if lic and len(lic) < 120 and "\n" not in lic:
        return lic
    return "; ".join(cls) or (lic.splitlines()[0][:80] if lic else "")


def collect() -> list[dict]:
    out = []
    for d in metadata.distributions():
        md = d.metadata
        name = md.get("Name")
        if not name:
            continue
        lic = _license_of(md)
        if name.lower() == "bree-vision":
            continue
        out.append({"name": name, "version": d.version, "license": lic, "class": classify(lic),
                    "home": md.get("Home-page") or ""})
    return sorted(out, key=lambda r: r["name"].lower())


def weights_manifest(root: str | Path) -> list[dict]:
    """Model files under weights/ and models/ with size and sha256 (first 12 hex), and the upstream licence where known."""
    upstream = {"yolo26": "AGPL-3.0 (Ultralytics) unless an Enterprise licence is bought",
                "face_detection_yunet": "MIT (opencv_zoo)", "conceal_": "trained in this repo (BREE)", "sim_sku": "trained in this repo (BREE)"}
    out = []
    for sub in ("weights", "models"):
        for p in sorted((Path(root) / sub).glob("*")):
            if p.suffix in (".pt", ".onnx") and p.is_file():
                lic = next((v for k, v in upstream.items() if p.name.startswith(k)), "unknown")
                out.append({"file": f"{sub}/{p.name}", "bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:12], "license": lic})
    return out


def cyclonedx(components: list[dict]) -> dict:
    return {"bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
            "metadata": {"component": {"type": "application", "name": "bree-vision"}},
            "components": [{"type": "library", "name": c["name"], "version": c["version"],
                            "purl": f"pkg:pypi/{c['name'].lower()}@{c['version']}",
                            "licenses": [{"license": {"name": c["license"] or "NOASSERTION"}}],
                            "properties": [{"name": "bree:licence-class", "value": c["class"]}]} for c in components]}


def copyleft(components: list[dict]) -> list[dict]:
    return [c for c in components if c["class"] in ("strong-copyleft", "weak-copyleft")]


def markdown(components: list[dict]) -> str:
    n = {k: sum(1 for c in components if c["class"] == k) for k in ("permissive", "weak-copyleft", "strong-copyleft", "proprietary", "unknown")}
    lines = [f"{len(components)} packages: " + ", ".join(f"{v} {k}" for k, v in n.items()), ""]
    for cls in ("strong-copyleft", "weak-copyleft", "proprietary", "unknown"):
        rows = [c for c in components if c["class"] == cls]
        if rows:
            lines += [f"### {cls}", "", "| package | version | declared licence |", "|---|---|---|"]
            lines += [f"| {c['name']} | {c['version']} | {c['license'][:70] or '(none declared)'} |" for c in rows] + [""]
    return "\n".join(lines)
