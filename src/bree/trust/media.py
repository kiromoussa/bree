"""The one place images and clips leave memory: every stored frame goes through `save_image` / `ClipWriter`, which
pixelate heads and faces first (bree.trust.privacy) and write an audit record per file.

`save_image(path, img, persons, ...)` and `ClipWriter(path, ...).write(img, persons)` take UNPROTECTED frames and
protect them. Frames already protected upstream (the pipeline's evidence ring buffer holds pixelated frames) go in with
`protected=True`; that flag is a promise, and tests/test_privacy.py::test_no_unregistered_writers fails when a new
cv2.imwrite / VideoWriter / imencode appears anywhere in src/ that is not listed in WRITER_SITES below.
`raw=True` is for the few places that must keep unblurred footage (RawRecorder, edge bursts): it needs a reason, is
audited, and the file is subject to the short raw retention rule.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bree.trust.audit import AuditLog, NullAudit
from bree.trust.privacy import Guard, still_guard

_AUDIT: AuditLog | NullAudit = NullAudit()

# Every file in src/ that opens an image or video for writing, with why it is allowed. Anything else fails the test.
WRITER_SITES = {
    "src/bree/trust/media.py": "the gateway itself",
    "src/bree/trust/redact.py": "redacted copies (Guard applied inside)",
    "src/bree/alerts/annotate.py": "blur_heads (the pose layer, Guard builds on it)",
    "src/bree/alerts/writer.py": "alert clips from frames pixelated before buffering (draw_frame blur=True)",
    "src/bree/pipeline.py": "annotated.mp4 from draw_frame(blur=True)",
    "src/bree/dashboard/server.py": "live frame for the dashboard, annotated (blurred) frame only",
    "src/bree/review/store.py": "save_evidence_frame blurs heads",
    "src/bree/review/synthetic.py": "synthetic demo frames, no people",
    "src/bree/edge/capture.py": "node JPEG encode: raw by design, leaves the node only to the hub (see data_flow.md)",
    "src/bree/edge/hub.py": "stores raw node bursts: raw by design, short retention rule node_bursts",
    "src/bree/shadow.py": "RawRecorder (opt-in, 24 h) and annotated clips",
    "src/bree/shelf/diff.py": "shelf evidence crops: need save_image (flagged in docs/ultra/trust.md)",
    "src/bree/plates/pump.py": "plate crop JPEG: plate store retention applies",
    "src/bree/plates/synth.py": "synthetic plates",
    "src/bree/plates/retention.py": "plate crops under the plate store's own retention",
    "src/bree/review/labels.py": "label export from already pixelated evidence",
}


def set_audit(audit: AuditLog | NullAudit) -> None:
    global _AUDIT
    _AUDIT = audit


def get_audit():
    return _AUDIT


def save_image(path: str | Path, img: np.ndarray, persons=None, *, guard: Guard | None = None, protected: bool = False,
               raw: bool = False, reason: str = "", quality: int = 90, actor: str = "bree") -> str:
    """Write one JPEG/PNG (by extension). Pixelated unless protected=True (already done) or raw=True (needs a reason)."""
    if raw and not reason:
        raise ValueError("raw=True keeps faces in the file: say why (reason=...)")
    out = img
    if not (protected or raw):
        out = (guard or still_guard()).protect(img, persons, inplace=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"could not write {path}")
    _AUDIT.record(actor, "write_raw_image" if raw else "write_image", path.name, protected=bool(protected or not raw), reason=reason)
    return str(path)


class ClipWriter:
    """mp4 writer that pixelates each frame (one Guard per clip, so the hold works across frames)."""

    def __init__(self, path: str | Path, fps: float, guard: Guard | None = None, raw: bool = False, reason: str = "",
                 protected: bool = False, actor: str = "bree", fourcc: str = "mp4v"):
        if raw and not reason:
            raise ValueError("raw=True keeps faces in the file: say why (reason=...)")
        self.path, self.fps, self.raw, self.protected, self.actor = Path(path), fps, raw, protected, actor
        self.guard = guard or Guard()
        self.fourcc, self._vw, self.frames = fourcc, None, 0
        self.reason = reason

    def write(self, img: np.ndarray, persons=None) -> None:
        if self._vw is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = img.shape[:2]
            self._vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h))
        self._vw.write(img if (self.raw or self.protected) else self.guard.protect(img, persons, inplace=False))
        self.frames += 1

    def close(self) -> None:
        if self._vw is not None:
            self._vw.release()
            self._vw = None
            _AUDIT.record(self.actor, "write_raw_clip" if self.raw else "write_clip", self.path.name, frames=self.frames,
                          protected=not self.raw, reason=self.reason)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
