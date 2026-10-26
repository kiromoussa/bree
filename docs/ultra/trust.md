# Trust layer: privacy, security, deployment (track `trust`)

Goal: make a station operator able to say yes. Pixelation on everything stored, retention that deletes, an audit trail,
no credentials in argv/configs/logs, a licence check, and packaging that restarts, rolls back and refuses to run on a
broken box. Branch `ultra/trust`. Code in `src/bree/trust/`, tests `tests/test_trust.py` (fast, no footage) and
`tests/test_privacy_real.py` (real footage in the data dir, skipped without it).

Label key for every number: SIMULATED, REAL-UNLABELLED (ran on real footage, no truth), REAL-MODEL-LABELLED (I looked at
frames and judged: useful, not ground truth), REAL-GROUND-TRUTH (none exists).

## What was built

| Area | Module / file | Notes |
|---|---|---|
| Head and face pixelation | `trust/privacy.py` (`Guard`, `still_guard`, `head_regions`, `scan_faces`) | three layers: pose/person-box heads, YuNet face detector over the whole frame, a hold of a few frames. 5x5 mosaic. `overhead` profile for ceiling cameras (`camera.privacy_profile` in the store YAML). Weights `weights/face_detection_yunet_2023mar.onnx` (227 KB, MIT). |
| One gateway for stored media | `trust/media.py` (`save_image`, `ClipWriter`, `WRITER_SITES`) | pixelates by default; `raw=True` needs a reason and is audited. `test_no_unregistered_writers` fails when any new `cv2.imwrite`/`VideoWriter`/`imencode` appears in `src/` outside the registry. |
| Redaction tool | `trust/redact.py`, `python -m bree.trust redact in.mp4 out.mp4` | detector + pose + Guard; for sharing or labelling clips. |
| Leak scanner | `python -m bree.trust scan <files/dirs>` | independent multi-scale face scan of stored images and sampled video frames; exit 1 on any detection. |
| Retention + purge | `trust/retention.py`, `configs/retention_example.yaml`, `bree-purge.timer` | dry run by default, `--apply` to delete; per-rule ages, `max_total_gb` cap over evictable rules, holds (glob or `.hold` file), refuses `/`, `$HOME` and shallow roots, never follows symlinks, audits counts only. |
| Audit log | `trust/audit.py` | hash-chained JSONL (mode 0600); `verify()` detects edit, delete, reorder; truncation only with a kept head hash. Wired into the dashboard/review servers (clip views, review decisions) and `run_shadow` (start, raw recording flag), purge, and the media gateway. Secret-looking keys and URL queries are scrubbed before writing. |
| Secrets | `trust/secrets.py` | `Secret` (never prints), 0600/owner/symlink/parent checks, `file:`/`env:` references, `{env:}`/`{file:}` inside RTSP URLs, tree and process scanners, `private_playlist` to keep tokenised URLs out of argv. Applied: hub `--token-file` / `BREE_HUB_TOKEN`, node `token: file:/...`, shadow `rtsp_url` placeholders and `repr=False`. |
| SBOM and licences | `trust/sbom.py`, `docs/ultra/sbom.cdx.json`, `docs/ultra/sbom.md` | CycloneDX 1.5 plus classes; model weights with sha256 and upstream licence. |
| Health | `trust/health.py` | clock (timedatectl/chrony), DVR clock vs box (timezone offsets recognised), disk, credential and folder modes, audit chain, retention backlog, `DiskGuard`. |
| Watchdog | `trust/supervise.py` | heartbeat file + `sd_notify`; see `deployment.md`. |
| Deployment | `deploy/systemd/`, `deploy/update.sh`, `deploy/container/Containerfile` | see `deployment.md`. |
| Docs | `data_flow.md`, `privacy_signage.md`, `security_imou.md`, `deployment.md`, `sbom.md` | privacy note flagged not legal advice. |

Edits to shared files (small): `alerts/annotate.py` (`guard` argument), `pipeline.py` (per-camera guard, heartbeat),
`events/zones.py` (`privacy_profile`), `shadow.py` (audit, URL placeholders), `edge/hub.py` and `edge/node.py` (secret
sources), `dashboard/server.py` and `review/server.py` (audit hooks; the dashboard now also rejects non-loopback `Host`
headers: it did not before, so DNS rebinding could have read frames and clips from the operator's browser session).

## Measured

### Does the pixelation hide heads on the real store footage?

Footage: the 20-minute-per-channel recordings from Truman Rd. (about 18:28 to 18:56 EDT, one afternoon, channels 7, 10, 12, 14,
15, 29). Held-out discipline: nothing was tuned on these frames; the guard has no learned parameters apart from the pretrained
YuNet weights and the thresholds written before the first run.

**A. Detector counts, REAL-UNLABELLED** (`scripts/trust/leak_eval.py`; 144 random frames, 24 per channel, 6 channels; 127 person
detections; face detector at score 0.5 on 1280 and 2560 px wide):

| | detections |
|---|---|
| raw frames | 954 in 143 of 144 frames |
| after the old routine (`blur_heads`, pose heads only) | 857 |
| after `bree.trust.privacy` (still guard) | 187 |

These are **not** face counts. The store is full of faces that are not people (balaclava displays, mannequin heads, packaged
goods, a cutout), so the raw detector sees about 6.6 "faces" per frame, including in frames with nobody in them (a
channel 15 frame with zero persons has 9). The drop from 857 to 187 mostly means the guard now covers those decoys as well, which
is the cautious direction. The 187 that remain at 0.5 are, in the frames I looked at, shelf clutter at 0.54 (e.g. bottle
caps) rather than faces, but I did not inspect all 187. Pose-model audit on the output (3+ confident head keypoints on a
person): 2 of 127 people (REAL-UNLABELLED).

**B. By eye, REAL-MODEL-LABELLED** (`scripts/trust/leak_crops.py`; 70 random frames with people, 110 person crops; I viewed two
contact sheets, 36 crops, old routine and new routine side by side). Of the crops where a head was in view (about 28):

* old routine: about 4 show part of a head (hair, profile, face edge) outside the mosaic;
* new guard: none shows a recognisable face. A few mosaics are only part of the head (hair above or beside them).

Small sample, one viewer, tiles of 220 px, judgements are subjective. Crops of legs, hands and feet (persons with no head in the
frame) were excluded. I did not view the other five sheets. This supports "better than the old routine", not a recall number.

**C. Regression test**, `tests/test_privacy_real.py`, passes on 12 frames from channels 7 and 12 (no face detection in any
person's head area after guarding). Same detector family as the guard: it protects against regressions, it does not measure recall.

### Cost of over-protection (negative result)

The face detector also fires on non-faces, and the guard pixelates what it finds. Seen: a 120x130 px block over a banana basket
(frame ch14), a large block over a cooler customer's torso on ch12 when the person box was cut by the frame, mosaics on mannequin
heads. For evidence clips this hides some context. A static-decoy mask learned per camera from hours of empty-store footage
would remove the clutter blocks; not done (needs more footage than the 20 minutes available).

### Cost

See the resource table in `deployment.md`. Face detector at 1280 px: 25 ms idle (REAL-UNLABELLED), about 8x that while the
machine ran at load average 65.

### Secrets and argv, this machine

`bree-trust secrets --root <repo> --env ~/.config/bree/imou.env`: 0 secret-looking lines in the tree; none of the Imou credential
values appear in any file of the repo; no process currently exposes a credential value in argv (the recorder had already
stopped). The structural exposure (tokenised URL passed to ffmpeg on the command line) is in the recorder's source and is
documented in `security_imou.md`, with a verified fix.

### Container, REAL-UNLABELLED build facts

`docker build -f deploy/container/Containerfile` succeeded on this machine (x86-64, CPU torch 2.14.0): image 3.46 GB. Verified
inside it with `--read-only --cap-drop ALL`: `hw` runs (cpu, yolo26n selected), `bree.trust sbom` sees 57 packages with no
NVIDIA wheels, `doctor` runs (clock is only a warning inside a container, which cannot see the host's NTP state; check the clock
on the host). A full pipeline run inside the container was not done (weights are mounted, not baked in).

### Licences

85 installed packages: 63 permissive, 2 weak copyleft (certifi, tqdm: MPL), 3 strong copyleft, 14 NVIDIA proprietary
(CUDA runtime wheels), 1 undeclared. The strong copyleft ones are **ultralytics, ultralytics-thop, ultralytics-platform
(AGPL-3.0)**. BREE's detector, pose model and tracker run on Ultralytics, and the YOLO26 weights come from the same project.
AGPL reaches a network service built on it and ships with obligations on source disclosure; Ultralytics sells an Enterprise
licence. This needs a decision (and a lawyer) before BREE is a product: buy the licence, or move to a permissively
licensed detector, which the ONNX path makes feasible. System ffmpeg is LGPL/GPL depending on the build, and the OpenCV wheels
bundle LGPL FFmpeg: dynamic linking, fine for internal use, check before redistributing. Not legal advice.

## Known gaps (honest list)

1. **Shelf evidence crops** (`shelf/diff.py::_evidence`) call plain `cv2.imwrite` on frames that can contain a customer's face.
   Registered in `WRITER_SITES` with a note; the fix is one line (`media.save_image(path, crop, persons=None)`), left to the shelf
   track's owner/integrator to avoid a merge conflict.
2. **Edge hub and `RawRecorder`** store unpixelated frames by design (re-running models later). Protected only by short
   retention (24 h), folder modes, the audit entry, and the purge job. The Imou recorder's raw segments are the same.
3. **People the person detector misses and the face detector misses** are not pixelated (very small, side-on, or behind glass).
   The face detector needs about 12 px faces; far figures on ch07/ch29 are smaller than that.
4. **Imou recorder not patched**: it runs from a data-dir copy of `scripts/imou_record.py`; the fixes (private playlist,
   https only, `-tls_verify 1`) are documented, not applied, because the recorder was live when written and then stopped
   by the flow limit.
5. `DiskGuard` is not yet called by the pipeline writers; audit hooks cover the dashboard/review servers and the gateway but
   not every writer; the audit log has no rotation.
6. Node-to-hub traffic is plain HTTP with a Bearer token (existing design); use a trusted VLAN or a TLS proxy.
7. Not tested: real systemd watchdog behaviour, Jetson/ARM, GPU container, a station other than Truman Rd.
8. One read-only Imou metadata call (`accessToken` + `getLiveStreamInfo`, no stream pulled) was made at about 20:05 local for
   the security review, after the integrator's note said not to call live APIs again. It returned status information only; I
   do not know whether the platform meters that call. Nothing else touched the Imou API.

## Next steps (if continuing)

Wire `DiskGuard` and `media.save_image` into the remaining writers; static-decoy mask per camera; label 200+ head crops with a
second viewer for a real recall figure; LAN capture path so no video leaves the store; decide the Ultralytics licence question.
