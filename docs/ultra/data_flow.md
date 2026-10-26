# Data flow: what BREE sees, keeps, and sends

Status of every statement below: read from the code in this repo or measured on this machine (labels in `trust.md`).
Where a sentence describes a design intent that the code does not yet enforce, it says so.

## 1. Where video goes today (Truman Rd. pilot)

```
 cameras --> store DVR (Dahua) --> Imou cloud relay --> HLS (https) --> this machine (ffmpeg -c copy) --> mp4 segments
                                   (Imou Open Platform)                  imou_record.py                  ~/Projects/bree-data/real/
```

* The DVR is reached through the **Imou cloud**, not over the store LAN. Video leaves the store, goes through Imou's
  servers, and comes to us. "On-device" is true for BREE's own processing, not for this capture path. The operator must
  be told this plainly. (`security_imou.md` has the detail.)
* Segments on disk are **raw and unblurred** (they show faces). They live only under `~/Projects/bree-data/real/<station>/chNN/`
  (mode 0700, outside every git repo). Nothing in git, nothing uploaded, no frame described to any outside service.
* For a station running its own LAN capture (RTSP from the DVR/NVR) the cloud hop disappears; the same box then needs
  nothing outbound except optional alert delivery.

## 2. Inside the box (the pipeline)

```
 frames --> person detect + pose --> tracker --> event engine --> ledger <-- POS receipts
    |                                                |               |
    |                                                |               +--> would_be_alerts.jsonl (text: ids, items, tier)
    |                                                +--> events.jsonl, frames.jsonl (boxes, keypoints, no pixels)
    +--> draw_frame(blur=True) --> evidence ring buffer (RAM, 2 s, pixelated, 640 px) --> alerts/<id>.mp4 (only if flagged)
    +--> annotated.mp4 (debug; pixelated; off in shadow mode)
```

| Artifact | Contains people's faces? | Where | Default retention (`configs/retention_example.yaml`) |
|---|---|---|---|
| Raw DVR segments (recorder) | yes | `real/<station>/chNN/*.mp4` | operator-set; recorder's size cap (80 GB). Not covered by the pipeline's purge: give it its own policy |
| `raw/<camera>/*.mp4` (`record_raw`, off by default) | yes | shadow output dir | 24 h |
| Edge node bursts at the hub | yes (full-res JPEGs) | hub `--out` | 24 h (hub's own `--retain-hours`), 48 h in the purge policy as a backstop |
| Alert evidence clips `alerts/<id>.mp4` | no (pixelated, see `trust.md` for measured limits) | shadow output dir | 30 days |
| Annotated debug video | no (pixelated) | run output dir | 7 days |
| Review label frames (`frames/*.jpg`) | no (pixelated) | review store | 30 days |
| Shelf evidence crops (`evidence/*.jpg`) | **possibly**: written with plain `cv2.imwrite`; see gap list | shelf evidence dir | 30 days |
| `frames.jsonl`, `events.jsonl` | no pixels; boxes, keypoints, track ids | run output dir | 30 days |
| Ledger / alert JSON | no pixels; item names, times, tier, reasons | run output dir | 90 days |
| Plate crops and text (fuel drive-off) | licence plates | plate store (sqlite + crops) | its own: 72 h, 30 days if confirmed, access log 365 days |
| Audit log | no; actor, action, target name, counters | `audit.jsonl` | never purged by the job; rotate by archiving |

Not stored at all: face crops, embeddings of faces, identities. Person re-ID features (body appearance) live in memory only and
are dropped with the track. There is no cross-visit identity.

## 3. What leaves the box

| Flow | Direction | Protocol | Auth | Notes |
|---|---|---|---|---|
| Imou API (token, bind, live address) | out | HTTPS, TLS verified (urllib default) | appId + appSecret signature | metadata only; see `security_imou.md` |
| Imou HLS media | in | HTTPS (the recorder prefers https; http:8888 is also offered) | the URL itself is the credential | video passes through Imou servers |
| Edge node -> hub | LAN | **plain HTTP**, Bearer token | shared token | token and JPEGs are readable on the LAN. Use a trusted VLAN or a TLS-terminating proxy; see `deployment.md` |
| Dashboard / review page | local | HTTP on 127.0.0.1 only | none (loopback + Host header check on review) | do not publish with a port forward |
| POS receipts | in | file drop / stdin / local HTTP | none | local only |
| Alert delivery (SMS/email/Slack) | out | not implemented in the pipeline | n/a | when added, send text + link only, never a clip |

BREE itself makes **no** outbound connection for analytics, model download at run time (except the one-time DINOv2 fetch
if re-ID is enabled; mirror it locally for an offline box), telemetry, or updates.

## 4. Who can read what

* Anyone with a shell on the box can read footage unless folders are 0700 (`bree-trust doctor` fails on world-readable folders).
* Reads through the dashboard/review page are logged in the audit log when `bree.trust.media.set_audit` is wired in
  (done in `trust.md`'s integration notes; the servers call `audit.record` on clip/frame serving).
* The audit log is tamper-evident (hash chain), not tamper-proof: root on the box can rewrite the whole chain. Copy the head
  hash (`bree-trust audit --head`) into the daily digest to anchor it off-box.
