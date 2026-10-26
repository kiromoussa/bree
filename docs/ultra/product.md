# Operator product: dashboard, digest, review, notifications

Track `ultra/product`. Code: `src/bree/product/`, tests: `tests/test_product.py`, scripts: `scripts/product_shot.js`
(headless Firefox screenshots), `scripts/product_eval_occupancy.py`. Nothing here talks to the network, nothing
here stores images except what the existing review store already holds (head-pixelated clips).

## What it is

A small stdlib web app for a non-technical store owner, plus printable digests and threshold notifications.
Everything reads one contract, the **signal log**, so perception, shelf, theft and simulator tracks can feed
it without importing this package. Review items come from the existing `bree.review.ReviewStore`.

    python -m bree.product demo   --out DIR            # SIMULATED 3 weeks of data + synthetic review items
    python -m bree.product serve  --data DIR [--replay] [--port 8090] [--token T --host 0.0.0.0]
    python -m bree.product ingest --data DIR --camera ch07 --role door --frames run/frames.jsonl \
                                  --segment 20261025_182851.mp4 --store-yaml store.yaml
    python -m bree.product digest --data DIR --date 2026-10-25 [--weekly] --out digest.html
    python -m bree.product notify --data DIR           # dry run of the rules, prints what would fire

Data folder: `station.yaml`, `signals/signals-YYYYMMDD.jsonl`, `review/` (ReviewStore), `notifications.jsonl`, `cache/clips/`.

Screens (phone first, no CDN, works offline): **Now** (people inside, register line, came in today, items to
review, "needs your attention", cameras working or not, shelf/cooler state), **Day** (hourly traffic, occupancy
curve, register-line episodes, time spent per area, shelf problems, date picker), **Review** (clip with blurred
heads, three big buttons Theft / Not theft / Can't tell, reviewer name remembered), **Reports** (daily and weekly digest).
A banner on every screen and every digest names the weakest data source present (SIMULATED, ESTIMATES, ...).

## The signal contract (v1)

One JSON object per line, `bree.product.signals.Signal`:

    {"v":1,"ts":1792970000.5,"store":"trumanrd","camera":"ch07","kind":"door_in","value":1,
     "zone":null,"source":"real_unlabelled","meta":{}}

`ts` UTC epoch seconds of the store's real clock (apply the DVR offset before writing). Kinds:

| kind | value | notes |
|---|---|---|
| occupancy | people | per camera, or `camera:"store"` for a whole-store count (preferred when a track has one) |
| door_in / door_out | people (usually 1) | only from a camera that sees the door |
| queue | people waiting at the register | |
| dwell | seconds | one finished visit, `zone` required |
| shelf | fullness 0..1 | `meta.state` in ok, low, empty, door_open, misplaced, dirty, unknown |
| health | 0 or 1 | camera heartbeat; silence is also detected from the wall clock |
| pos_sale | USD | when a POS feed exists |

`source` is one of simulated, real_unlabelled, real_model_labelled, real_ground_truth, live. It is never dropped:
tiles, banners and digests show the weakest source in the window. Invalid signals are rejected before anything is
written (whole batch). Readers skip malformed lines (half-written tail). Files are partitioned by store-local date.

Other tracks: write `Signal` lines (or call `SignalLog.append`). Shelf/cooler track: emit `shelf` with a zone name per
section and `meta.state`. Review/theft track: keep using `ReviewStore.add_alert`; the app reads it. Existing pipeline
runs are converted by `bree.product.adapt` (below). The contract is additive: new kinds need a new entry in `KINDS`.

## Metrics rules (bree.product.metrics)

Samples are weighted by the time they cover (capped), never counted; several cameras' occupancy is the max over
cameras per minute (can under-count, never double counts); no data is `None`/"n/a"/a dot, never 0; a camera that
reports every 5 minutes is not "stale" after 2 (staleness scales with its own cadence); hours with under 50 minutes of
data are drawn light. Shrink summary only counts what reviewers confirmed, no dollar estimate.

## Pipeline adapter (bree.product.adapt) and what it does to real tracker output

Real baselines showed the tracker churning ids (53 ids for about 6 people on ch07), so the adapter counts a track
only if it lives at least 2 s and 8 frames, has median confidence at least 0.45, and is not a stationary low-confidence
ghost; it never uses the engine's per-track `enter` events. Door in/out needs a door-role camera and a track that is
born (or ends) within 150 px of a door-kind zone. Queue and staff exclusion need a `queue_zone`/`staff_zone` name
(none configured yet for Truman Rd., so there are no real queue or staff-aware numbers).

### Measurements (all REAL-MODEL-LABELLED, tiny samples, frozen defaults, no tuning)

Labels are the agent-made people counts in `bree-data/real/labels` (not ground truth; 14 to 40 frames per cell,
frames not independent). "raw" = every tracker box in that frame; "filtered" = after the adapter filters. Exact =
share of labelled frames where the count equals the label. `scripts/product_eval_occupancy.py`.

| camera | split | n | raw exact | filtered exact | filtered bias |
|---|---|---|---|---|---|
| ch07 | dev | 40 | 82% | 78% | -0.23 |
| ch07 | heldout | 40 | 62% | 65% | -0.30 |
| ch10 | dev / heldout | 39 / 40 | 100% / 85% | 100% / 85% | 0 / -0.15 |
| ch12 | dev | 15 | 53% | 87% | -0.13 |
| ch14 | dev / heldout | 15 / 14 | 33% / 29% | **0% / 0%** | -1.1 / -1.2 |
| ch15 | dev / heldout | 40 / 40 | 100% / 68% | 100% / 92% | 0 / -0.07 |
| ch29 | dev / heldout | 40 / 40 | 92% / 48% | 85% / 45% | -0.05 / -0.65 |

Reading: the filter is **not** an occupancy accuracy win overall. It removes ghost boxes (ch12 dev, ch15 held-out)
but on ch14, where real people are small and low confidence, it removes every true person (negative result: the
0.45 confidence floor is wrong for that camera; a per-camera floor is needed and must be tuned on dev labels only).
ch29 held-out under-counts people at the frame edge whatever the filter. Its purpose is event counts (door, dwell),
which have no labels at all: door_in/door_out on ch07 are 3/4 (dev) and 5/6 (held-out), against my earlier rough viewing
counts of about 2-3 and about 8 entries (REAL-UNLABELLED, no truth). Do not quote them as accuracy.

### Real-footage replay

`bree-data/product/trumanrd/` (local only): 11 baseline runs (6 cameras, 20 min of one afternoon) ingested as
REAL-UNLABELLED signals. The dashboard shows 8 came in, 3 most inside at once, with a banner saying so. Station time zone is
ASSUMED America/Chicago and the DVR clock offset is unreconciled, so the hour labels are unconfirmed.
Daily/weekly numbers for this store are not meaningful with 20 minutes of footage.

## Notifications (bree.product.notify)

`Rule(kind, op, threshold, for_s, cooldown_s, severity, state, scope)` fires when a condition has held for `for_s` with
a cooldown; a gap longer than 15 min or a recovery resets the timer. Defaults (starting points, not measured): queue of
3 for 60 s, 15 people for 2 min, cooler door open 3 min, shelf empty 10 min, camera silent 2 min, and a flagged basket
(tier alert) from the review store. Overridable per station in `station.yaml` (`rules:`). Sinks: `LogSink`,
`FileSink` (the dashboard bell file), `WebhookSink`, which is a STUB: it only writes the request it would make to an
outbox file and never opens a socket. A failing sink never blocks the others.

## Security and privacy

Binds 127.0.0.1; any other host requires `--token` (bearer header, cookie or `?token=`, constant-time compare);
POST only accepts JSON (no cross-site forms); strict CSP, no external resources. Clips come only from the review store
(already head-pixelated, raw folders refused) and are re-encoded to H.264 on first play if OpenCV wrote MPEG-4
(browsers cannot play that; found while testing: the synthetic clips were unplayable in Firefox). The signal log holds
no images and no identities. Reviewer names are free text typed by the owner.

## Verification

`pytest tests/test_product.py` (18 tests, synthetic): contract validation, atomic bad batch, torn lines, metrics
edge cases, notifier timing/cooldown/gap rules, webhook never sends, adapter filters and role rules, digests
(escaping, empty days, weekly compare), server API, auth, CSRF content type, review decision flow, alert monitor.
Screenshots were taken at 420 px and 1100 px with `scripts/product_shot.js` and viewed (found and fixed:
tiny chart text on phones, 5-minute cameras shown as stale, unplayable clips, replay gaps flagged as outages).

## Known gaps

- No live pipeline hook: ingestion is per finished run (`ingest`). A tail/watch mode for `frames.jsonl` is not built.
- No shelf/cooler signals exist from real footage yet (shelf track output not available when this was written); the
  shelf screens are exercised only by the simulator.
- No queue or staff-aware numbers for the real store (needs zones the operator has not confirmed).
- No POS feed, so no sales tiles, no revenue-per-hour, no conversion rate.
- The webhook sink is a stub by design. No auth beyond a shared token; no per-user accounts.
- Weekly digest is untested on real data (only about 20 minutes exist).
- The review screen only records Theft / Not theft / Can't tell; wrong-item correction exists in the store but not in the UI.
