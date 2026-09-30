# DECISIONS

Every judgment call made while building this unattended, with the reason. Newest at the bottom of each section.

## Environment / hardware
- **Build machine**: Linux x86_64, Intel Xeon @ 2.80GHz, 4 vCPUs, 15 GB RAM, **no GPU** (no `nvidia-smi`, `torch.cuda.is_available() == False`, not Apple Silicon). Python 3.11.15.
- **Model sizes**: CPU-only -> **YOLO26 nano** (`yolo26n.pt` detection, `yolo26n-pose.pt` pose), the newest Ultralytics family available (ultralytics 8.4.166). `bree.hw.detect_hardware()` switches to the small (`s`) variants on CUDA or Apple MPS. Recorded by `make hw`.
- **Network**: PyPI and GitHub (incl. Ultralytics release assets) reachable. `kaggle.com`, `data.mendeley.com`, `huggingface.co` are **blocked by the egress proxy (HTTP 403)** in this build environment. Consequence: the Simuletic (Kaggle) and DCSASS (Mendeley) datasets cannot be fetched tonight; instructions for doing it manually are in `data/README.md`.
- **Packaging**: `pyproject.toml` with exact pins; `uv` for installs. Core (events/ledger/sim) depends only on numpy + pyyaml so the theft logic can be tested/installed without torch. Vision deps are the `[vision]` extra.

## Event + ledger design (Stage 3)
- **One pipeline, two products.** Vision emits a small vocabulary of store events (`enter, pick, put_back, conceal, pay, exit`). The ledger never looks at pixels. That makes the theft logic testable with the event-level simulator and keeps it debuggable: every person record keeps a human-readable audit log.
- **Payments are attributed by place and time**, not by identity. A POS receipt names a terminal; the store YAML maps terminal -> zone; the ledger credits whoever was standing in that zone at that time. Queue at the counter: prefer (1) standing there at exactly that time over "within 3 s slack", (2) hasn't already paid this visit, (3) got to the counter first. Cooler card/RFID tap: credited to whoever picked from that cooler closest in time (within 20 s).
- **Two payment modes.** `pos` (we get a POS feed: partial payment is detectable) and `dwell` (no POS feed: a register visit is assumed to pay for everything picked before it — only walkouts and items picked after the register visit can be caught). Pilot should push hard for the POS feed.
- **Items are matched by SKU when known, else by category.** Vision can realistically tell "energy drink can" from "candy bar", not Red Bull from Monster. The catalog maps every SKU to one category.
- **Payments clear the least suspicious item first.** If someone had two sodas, pocketed one and paid for one, the pocketed one is the unpaid one.
- **Evidence scoring, not a black box.** Per unpaid item: `0.5 x pick_conf` + `0.35` if concealed + `0.2` if visibly carried out + `0.15` if never went to the register; x `0.5` if the pick was crowded/ambiguous. Items combine by noisy-OR. `>= 0.7` = **alert** (tell staff now), `>= 0.4` = **review** (manager looks at it later, nobody is confronted), else dropped (kept in the debug log). Consequence by design: **a single uncorroborated pick can never trigger an alert** (max 0.5 + 0.15 = 0.65), because a missed put-back looks exactly like that.
- **Hold decisions instead of guessing.** After exit, wait `exit_grace_s = 5 s` for late POS messages. Hold longer (up to 120 s) while (a) someone who entered within 4 s of this person (likely same party — one person often pays for the group) is still inside, or (b) someone who reached into the same cooler at the same moment (crowded pick) is still inside. Parties are reconciled with pooled baskets and payments; crowded-pick items are handed to the other candidate if that person paid for one extra.
- **Conceal without a seen pick** adds the item to the basket (at the conceal confidence), since the pick was evidently missed.
- **Put-back of an item not in the basket is ignored** (e.g. their own drink brought from the car).
