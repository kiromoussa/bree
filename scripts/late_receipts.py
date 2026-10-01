"""Late POS receipts: simulator with batched POS exports, with and without alert retraction.
Seed 2 (held out), baseline assumed noise, 200 h, POS feed. Writes results/late_receipts.json. CPU only.
"""
import json
from dataclasses import replace
from pathlib import Path

from bree.eval.metrics import run_event_bench
from bree.events.zones import load_store_config
from bree.sim.events_sim import SimConfig, VisionNoise

store = load_store_config("configs/store_gas_station_small.yaml")
ROWS = [("live POS (default)", 0, {}),
        ("60 s batches, no retraction", 60, {"late_receipt_window_s": 0}),
        ("60 s batches, retraction", 60, {}),
        ("300 s batches, no retraction", 300, {"late_receipt_window_s": 0}),
        ("300 s batches, retraction", 300, {}),
        ("60 s batches, exit_grace_s 65, no retraction", 60, {"exit_grace_s": 65, "late_receipt_window_s": 0}),
        ("60 s batches, exit_grace_s 65 + retraction", 60, {"exit_grace_s": 65})]
out = {"seed": 2, "hours": 200, "noise": "baseline (assumed)", "payment_mode": "pos", "rows": []}
for name, batch, led in ROWS:
    r = run_event_bench(store, SimConfig(hours=200), replace(VisionNoise(), pos_batch_s=batch), 2, ledger_overrides=led)
    row = {"setting": name, "pos_batch_s": batch, "ledger": led,
           **{k: r[k] for k in ("precision", "recall", "false_alerts_per_hour", "reviews_on_honest_per_hour")}}
    out["rows"].append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
Path("results/late_receipts.json").write_text(json.dumps(out, indent=1))
