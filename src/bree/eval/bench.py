"""`bree bench`: every number we can honestly measure tonight, in one place.

1. Event-level simulator (ledger logic, independent of vision), on a HELD-OUT seed
   (all development/tuning used seed 1; the benchmark reports seed 2):
   - noise levels: perfect vision / baseline assumed error rates / 2x error rates
   - payment modes: POS feed vs. register-dwell only
   - alert-threshold sweep and a per-error-source ablation at baseline noise
2. Toy video clips (rendered, labelled TOY DATA): full pipeline end to end.
3. Real footage FPS: YOLO26n detector + crop pose + ByteTrack on OpenCV's vtest.avi
   (real pedestrians, no theft labels) on this machine.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bree.eval.metrics import noise_ablation, run_event_bench
from bree.events.zones import load_store_config
from bree.sim.events_sim import SimConfig, VisionNoise

TEST_SEED = 2
SPREAD_SEEDS = [2, 3, 4, 5, 6]
NOISE_LEVELS = {"perfect": 0.0, "baseline": 1.0, "pessimistic_2x": 2.0}


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v}"


def event_level(store, hours: float, seed: int = TEST_SEED) -> dict:
    out = {"seed": seed, "hours_per_run": hours, "runs": []}
    for mode in ("pos", "dwell"):
        for name, k in NOISE_LEVELS.items():
            r = run_event_bench(store, SimConfig(hours=hours, payment_mode=mode), VisionNoise().scaled(k), seed,
                                sweep=(mode == "pos" and name == "baseline"))
            r.update({"payment_mode": mode, "noise": name})
            out["runs"].append(r)
    out["ablation_baseline_pos"] = noise_ablation(store, SimConfig(hours=hours), seed)
    # Seed-to-seed spread of the headline rows (seeds 2..6; seed 1 was the dev seed).
    spread = {}
    for name in ("perfect", "baseline", "pessimistic_2x"):
        rs = [run_event_bench(store, SimConfig(hours=hours), VisionNoise().scaled(NOISE_LEVELS[name]), sd)
              for sd in SPREAD_SEEDS]
        spread[name] = {k: {"mean": round(float(np.mean([r[k] for r in rs])), 3),
                            "min": round(float(np.min([r[k] for r in rs])), 3),
                            "max": round(float(np.max([r[k] for r in rs])), 3)}
                        for k in ("precision", "recall", "recall_alert_or_review", "false_alerts_per_hour")}
    out["seed_spread_pos"] = {"seeds": SPREAD_SEEDS, "by_noise": spread}
    out["noise_model_baseline"] = asdict(VisionNoise())
    out["sim_config"] = asdict(SimConfig(hours=hours))
    return out


def toy_video(root: Path, out_dir: Path) -> dict:
    from bree.commands import ensure_toy
    from bree.eval.toy_eval import run_toy_suite, score_rows
    ensure_toy()
    rows, summaries = run_toy_suite(root / "data" / "toy", out_dir, root / "configs" / "store_gas_station_small.yaml")
    frames = sum(s.frames for s in summaries)
    wall = sum(s.wall_s for s in summaries)
    proc = [x for s in summaries for x in s.processing_latency_ms]
    return {"data": "TOY DATA: 8 rendered 2D clips (bree render-toy)", "rows": rows, "score": score_rows(rows),
            "frames": frames, "pipeline_fps": round(frames / wall, 2),
            "stage_ms_mean": {k: round(sum(s.stage_ms[k] * s.frames for s in summaries) / frames, 2)
                              for k in summaries[0].stage_ms},
            "alert_processing_latency_ms": proc}


def real_fps(root: Path, out_dir: Path, max_frames: int | None = None) -> dict:
    from bree.cli import make_backend
    from bree.hw import detect_hardware
    from bree.pipeline import run_pipeline
    video = root / "data" / "real" / "vtest.avi"
    if not video.exists():
        return {"skipped": f"{video} missing (run make data)"}
    store = load_store_config(root / "configs" / "store_gas_station_small.yaml")
    backend = make_backend("yolo", store)
    s = run_pipeline(str(video), store, backend, out_dir, save_video=True, max_frames=max_frames, verbose=False)
    return {"data": "OpenCV samples/data/vtest.avi (real pedestrians, 768x576 @ 10 fps, no theft labels)",
            "hardware": detect_hardware().to_dict(), "frames": s.frames, "pipeline_fps": s.pipeline_fps,
            "stage_ms_mean": s.stage_ms, "persons_tracked": s.persons_tracked,
            "note": "zones are the example store's, so events on this street scene are meaningless; FPS only"}


def print_tables(res: dict) -> str:
    lines = []
    ev = res.get("event_level")
    if ev:
        lines.append(f"\n## Event-level simulator (held-out seed {ev['seed']}, {ev['hours_per_run']:.0f} simulated store hours per row)")
        lines.append("| payment feed | vision noise | thieves | precision | recall (alert) | recall (alert+review) | false alerts / hour | honest reviews / hour | basket exact | latency p50 / p95 s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in ev["runs"]:
            lines.append(f"| {r['payment_mode']} | {r['noise']} | {r['thieves']} | {_fmt(r['precision'], True)} | {_fmt(r['recall'], True)} | "
                         f"{_fmt(r['recall_alert_or_review'], True)} | {r['false_alerts_per_hour']:.2f} | {r['reviews_on_honest_per_hour']:.2f} | "
                         f"{_fmt(r['basket_exact_match'], True)} | {r['decision_latency_s_p50']} / {r['decision_latency_s_p95']} |")
        base = next(r for r in ev["runs"] if r["payment_mode"] == "pos" and r["noise"] == "baseline")
        lines.append("\nAlert-threshold sweep (POS feed, baseline noise; uncorroborated decisions stay capped at review):")
        lines.append("| threshold | precision | recall | false alerts / hour |")
        lines.append("|---|---|---|---|")
        for d in base["threshold_sweep"]:
            lines.append(f"| {d['threshold']} | {_fmt(d['precision'], True)} | {_fmt(d['recall'], True)} | {d['false_alerts_per_hour']:.2f} |")
        pe = base["alert_precision_by_evidence"]
        lines.append(f"\nAlert precision by evidence (POS feed, baseline noise): concealment seen "
                     f"{pe['conceal_seen']['correct']}/{pe['conceal_seen']['alerts']} = {_fmt(pe['conceal_seen']['precision'], True)}; "
                     f"no concealment {pe['no_conceal']['correct']}/{pe['no_conceal']['alerts']} = {_fmt(pe['no_conceal']['precision'], True)}.")
        lines.append("\nRecall by theft type (POS feed, baseline noise):")
        lines.append("| theft type | thieves | alerted | alert or review |")
        lines.append("|---|---|---|---|")
        for k, v in base["recall_by_theft_type"].items():
            lines.append(f"| {k} | {v['thieves']} | {v['alerted']} | {v['alert_or_review']} |")
        sp = ev.get("seed_spread_pos")
        if sp:
            lines.append(f"\nSeed-to-seed spread (POS feed, seeds {sp['seeds']}, mean [min-max]):")
            lines.append("| vision noise | precision | recall (alert) | recall (alert+review) | false alerts / hour |")
            lines.append("|---|---|---|---|---|")
            for name, m in sp["by_noise"].items():
                cell = lambda k, pct=True: (f"{100*m[k]['mean']:.1f}% [{100*m[k]['min']:.1f}-{100*m[k]['max']:.1f}]" if pct
                                            else f"{m[k]['mean']:.2f} [{m[k]['min']:.2f}-{m[k]['max']:.2f}]")
                lines.append(f"| {name} | {cell('precision')} | {cell('recall')} | {cell('recall_alert_or_review')} | {cell('false_alerts_per_hour', False)} |")
        lines.append(f"\nWhich vision errors cause false alerts? (perfect vision + one error source at its baseline rate; same seed and {ev['hours_per_run']:.0f} hours):")
        lines.append("| error source | precision | recall | false alerts / hour |")
        lines.append("|---|---|---|---|")
        for a in ev["ablation_baseline_pos"]:
            lines.append(f"| {a['error_source']} | {_fmt(a['precision'], True)} | {_fmt(a['recall'], True)} | {a['false_alerts_per_hour']:.2f} |")
    tv = res.get("toy_video")
    if tv:
        sc = tv["score"]
        lines.append(f"\n## Toy video clips (TOY DATA, full pipeline)")
        lines.append(f"{sc['people']} people in 8 clips: {sc['alerts_on_thieves']}/{sc['thieves']} thieves alerted, "
                     f"{sc['alerts_on_honest']} alerts and {sc['reviews_on_honest']} reviews on honest people. "
                     f"Pipeline {tv['pipeline_fps']} FPS (toy colour detector). Alert processing latency (frame read -> alert "
                     f"written): {tv['alert_processing_latency_ms']} ms.")
    rf = res.get("real_fps")
    if rf and "pipeline_fps" in rf:
        lines.append(f"\n## Real footage throughput (vtest.avi, {rf['frames']} frames, {rf['hardware']['device']}: {rf['hardware']['device_name']}, {rf['hardware']['cpu_count']} cores)")
        lines.append(f"Pipeline {rf['pipeline_fps']} FPS end to end; mean ms/frame by stage: {rf['stage_ms_mean']}; {rf['persons_tracked']} person tracks.")
    text = "\n".join(lines)
    print(text)
    return text


def run_bench(root: Path, quick: bool = False, parts: tuple[str, ...] = ("event", "toy", "real")) -> dict:
    store = load_store_config(root / "configs" / "store_gas_station_small.yaml")
    out = root / "results"
    prev = out / "bench.json"
    # Partial runs (--only) update their sections and keep the rest of the last full run.
    res = json.loads(prev.read_text()) if prev.exists() and len(parts) < 3 else {}
    res.update({"generated_unix": time.time(), "machine": platform.platform(), "quick": quick})
    t0 = time.time()
    if "event" in parts:
        print("[bench] event-level simulator ...", flush=True)
        res["event_level"] = event_level(store, hours=50 if quick else 200)
    if "toy" in parts:
        print("[bench] toy video clips (full pipeline) ...", flush=True)
        res["toy_video"] = toy_video(root, root / "out" / "bench_toy")
    if "real" in parts:
        print("[bench] real footage FPS (YOLO) ...", flush=True)
        res["real_fps"] = real_fps(root, root / "out" / "bench_vtest", max_frames=200 if quick else None)
    res["bench_wall_s"] = round(time.time() - t0, 1)
    out.mkdir(exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "bench.md").write_text("# Benchmark results\n\nGenerated by `make bench`. See REPORT.md for what these do and don't prove.\n"
                                  + print_tables(res) + "\n")
    print(f"\nwrote {out / 'bench.json'} and {out / 'bench.md'} ({res['bench_wall_s']} s)")
    return res
