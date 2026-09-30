"""`bree bench`: every number we can honestly measure tonight, in one place.

1. Event-level simulator (ledger logic, independent of vision), on a HELD-OUT seed
   (all development/tuning used seed 1; the benchmark reports seed 2):
   - noise levels: perfect vision / baseline assumed error rates / 2x error rates
   - payment modes: POS feed vs. register-dwell only
   - alert-threshold sweep and a per-error-source ablation at baseline noise
2. Toy video clips (rendered, labelled TOY DATA): full pipeline end to end.
3. Real footage FPS: YOLO26n detector + crop pose + ByteTrack on OpenCV's vtest.avi
   (real pedestrians, no theft labels) on this machine.
4. Real data (Phase 2): collects the measured results written by scripts/measure_merl.py,
   scripts/conceal_experiment.py, scripts/eval_retails.py, scripts/eval_ucf.py and
   scripts/measured_error_rates.py, and the simulator re-run with the measured error rates
   ("measured" noise row). Printed in its own section, never mixed with simulated numbers.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from bree.eval.metrics import noise_ablation, run_event_bench
from bree.events.zones import load_store_config
from bree.sim.events_sim import SimConfig, VisionNoise

TEST_SEED = 2
SPREAD_SEEDS = [2, 3, 4, 5, 6]
NOISE_LEVELS = {"perfect": 0.0, "baseline": 1.0, "pessimistic_2x": 2.0}
REAL_FILES = ("measured_error_rates", "merl_measure", "conceal_poselift", "conceal_retails", "conceal_ucf", "speed")


def measured_noise(root: Path) -> VisionNoise | None:
    """VisionNoise with the measured rates in place of the assumed ones (unmeasured fields stay assumed)."""
    f = root / "results" / "measured_error_rates.json"
    if not f.exists():
        return None
    over = {k: v["value"] for k, v in json.loads(f.read_text())["vision_noise"].items() if v.get("measured")}
    return replace(VisionNoise(), **over)


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v}"


def event_level(store, hours: float, seed: int = TEST_SEED, measured: VisionNoise | None = None) -> dict:
    out = {"seed": seed, "hours_per_run": hours, "runs": []}
    levels = [(n, VisionNoise().scaled(k)) for n, k in NOISE_LEVELS.items()]
    if measured is not None:
        levels.append(("measured", measured))
        out["noise_model_measured"] = asdict(measured)
    for mode in ("pos", "dwell"):
        for name, noise in levels:
            r = run_event_bench(store, SimConfig(hours=hours, payment_mode=mode), noise, seed,
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


def real_data(root: Path) -> dict:
    return {n: json.loads((root / "results" / f"{n}.json").read_text())
            for n in REAL_FILES if (root / "results" / f"{n}.json").exists()}


def _real_tables(rd: dict) -> list[str]:
    L = ["\n# REAL DATA (measured on real footage or real pose data; no simulator involved)"]
    if not rd:
        return L + ["No real-data results yet (run the scripts listed in REPORT.md)."]
    m = rd.get("merl_measure")
    if m:
        L.append(f"\n## Pose/track layer on MERL Shopping ({m['n_videos']} test videos, {m['minutes']:.0f} min, overhead camera, {m['hardware']['device']})")
        L.append(f"Reach recall {100*m['reach_recall']:.1f}% of {m['reach_instances']} labelled reaches; "
                 f"{m['false_reach_runs_per_min']:.2f} false reaches / min; {m['extra_person_tracks_per_video']:.1f} extra person track ids per single-shopper video.")
    for key, title in (("conceal_poselift", "PoseLift (held-out videos)"),):
        c = rd.get(key)
        if c:
            cv = c["protocols"]["cv5_by_video"]["pooled_out_of_fold_over_seeds"]
            L.append(f"\n## Concealment: {title}, 5-fold by video x 3 seeds ({cv['model']['n_frames']} frames, {cv['model']['n_pos']} shoplifting)")
            L.append("| scorer | AUC-ROC | AUC-PR | EER |")
            L.append("|---|---|---|---|")
            for k in ("rule", "model"):
                r = cv[k]
                L.append(f"| {k} | {r['auc_roc']['mean']:.3f} +- {r['auc_roc']['std']:.3f} | {r['auc_pr']['mean']:.3f} +- {r['auc_pr']['std']:.3f} | {r['eer']['mean']:.3f} +- {r['eer']['std']:.3f} |")
            ev = c["protocols"]["cv5_by_video"].get("events_pooled_over_folds_and_seeds", {})
            for k, by in ev.items():
                L.append(f"{k} events: " + "; ".join(f"th {th}: {c2['pos_hit']}/{c2['pos']} shoplifting clips caught, "
                                                    f"{c2['neg_triggered']}/{c2['neg']} clean clips triggered" for th, c2 in by.items()))
            lo = c["protocols"].get("leave_one_camera_out", [])
            if lo:
                L.append("Leave one camera out, AUC-ROC rule / model: " + ", ".join(
                    f"cam {r['camera']} {r['rule']['auc_roc']:.2f} / {r['model']['auc_roc']:.2f}" for r in lo))
    for key, title in (("conceal_retails", "RetailS (evaluation only)"), ("conceal_ucf", "UCF-Crime through our pipeline (evaluation only)")):
        c = rd.get(key)
        if not c:
            continue
        test = c.get("staged") or c.get("shoplifting")
        L.append(f"\n## Concealment: {title}")
        L.append("| scorer | AUC-ROC | AUC-PR | EER | would-be triggers / hour of normal footage at thresholds |")
        L.append("|---|---|---|---|---|")
        for k in ("rule", "model"):
            t = test[k]
            trig = ", ".join(f"{th}: {v:.2f}" for th, v in c["normal"][k]["triggers_per_hour_at"].items())
            L.append(f"| {k} | {t['auc_roc']:.3f} | {t['auc_pr']:.3f} | {t['eer']:.3f} | {trig} |")
        L.append(f"Test clips: {len(test.get('videos', [])) or test.get('n_videos')}; normal footage: {c['normal']['hours']:.1f} h.")
    me = rd.get("measured_error_rates")
    if me:
        L.append("\n## Measured vs assumed vision error rates (simulator parameters)")
        L.append("| parameter | assumed | measured | source |")
        L.append("|---|---|---|---|")
        for k, v in me["vision_noise"].items():
            L.append(f"| {k} | {v['assumed']} | {round(v['value'], 3) if v.get('measured') else 'not measured'} | {v['source']} |")
    sp = rd.get("speed")
    if sp:
        L.append(f"\n## Speed ({sp['machine']}, {sp['video']})")
        L.append("| runtime | detector | pose | full pipeline FPS | ms / frame |")
        L.append("|---|---|---|---|---|")
        for r in sp["runs"]:
            L.append(f"| {r['runtime']} | {r['detect_model']} | {r['pose_model']} | {r['fps']:.1f} | {r['ms_per_frame']:.1f} |")
    return L


def print_tables(res: dict) -> str:
    lines = _real_tables(res.get("real_data", {})) + ["\n# SIMULATED (event-level simulator and toy clips; vision error rates are inputs, see above)"]
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


def run_bench(root: Path, quick: bool = False, parts: tuple[str, ...] = ("event", "toy", "real", "realdata")) -> dict:
    store = load_store_config(root / "configs" / "store_gas_station_small.yaml")
    out = root / "results"
    prev = out / "bench.json"
    # Partial runs (--only) update their sections and keep the rest of the last full run.
    res = json.loads(prev.read_text()) if prev.exists() and len(parts) < 4 else {}
    res.update({"generated_unix": time.time(), "machine": platform.platform(), "quick": quick})
    t0 = time.time()
    if "event" in parts:
        print("[bench] event-level simulator ...", flush=True)
        res["event_level"] = event_level(store, hours=50 if quick else 200, measured=measured_noise(root))
    if "toy" in parts:
        print("[bench] toy video clips (full pipeline) ...", flush=True)
        res["toy_video"] = toy_video(root, root / "out" / "bench_toy")
    if "real" in parts:
        print("[bench] real footage FPS (YOLO) ...", flush=True)
        res["real_fps"] = real_fps(root, root / "out" / "bench_vtest", max_frames=200 if quick else None)
    res["real_data"] = real_data(root)   # always refreshed: it only reads result files
    res["bench_wall_s"] = round(time.time() - t0, 1)
    out.mkdir(exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "bench.md").write_text("# Benchmark results\n\nGenerated by `make bench`. See REPORT.md for what these do and don't prove.\n"
                                  + print_tables(res) + "\n")
    print(f"\nwrote {out / 'bench.json'} and {out / 'bench.md'} ({res['bench_wall_s']} s)")
    return res
