"""`bree` command line.

  bree hw                                  hardware + chosen model sizes
  bree run --source VIDEO|0|rtsp://...     full pipeline on a video / webcam / stream
  bree render-toy --out data/toy           render labelled toy clips
  bree demo                                toy clips end to end, with a scorecard
  bree sim                                 event-level simulator + ledger accuracy
  bree bench                               everything measurable -> results/bench.json
  bree export                              YOLO -> ONNX for edge devices
  bree dashboard                           local web dashboard
  bree shadow --config configs/shadow.yaml pilot shadow mode: silent, would-be alerts + review page
  bree shadow-labels --config ...          labelled shadow data: summary, export, review page
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = ROOT / "configs" / "store_gas_station_small.yaml"


def _weights(name: str) -> str:
    p = ROOT / "models" / name
    return str(p) if p.exists() else name   # ultralytics downloads by name if missing


def make_backend(kind: str, store, imgsz: int | None = None, products: bool = True):
    if kind == "toy":
        from bree.detect.toy import ToyBackend
        return ToyBackend()
    from bree.detect.yolo import YoloBackend
    from bree.hw import detect_hardware
    hw = detect_hardware()
    return YoloBackend(_weights(hw.pose_model), _weights(hw.detect_model), store.product_classes,
                       device=hw.device, imgsz=imgsz or hw.imgsz, products=products)


def cmd_hw(args) -> None:
    from bree.hw import detect_hardware
    print(json.dumps(detect_hardware().to_dict(), indent=2))


def cmd_run(args) -> None:
    import time
    from bree.events.zones import load_store_config
    from bree.ledger.payments import open_payments
    from bree.events.zones import merge_stores
    from bree.pipeline import CameraInput, run_store
    store_paths = args.store or [str(DEFAULT_STORE)]
    if len(store_paths) != len(args.source):
        raise SystemExit("give one --store per --source (several cameras of one store)")
    stores = [load_store_config(p) for p in store_paths]
    cams = [CameraInput(s.camera_id, src, s) for src, s in zip(args.source, stores)]
    backend = make_backend(args.backend, merge_stores(stores), args.imgsz, products=not args.no_products)
    payments = open_payments(args.payments, stream_start_wall=time.time())
    s = run_store(cams, backend, args.out, payments=payments,
                  save_video=not args.no_video, max_frames=args.max_frames)
    print(json.dumps({k: v for k, v in s.__dict__.items() if k != "alerts"}, indent=2))
    print(f"{len(s.alerts)} alert(s); outputs in {args.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="bree", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hw").set_defaults(func=cmd_hw)

    r = sub.add_parser("run", help="run the full pipeline on a video file / webcam / RTSP stream")
    r.add_argument("--source", required=True, action="append",
                   help="video path, webcam index (0), or rtsp:// URL; repeat (with --store) for "
                        "several cameras of one store: one person id across cameras, one ledger")
    r.add_argument("--store", action="append", default=None,
                   help=f"store layout YAML for the matching --source (default {DEFAULT_STORE.name})")
    r.add_argument("--out", default="out/run")
    r.add_argument("--backend", choices=["yolo", "toy"], default="yolo")
    r.add_argument("--payments", default=None, help="payments.jsonl | stdin | http[:PORT]")
    r.add_argument("--imgsz", type=int, default=None)
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4")
    r.add_argument("--no-products", action="store_true", help="people + pose only")
    r.set_defaults(func=cmd_run)

    # Commands implemented in their own modules (registered lazily to keep imports light).
    from bree import commands
    commands.register(sub)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
