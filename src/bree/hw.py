"""Hardware detection and model-size selection.

We pick the model variant from what the machine can actually run:
- CUDA GPU      -> small models ("s"), full 640px input
- Apple MPS     -> small models ("s"), full 640px input
- CPU only      -> nano models ("n"), 640px input (drop to 480 if FPS is too low)
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, asdict


@dataclass
class HardwareProfile:
    device: str          # "cuda:0", "mps", or "cpu"
    device_name: str
    cpu_count: int
    detect_model: str
    pose_model: str
    imgsz: int

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware() -> HardwareProfile:
    cpu_count = os.cpu_count() or 1
    try:
        import torch
    except ImportError:  # core install without the vision extras
        return HardwareProfile("cpu", platform.processor() or platform.machine(), cpu_count,
                               "yolo26n.pt", "yolo26n-pose.pt", 640)

    if torch.cuda.is_available():
        return HardwareProfile("cuda:0", torch.cuda.get_device_name(0), cpu_count,
                               "yolo26s.pt", "yolo26s-pose.pt", 640)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return HardwareProfile("mps", "Apple Silicon (MPS)", cpu_count,
                               "yolo26s.pt", "yolo26s-pose.pt", 640)
    return HardwareProfile("cpu", platform.processor() or platform.machine(), cpu_count,
                           "yolo26n.pt", "yolo26n-pose.pt", 640)


if __name__ == "__main__":
    import json
    print(json.dumps(detect_hardware().to_dict(), indent=2))
