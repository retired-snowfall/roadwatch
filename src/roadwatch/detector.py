"""YOLO detector wrapper: device/profile selection, batching, fixed output format."""
from __future__ import annotations

import os

os.environ.setdefault("YOLO_OFFLINE", "true")      # never reach for the network at inference
os.environ.setdefault("YOLO_VERBOSE", "false")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .config import PROFILES, SEED, WEIGHTS_DIR, DetectorProfile  # noqa: E402
from .constants import DETECT_CLASSES  # noqa: E402

_CACHE: dict[str, "Detector"] = {}


def device_kind() -> str:
    forced = os.environ.get("ROADWATCH_DEVICE")
    if forced in ("cpu", "gpu"):
        return forced
    return "gpu" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Detector:
    """Runs YOLO on batches of BGR frames and returns one (N, 6) array per frame:
    x1, y1, x2, y2, confidence, coco_class_id (float32, original pixel coordinates)."""

    def __init__(self, profile: DetectorProfile, device: str, conf: float = 0.10):
        from ultralytics import YOLO

        path = WEIGHTS_DIR / profile.weights
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run weights/download.sh")
        self.profile = profile
        self.device = "cuda:0" if device == "gpu" else "cpu"
        self.half = profile.half and device == "gpu"
        self.conf = conf
        self.model = YOLO(str(path))
        if device == "cpu":
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    def __call__(self, frames: list[np.ndarray], imgsz: int | None = None) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        bs = self.profile.batch
        for i in range(0, len(frames), bs):
            results = self.model.predict(frames[i:i + bs], imgsz=imgsz or self.profile.imgsz,
                                         conf=self.conf, iou=0.6, classes=DETECT_CLASSES,
                                         device=self.device, half=self.half, verbose=False)
            for r in results:
                b = r.boxes
                if b is None or len(b) == 0:
                    out.append(np.zeros((0, 6), np.float32))
                    continue
                out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                           b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32))
        return out


def get_detector(kind: str | None = None) -> Detector:
    """Process-wide detector (Part A and Part B share the loaded weights)."""
    kind = kind or device_kind()
    if kind not in _CACHE:
        seed_everything()
        _CACHE[kind] = Detector(PROFILES[kind], kind)
    return _CACHE[kind]
