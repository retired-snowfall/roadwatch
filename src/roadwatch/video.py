"""Video probing and strided frame iteration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def name(self) -> str:
        return Path(self.path).name


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    info = VideoInfo(str(path), fps if fps and fps > 0 else 25.0,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


class FrameSampler:
    """Decides which frame indices to analyse for a target rate; the rate may be lowered mid-stream."""

    def __init__(self, fps: float, target_fps: float):
        self.fps = fps
        self.set_rate(target_fps)
        self._next = 0.0

    def set_rate(self, target_fps: float) -> None:
        self.step = max(1.0, self.fps / max(target_fps, 1e-3))

    def take(self, idx: int) -> bool:
        if idx + 1e-6 >= self._next:
            self._next += self.step
            if self._next <= idx:  # catch up after a rate change
                self._next = idx + self.step
            return True
        return False


def iter_frames(path: str, sampler: FrameSampler) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, t_sec, BGR frame) for the frames the sampler selects.

    Skipped frames are only grabbed (decoded but not converted), which is the
    cheapest way to advance an H.264 stream with OpenCV.
    """
    cap = cv2.VideoCapture(str(path))
    fps = sampler.fps
    idx = 0
    try:
        while True:
            if sampler.take(idx):
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / fps, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
