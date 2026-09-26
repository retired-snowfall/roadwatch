"""Video probing and fast, strided frame reading at a fixed working resolution.

The organisers' camera writes 4K 10-bit 4:2:2 H.264 at ~140 Mbit/s; decoding every frame at
full size costs about one video duration on 4 CPU cores. Analysis never needs more than
~10 frames per second or more than 1920 px of width, so the reader:

* decodes with PyAV and tells the decoder to drop non-reference frames (the camera's
  B-frames, two of every three) - they are never displayed to us, so nothing is corrupted;
* converts and downsizes the kept frames in libswscale straight to the working size;
* runs in a background thread so decoding overlaps detection on the GPU.

All coordinates downstream (tracks, scene model, events) are in working pixels, so a 4K
test video and a 1080p working copy of the same camera give the same geometry.
If PyAV is unavailable the reader falls back to OpenCV (every frame decoded).
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger("roadwatch")
WORK_WIDTH = 1920


def work_size(width: int, height: int, max_width: int = WORK_WIDTH) -> tuple[int, int]:
    """Working (width, height): the frame downsized to at most max_width, even dimensions."""
    if width <= max_width:
        return width, height
    s = max_width / width
    return max_width, max(2, int(round(height * s / 2)) * 2)


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int                        # source pixels
    height: int
    n_frames: int
    work_width: int = 0               # analysis pixels (see work_size)
    work_height: int = 0

    def __post_init__(self):
        if not self.work_width:
            self.work_width, self.work_height = work_size(self.width, self.height)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def work(self) -> tuple[int, int]:
        return self.work_width, self.work_height


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
    """Decides which frames to analyse for a target rate; the rate may be lowered mid-stream.

    take(idx) is for a stream that delivers every frame (Part B); take_time(t) is for a reader
    that delivers an irregular subset: a frame is kept when it opens a new 1/rate time bin.
    """

    def __init__(self, fps: float, target_fps: float):
        self.fps = fps
        self._next = 0.0
        self._bin = -1
        self._last_t: float | None = None
        self.set_rate(target_fps)

    def set_rate(self, target_fps: float) -> None:
        self.step = max(1.0, self.fps / max(target_fps, 1e-3))
        if self._last_t is not None:  # re-express the last kept frame's bin in the new bin width
            self._bin = int((self._last_t + 1e-6) * self.fps / self.step)

    @property
    def rate(self) -> float:
        return self.fps / self.step

    def take(self, idx: int) -> bool:
        if idx + 1e-6 >= self._next:
            self._next += self.step
            if self._next <= idx:  # catch up after a rate change
                self._next = idx + self.step
            return True
        return False

    def take_time(self, t: float) -> bool:
        b = int((t + 1e-6) * self.fps / self.step)
        if b > self._bin:
            self._bin = b
            self._last_t = t
            return True
        return False


def _decode_pyav(path: str, sampler: FrameSampler, size: tuple[int, int], out: queue.Queue,
                 stop: threading.Event) -> None:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.skip_frame = "NONREF"
        tb = float(stream.time_base)
        t0 = stream.start_time if stream.start_time is not None else 0
        for frame in container.decode(stream):
            if stop.is_set():
                return
            if frame.pts is None:
                continue
            t = max(0.0, (frame.pts - t0) * tb)
            if not sampler.take_time(t):
                continue
            img = frame.to_ndarray(width=size[0], height=size[1], format="bgr24", interpolation="AREA")
            idx = int(round(t * sampler.fps))
            _put(out, (idx, idx / sampler.fps, img), stop)


def _decode_cv2(path: str, sampler: FrameSampler, size: tuple[int, int], out: queue.Queue,
                stop: threading.Event) -> None:
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while not stop.is_set():
            if sampler.take_time(idx / sampler.fps):
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != size:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                _put(out, (idx, idx / sampler.fps, frame), stop)
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()


def _put(out: queue.Queue, item, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            out.put(item, timeout=0.2)
            return
        except queue.Full:
            continue


_DONE = object()


def _threaded(decode, path: str, sampler: FrameSampler, size: tuple[int, int], prefetch: int,
              state: dict) -> Iterator[tuple[int, float, np.ndarray]]:
    """Run a decoder in a background thread; frames pass through a bounded queue."""
    out: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def worker() -> None:
        try:
            decode(path, sampler, size, out, stop)
        except Exception as exc:  # re-raised in the consumer thread below
            state["error"] = exc
        finally:
            _put(out, _DONE, stop)

    th = threading.Thread(target=worker, name="roadwatch-decode", daemon=True)
    th.start()
    try:
        while True:
            item = out.get()
            if item is _DONE:
                break
            state["delivered"] += 1
            yield item
    finally:
        stop.set()
        th.join(timeout=5)


def iter_frames(path: str, sampler: FrameSampler, size: tuple[int, int] | None = None,
                prefetch: int = 16) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, t_sec, BGR frame at `size`) for the frames the sampler selects.

    t_sec is frame_index / fps, the same clock run_submission.py gives Part B.
    """
    if size is None:
        size = probe(path).work
    try:
        import av  # noqa: F401
        decode = _decode_pyav
    except ImportError:
        decode = _decode_cv2
    state = {"delivered": 0, "error": None}
    yield from _threaded(decode, path, sampler, size, prefetch, state)
    if state["error"] is None:
        return
    if decode is _decode_pyav and state["delivered"] == 0:
        log.warning("PyAV could not read %s (%s); falling back to OpenCV", path, state["error"])
        sampler._bin, sampler._last_t = -1, None
        state = {"delivered": 0, "error": None}
        yield from _threaded(_decode_cv2, path, sampler, size, prefetch, state)
    if state["error"] is not None:
        raise state["error"]
