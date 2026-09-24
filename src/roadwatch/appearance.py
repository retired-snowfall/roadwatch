"""Low-rate pixel analysis for classes that detectors do not cover: static obstacles, fire, smoke.

Frames are sampled at ~1 fps and downscaled. At the end of the video:
* background   = per-pixel median of all samples;
* static blobs = regions where a short-term median (8 s) differs from the background,
                 not covered by any detected road user: objects that appeared and stayed;
* fire         = saturated orange/red, very bright pixels that flicker between samples;
* smoke        = regions turning grey-white, textureless and growing, not covered by objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Blob:
    t0: float
    t1: float
    box: np.ndarray           # full-resolution pixels
    area: float               # share of the image
    hits: int = 1
    covered: int = 0


@dataclass
class AppearanceMonitor:
    width: int
    height: int
    rate: float = 1.0
    work_width: int = 480
    times: list = field(default_factory=list)
    grays: list = field(default_factory=list)
    colors: list = field(default_factory=list)
    covers: list = field(default_factory=list)
    fire: list = field(default_factory=list)
    smoke_hint: list = field(default_factory=list)
    _next: float = 0.0
    max_color: int = 160

    def __post_init__(self):
        self.scale = self.work_width / self.width
        self.work_size = (self.work_width, max(2, int(round(self.height * self.scale))))

    def wants(self, t: float) -> bool:
        return t + 1e-6 >= self._next

    def add(self, t: float, frame: np.ndarray, boxes: np.ndarray) -> None:
        self._next = t + 1.0 / self.rate
        small = cv2.resize(frame, self.work_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cover = np.zeros(gray.shape, np.uint8)
        for b in np.asarray(boxes).reshape(-1, 4) * self.scale:
            x1, y1, x2, y2 = b.astype(int)
            pad = max(2, int(0.15 * (x2 - x1)))
            cover[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 1
        self.times.append(t)
        self.grays.append(gray)
        self.covers.append(np.packbits(cover))
        if len(self.colors) < self.max_color:
            self.colors.append(small)
        self.fire.append(np.packbits(fire_mask(small)))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        smoke = ((hsv[..., 1] <= 35) & (hsv[..., 2] >= 110) & (hsv[..., 2] <= 235)).astype(np.uint8)
        self.smoke_hint.append(np.packbits(smoke))

    # ------------------------------------------------------------------ results
    def _unpack(self, packed: np.ndarray) -> np.ndarray:
        h, w = self.work_size[1], self.work_size[0]
        return np.unpackbits(packed)[: h * w].reshape(h, w).astype(bool)

    def background(self) -> np.ndarray | None:
        if not self.colors:
            return None
        return np.median(np.stack(self.colors), axis=0).astype(np.uint8)

    def static_blobs(self, road: np.ndarray, min_area: float, window: int = 8, thr: float = 28.0) -> list[Blob]:
        """road: boolean mask at work resolution."""
        if len(self.grays) < window + 2:
            return []
        stack = np.stack(self.grays).astype(np.int16)
        bg = np.median(stack, axis=0)
        img_area = stack.shape[1] * stack.shape[2]
        blobs: list[Blob] = []
        kernel = np.ones((3, 3), np.uint8)
        for k in range(window, len(self.grays)):
            short = np.median(stack[k - window:k + 1], axis=0)
            diff = short - bg
            diff -= np.median(diff)                      # global illumination drift
            covered = np.zeros(bg.shape, bool)
            for j in range(k - window, k + 1):
                covered |= self._unpack(self.covers[j])
            mask = ((np.abs(diff) > thr) & road & ~covered).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
            t = self.times[k]
            for c in range(1, n):
                x, y, w, h, a = stats[c]
                if a / img_area < min_area:
                    continue
                box = np.array([x, y, x + w, y + h], float) / self.scale
                for bl in blobs:
                    if bl.t1 >= t - 3.0 and _iou(bl.box, box) > 0.3:
                        bl.t1, bl.hits = t, bl.hits + 1
                        bl.box = 0.7 * bl.box + 0.3 * box
                        break
                else:
                    blobs.append(Blob(t - window / self.rate / 2, t, box, a / img_area))
        return blobs

    def fire_series(self, moving_boxes: dict[int, np.ndarray], road: np.ndarray,
                    min_area: float = 0.0008) -> list[tuple[float, float]]:
        """(t, fire_area_share) per sample.

        Fire pixels must be absent from the background (paint, signs, lamps), near the road,
        outside moving vehicles (tail lights), and flicker between consecutive samples.
        """
        bg = self.background()
        if bg is None:
            return []
        static = cv2.dilate(fire_mask(bg), np.ones((7, 7), np.uint8)).astype(bool)
        near_road = cv2.dilate(road.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        out = []
        prev = None
        for k, (t, packed) in enumerate(zip(self.times, self.fire)):
            m = self._unpack(packed) & ~static & near_road
            for b in moving_boxes.get(k, ()):
                x1, y1, x2, y2 = (np.asarray(b) * self.scale).astype(int)
                m[max(0, y1):y2, max(0, x1):x2] = False
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            area = m.mean()
            flicker = 0.0
            if prev is not None and m.any():
                flicker = (m ^ prev).sum() / max(1, (m | prev).sum())
            prev = m
            out.append((t, area if area >= min_area and 0.25 <= flicker <= 0.9 else 0.0))
        return out

    def smoke_series(self, road: np.ndarray, min_area: float = 0.01) -> list[tuple[float, float]]:
        """(t, smoke_area_share): grey, textureless regions not in the background, off any object."""
        if len(self.grays) < 5:
            return []
        stack = np.stack(self.grays).astype(np.int16)
        bg = np.median(stack, axis=0)
        near_road = cv2.dilate(road.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        out = []
        for k, t in enumerate(self.times):
            g = self.grays[k]
            texture = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (5, 5), 0), cv2.CV_16S)) < 6
            m = self._unpack(self.smoke_hint[k]) & ((stack[k] - bg) > 25) & texture & near_road \
                & ~self._unpack(self.covers[k])
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            out.append((t, float(m.mean()) if m.mean() >= min_area else 0.0))
        return out


def fire_mask(bgr: np.ndarray) -> np.ndarray:
    """Very bright, saturated orange-red pixels (uint8 0/1)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r, b = bgr[..., 2].astype(int), bgr[..., 0].astype(int)
    return (((h <= 18) | (h >= 172)) & (s >= 130) & (v >= 225) & (r - b > 90)).astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
