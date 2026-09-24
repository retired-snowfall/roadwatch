"""Traffic-signal state per stop line.

Two sources, fused per stop line:
* behaviour: while the head of a queue waits at the stop line, that approach is red;
  the moment it starts moving is the green onset;
* traffic-light colour, when the camera sees the signal heads: YOLO boxes of class
  "traffic light" are clustered by position; each cluster's lit colour is read from
  HSV and the cluster is attached to the stop line whose waiting pattern it explains best.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .scene import angle_diff, runs
from .tracks import Track

_HSV = {  # (hue ranges, min saturation, min value)
    "red": ([(0, 10), (165, 180)], 90, 140),
    "yellow": ([(12, 35)], 90, 140),
    "green": ([(40, 100)], 60, 120),
}


def light_color(crop: np.ndarray) -> str | None:
    """Colour of the lit lamp in a traffic-light crop, or None if nothing is clearly lit."""
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    counts = {}
    for name, (ranges, smin, vmin) in _HSV.items():
        m = np.zeros(h.shape, bool)
        for lo, hi in ranges:
            m |= (h >= lo) & (h <= hi)
        counts[name] = int((m & (s >= smin) & (v >= vmin)).sum())
    best = max(counts, key=counts.get)
    return best if counts[best] >= max(3, 0.02 * h.size) else None


@dataclass
class LightTrack:
    box: np.ndarray
    t: list = field(default_factory=list)
    state: list = field(default_factory=list)

    def red_at(self, t: float, tol: float = 0.6) -> bool | None:
        if not self.t:
            return None
        ts = np.asarray(self.t)
        k = int(np.argmin(np.abs(ts - t)))
        if abs(ts[k] - t) > tol or self.state[k] is None:
            return None
        return self.state[k] == "red"


def cluster_lights(observations: list[tuple[float, np.ndarray, str | None]], min_obs: int = 10) -> list[LightTrack]:
    """observations: (t, box, colour). Static lights are grouped by box overlap."""
    lights: list[LightTrack] = []
    for t, box, col in observations:
        for lt in lights:
            ix = max(0.0, min(box[2], lt.box[2]) - max(box[0], lt.box[0]))
            iy = max(0.0, min(box[3], lt.box[3]) - max(box[1], lt.box[1]))
            if ix * iy > 0.3 * (box[2] - box[0]) * (box[3] - box[1]):
                lt.t.append(t)
                lt.state.append(col)
                break
        else:
            lights.append(LightTrack(np.asarray(box, float), [t], [col]))
    return [lt for lt in lights if len(lt.t) >= min_obs]


@dataclass
class StopLine:
    a: np.ndarray
    b: np.ndarray
    dir: float

    @property
    def u(self) -> np.ndarray:
        r = np.radians(self.dir)
        return np.array([np.cos(r), np.sin(r)])

    @property
    def n(self) -> np.ndarray:
        return np.array([-self.u[1], self.u[0]])

    @property
    def mid(self) -> np.ndarray:
        return (self.a + self.b) / 2

    @property
    def half(self) -> float:
        return float(np.linalg.norm(self.b - self.a) / 2)

    def along(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts).reshape(-1, 2) - self.mid) @ self.u

    def across(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts).reshape(-1, 2) - self.mid) @ self.n


def front_points(tr: Track, line: StopLine) -> np.ndarray:
    """Approximate front bumper: the box bottom is the rear for vehicles driving away from the camera."""
    away = max(0.0, -line.u[1])
    return tr.foot + line.u[None, :] * (0.8 * tr.size[:, None] * away)


def approaching(tr: Track, line: StopLine, max_angle: float = 60.0) -> np.ndarray:
    return (tr.speed < 0.3) | (angle_diff(tr.heading, line.dir) < max_angle)


def waiting_intervals(tracks: list[Track], line: StopLine, stationary: float) -> list[tuple[float, float, int]]:
    """(t_stop, t_go, tid): vehicles waiting with their front at the line."""
    out = []
    for tr in tracks:
        if tr.group != "vehicle":
            continue
        fr = front_points(tr, line)
        al, ac = line.along(fr), line.across(fr)
        at_line = (al > -2.0 * tr.size) & (al < 0.5 * tr.size) & (np.abs(ac) < line.half + 0.5 * tr.size) \
            & (tr.speed < stationary) & approaching(tr, line)
        for s, e in runs(at_line):
            if tr.t[e] - tr.t[s] >= 2.0 and tr.arrived_moving(s):   # parked cars are not a queue
                out.append((float(tr.t[s]), float(tr.t[e]), tr.tid))
    return out


def red_intervals(waits: list[tuple[float, float, int]], light: LightTrack | None) -> list[tuple[float, float]]:
    iv = [(s, e) for s, e, _ in waits]
    if light is not None:
        ts = np.asarray(light.t)
        red = np.array([st == "red" for st in light.state])
        iv += [(float(ts[s]), float(ts[e])) for s, e in runs(red) if ts[e] - ts[s] >= 1.0]
    iv.sort()
    merged: list[list[float]] = []
    for s, e in iv:
        if merged and s - merged[-1][1] <= 2.0:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def assign_light(waits: list[tuple[float, float, int]], lights: list[LightTrack], duration: float) -> LightTrack | None:
    """The light whose red periods best overlap this line's waiting periods (Jaccard >= 0.3)."""
    if not lights or not waits:
        return None
    grid = np.arange(0, duration, 0.5)
    wait = np.zeros(len(grid), bool)
    for s, e, _ in waits:
        wait |= (grid >= s) & (grid <= e)
    best, best_j = None, 0.3
    for lt in lights:
        red = np.array([bool(lt.red_at(t)) for t in grid])
        inter, union = (red & wait).sum(), (red | wait).sum()
        j = inter / union if union else 0.0
        if j > best_j:
            best, best_j = lt, j
    return best
