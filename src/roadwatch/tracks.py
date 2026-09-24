"""Offline tracks: assembly from tracker output, stitching, rider suppression, kinematics."""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import KinematicsCfg
from .constants import COCO_NAMES


def local_linear(t: np.ndarray, x: np.ndarray, half_window: float) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through samples within +-half_window of each t_i.

    Returns (value, slope) at every sample; x may be (N,) or (N, D). Vectorised with prefix sums.
    """
    x2 = x.reshape(len(t), -1).astype(np.float64)
    t = t.astype(np.float64)
    lo = np.searchsorted(t, t - half_window, "left")
    hi = np.searchsorted(t, t + half_window, "right")

    def csum(a):
        return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])

    c1 = csum(np.ones_like(t))
    ct = csum(t)
    ctt = csum(t * t)
    cx = csum(x2)
    ctx = csum(t[:, None] * x2)
    n = c1[hi] - c1[lo]
    st = ct[hi] - ct[lo] - n * t                       # sum of (t_j - t_i)
    stt = ctt[hi] - ctt[lo] - 2 * t * (ct[hi] - ct[lo]) + n * t * t
    sx = cx[hi] - cx[lo]
    stx = ctx[hi] - ctx[lo] - t[:, None] * sx
    den = n * stt - st * st
    ok = den > 1e-9
    slope = np.where(ok[:, None], (n[:, None] * stx - st[:, None] * sx) / np.where(ok, den, 1)[:, None], 0.0)
    value = np.where(n[:, None] > 0, (sx - slope * st[:, None]) / np.maximum(n, 1)[:, None], x2)
    return value.reshape(x.shape), slope.reshape(x.shape)


@dataclass
class Track:
    tid: int
    group: str
    cls: int
    fidx: np.ndarray            # analysed frame indices
    t: np.ndarray               # seconds
    box: np.ndarray             # (N, 4) raw detections
    conf: np.ndarray
    edge: np.ndarray = field(default=None)       # box touches the image border
    foot: np.ndarray = field(default=None)       # (N, 2) smoothed bottom-centre, px
    vel: np.ndarray = field(default=None)        # (N, 2) px/s
    size: np.ndarray = field(default=None)       # (N,) smoothed sqrt(area), px
    speed: np.ndarray = field(default=None)      # (N,) sizes/s
    accel: np.ndarray = field(default=None)      # (N,) d speed / dt, sizes/s^2
    heading: np.ndarray = field(default=None)    # (N,) degrees, image frame (y down)
    speed_fast: np.ndarray = field(default=None)  # (N,) lightly smoothed speed for abrupt changes
    accel_fast: np.ndarray = field(default=None)

    @property
    def label(self) -> str:
        return COCO_NAMES.get(self.cls, str(self.cls))

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    @property
    def duration(self) -> float:
        return self.end - self.start

    def index_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def compute_kinematics(self, cfg: KinematicsCfg, width: int, height: int) -> None:
        b = self.box
        raw_foot = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
        raw_size = np.sqrt(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1, None))
        self.edge = (b[:, 0] < 3) | (b[:, 1] < 3) | (b[:, 2] > width - 3) | (b[:, 3] > height - 3)
        hw = cfg.smooth_window / 2
        self.foot, self.vel = local_linear(self.t, raw_foot, hw)
        self.size, _ = local_linear(self.t, raw_size, hw)
        self.size = np.maximum(self.size, 4.0)
        self.speed = np.hypot(self.vel[:, 0], self.vel[:, 1]) / self.size
        _, self.accel = local_linear(self.t, self.speed, hw)
        self.heading = np.degrees(np.arctan2(self.vel[:, 1], self.vel[:, 0]))
        _, vel_fast = local_linear(self.t, raw_foot, hw / 2)
        self.speed_fast = np.hypot(vel_fast[:, 0], vel_fast[:, 1]) / self.size
        _, self.accel_fast = local_linear(self.t, self.speed_fast, hw)

    def window(self, t0: float, t1: float) -> slice:
        """Sample slice with t0 <= t <= t1."""
        return slice(int(np.searchsorted(self.t, t0, "left")), int(np.searchsorted(self.t, t1, "right")))

    def median_in(self, values: np.ndarray, t0: float, t1: float, default: float = np.nan) -> float:
        w = values[self.window(t0, t1)]
        return float(np.median(w)) if len(w) else default

    def travelled(self, t0: float, t1: float) -> float:
        """Straight-line displacement between t0 and t1, in object sizes."""
        w = self.window(t0, t1)
        if w.stop - w.start < 2:
            return 0.0
        f = self.foot[w]
        return float(np.linalg.norm(f[-1] - f[0]) / np.median(self.size[w]))

    def arrived_moving(self, i: int, window: float = 3.0) -> bool:
        """Did the object drive in before sample i? Parked vehicles never did."""
        t = self.t[i]
        return t - self.start >= 1.0 and self.travelled(t - window, t) >= 1.5

    def departs_moving(self, i: int, window: float = 4.0) -> bool:
        """Does the object drive off after sample i?"""
        t = self.t[i]
        return self.end - t >= 1.0 and self.travelled(t, t + window) >= 1.5

    def aspect(self) -> np.ndarray:
        """Height / width of the raw boxes (a fallen pedestrian drops well below 1)."""
        return (self.box[:, 3] - self.box[:, 1]) / np.maximum(self.box[:, 2] - self.box[:, 0], 1.0)

    def to_json(self, stride: int = 1) -> dict:
        s = slice(None, None, stride)
        return {"id": self.tid, "group": self.group, "label": self.label,
                "t": np.round(self.t[s], 3).tolist(), "box": np.round(self.box[s]).astype(int).tolist()}


def assemble(records: list[tuple], min_samples: int = 3) -> list[Track]:
    """records: (frame_idx, t, tid, x1, y1, x2, y2, conf, coco_cls, group) per matched detection."""
    by_id: dict[int, list] = defaultdict(list)
    for r in records:
        by_id[r[2]].append(r)
    tracks = []
    for tid, rows in sorted(by_id.items()):
        if len(rows) < min_samples:
            continue
        rows.sort(key=lambda r: r[0])
        arr = np.array([r[:9] for r in rows], dtype=np.float64)
        cls = Counter(int(r[8]) for r in rows).most_common(1)[0][0]
        tracks.append(Track(tid, rows[0][9], cls, arr[:, 0].astype(int), arr[:, 1], arr[:, 3:7], arr[:, 7]))
    return tracks


def _concat(a: Track, b: Track) -> Track:
    votes = Counter({a.cls: len(a.t)}) + Counter({b.cls: len(b.t)})
    return Track(a.tid, a.group, votes.most_common(1)[0][0], np.r_[a.fidx, b.fidx], np.r_[a.t, b.t],
                 np.r_[a.box, b.box], np.r_[a.conf, b.conf])


def stitch(tracks: list[Track], cfg: KinematicsCfg) -> list[Track]:
    """Join fragments of one object broken by occlusion or missed detections.

    A fragment b continues a if it starts after a ends, in the same group, and either
    (moving) b's first box is where a's end velocity predicts within `stitch_gap`, or
    (stationary) both are still and their boxes overlap, within `stitch_gap_static`.
    """
    by_start = sorted(tracks, key=lambda tr: tr.start)
    starts = [tr.start for tr in by_start]
    alive = {tr.tid: tr for tr in tracks}
    max_gap = max(cfg.stitch_gap, cfg.stitch_gap_static)
    # Process in order of end time: any continuation b starts after a ends, so b has not yet
    # absorbed anything itself; a is extended repeatedly until no continuation fits.
    for a0 in sorted(tracks, key=lambda tr: tr.end):
        if a0.tid not in alive:
            continue
        while True:
            a = alive[a0.tid]
            a_end_box = a.box[-1]
            n = min(len(a.t), 6)
            span = max(a.t[-1] - a.t[-n], 1e-3)
            a_vel = (a.box[-1] - a.box[-n]) / span
            a_size = np.sqrt(max((a_end_box[2] - a_end_box[0]) * (a_end_box[3] - a_end_box[1]), 1))
            a_static = np.linalg.norm(a_vel[:2]) / a_size < 0.15
            best, best_cost = None, np.inf
            lo = bisect.bisect_right(starts, a.end)
            hi = bisect.bisect_right(starts, a.end + max_gap)
            for b in by_start[lo:hi]:
                if b.tid not in alive or b.tid == a.tid or b.group != a.group:
                    continue
                gap = b.start - a.end
                m = min(len(b.t), 6)
                b_span = max(b.t[m - 1] - b.t[0], 1e-3)
                b_static = np.linalg.norm((b.box[m - 1] - b.box[0])[:2]) / b_span / a_size < 0.15
                if a_static and b_static and gap <= cfg.stitch_gap_static:
                    iou = _box_iou(a_end_box, b.box[0])
                    if iou > 0.5:
                        cost = gap * (1.5 - iou)
                        if cost < best_cost:
                            best, best_cost = b, cost
                elif gap <= cfg.stitch_gap:
                    pred = a_end_box + a_vel * gap
                    d = np.linalg.norm(((pred[:2] + pred[2:]) - (b.box[0, :2] + b.box[0, 2:])) / 2) / a_size
                    if d < 0.8 + 0.3 * gap:
                        cost = d + gap
                        if cost < best_cost:
                            best, best_cost = b, cost
            if best is None:
                break
            alive[a.tid] = _concat(a, best)
            del alive[best.tid]
    return sorted(alive.values(), key=lambda tr: tr.start)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def drop_riders(tracks: list[Track]) -> list[Track]:
    """Remove person tracks that ride a bicycle/motorcycle for most of their life."""
    wheels = [tr for tr in tracks if tr.group == "two_wheeler"]
    if not wheels:
        return tracks
    frame_boxes: dict[int, list[np.ndarray]] = defaultdict(list)
    for w in wheels:
        for f, b in zip(w.fidx, w.box):
            frame_boxes[int(f)].append(b)
    kept = []
    for tr in tracks:
        if tr.group != "person":
            kept.append(tr)
            continue
        riding = 0
        for f, p in zip(tr.fidx, tr.box):
            parea = max((p[2] - p[0]) * (p[3] - p[1]), 1.0)
            for w in frame_boxes.get(int(f), ()):
                ix = max(0.0, min(p[2], w[2]) - max(p[0], w[0]))
                iy = max(0.0, min(p[3], w[3]) - max(p[1], w[1]))
                if ix * iy / parea > 0.3 and p[3] <= w[3] + 0.25 * (w[3] - w[1]):
                    riding += 1
                    break
        if riding < 0.4 * len(tr.t):
            kept.append(tr)
    return kept


def build_tracks(records: list[tuple], cfg: KinematicsCfg, width: int, height: int) -> list[Track]:
    tracks = assemble(records)
    tracks = stitch(tracks, cfg)
    tracks = drop_riders(tracks)
    out = []
    for tr in tracks:
        if tr.duration < cfg.min_track_duration:
            continue
        tr.compute_kinematics(cfg, width, height)
        out.append(tr)
    return out


class FrameIndex:
    """Which (track, sample) pairs exist at each analysed frame, for pairwise queries."""

    def __init__(self, tracks: list[Track]):
        self.by_frame: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ti, tr in enumerate(tracks):
            for si, f in enumerate(tr.fidx):
                self.by_frame[int(f)].append((ti, si))
        self.frames = np.array(sorted(self.by_frame))
