"""Online multi-object tracker in the style of ByteTrack (Zhang et al., 2022, MIT licence).

Differences from the reference implementation, all for a fixed CCTV view analysed at 5-8 fps:
* time steps are in seconds (frames may be skipped irregularly);
* association never crosses class groups (vehicle / two_wheeler / person / animal);
* a centre-distance fallback rescues fast objects whose predicted box no longer overlaps;
* stationary tracks survive much longer without detections (occlusion by passing traffic).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackerCfg
from .constants import COCO_GROUPS

_REF_FPS = 30.0
_SP, _SV = 1 / 20, 1 / 160


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    w = m[2] * m[3]
    return np.array([m[0] - w / 2, m[1] - m[3] / 2, m[0] + w / 2, m[1] + m[3] / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height); velocities in units per second."""

    @staticmethod
    def initiate(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = z[3]
        mean = np.r_[z, np.zeros(4)]
        vel = _SV * h * _REF_FPS
        std = [2 * _SP * h, 2 * _SP * h, 1e-2, 2 * _SP * h, 10 * vel, 10 * vel, 1e-4, 10 * vel]
        return mean, np.diag(np.square(std))

    @staticmethod
    def predict(mean: np.ndarray, cov: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        f = np.eye(8)
        f[:4, 4:] = np.eye(4) * dt
        k = max(dt * _REF_FPS, 1e-3)
        h = mean[3]
        vel = _SV * h * _REF_FPS
        q = np.diag(np.square([_SP * h, _SP * h, 1e-2, _SP * h, vel, vel, 1e-4, vel])) * k
        return f @ mean, f @ cov @ f.T + q

    @staticmethod
    def update(mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        hmat = np.eye(4, 8)
        r = np.diag(np.square([_SP * h, _SP * h, 1e-1, _SP * h]))
        s = hmat @ cov @ hmat.T + r
        gain = np.linalg.solve(s, (cov @ hmat.T).T).T
        mean = mean + gain @ (z - hmat @ mean)
        cov = cov - gain @ s @ gain.T
        return mean, cov


@dataclass
class STrack:
    tid: int
    group: str
    mean: np.ndarray
    cov: np.ndarray
    t_seen: float
    t_state: float
    conf: float
    box: np.ndarray
    hits: int = 1
    state: str = "tentative"  # tentative | tracked | lost
    votes: Counter = field(default_factory=Counter)

    @property
    def pred_box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the bottom-centre (ground contact) point in px/s."""
        return np.array([self.mean[4], self.mean[5] + self.mean[7] / 2])

    @property
    def size(self) -> float:
        w = self.mean[2] * self.mean[3]
        return float(np.sqrt(max(w * self.mean[3], 1.0)))

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity) / self.size)

    @property
    def cls(self) -> int:
        return self.votes.most_common(1)[0][0]


def _dedupe(dets: np.ndarray, groups: np.ndarray, thr: float = 0.7) -> np.ndarray:
    """Class-agnostic NMS within a group (YOLO often returns car+truck boxes on one vehicle)."""
    keep = np.ones(len(dets), bool)
    order = np.argsort(-dets[:, 4], kind="stable")
    iou = iou_matrix(dets[:, :4], dets[:, :4])
    for i_pos, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[i_pos + 1:]:
            if keep[j] and groups[i] == groups[j] and iou[i, j] > thr:
                keep[j] = False
    return keep


class ByteTracker:
    def __init__(self, cfg: TrackerCfg | None = None):
        self.cfg = cfg or TrackerCfg()
        self.tracks: list[STrack] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks, self._next_id = [], 1

    # ------------------------------------------------------------------ association helpers
    def _assign(self, tracks: list[STrack], boxes: np.ndarray, groups: np.ndarray, min_iou: float):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        tb = np.array([t.pred_box for t in tracks])
        iou = iou_matrix(tb, boxes)
        tg = np.array([t.group for t in tracks])
        iou[tg[:, None] != groups[None, :]] = 0.0
        cost = 1.0 - iou
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
        mr = {r for r, _ in matches}
        mc = {c for _, c in matches}
        return matches, [i for i in range(len(tracks)) if i not in mr], [j for j in range(len(boxes)) if j not in mc]

    def _assign_centres(self, tracks: list[STrack], boxes: np.ndarray, groups: np.ndarray, t: float):
        """Centre-distance fallback. Young tracks (velocity still unknown) get the full gate; an
        established track only reaches as far as its own motion could take it, so a car stopped
        in a queue cannot hand its ID to the car behind it while it is briefly hidden."""
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        cfg = self.cfg
        gate = np.array([cfg.center_gate if trk.hits < cfg.young_hits else
                         min(cfg.center_gate, cfg.center_gate_static + 1.5 * trk.speed * max(t - trk.t_seen, 0.1))
                         for trk in tracks])
        tc = np.array([t.mean[:2] for t in tracks])
        ts = np.array([t.size for t in tracks])
        bc = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
        bs = np.sqrt(np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1, None))
        d = np.linalg.norm(tc[:, None] - bc[None], axis=2) / ts[:, None]
        ratio = bs[None, :] / ts[:, None]
        bad = (np.array([t.group for t in tracks])[:, None] != groups[None, :]) | (ratio < 0.6) | (ratio > 1.7)
        d[bad] = 1e6
        rows, cols = linear_sum_assignment(d)
        matches = [(r, c) for r, c in zip(rows, cols) if d[r, c] < gate[r]]
        mr = {r for r, _ in matches}
        mc = {c for _, c in matches}
        return matches, [i for i in range(len(tracks)) if i not in mr], [j for j in range(len(boxes)) if j not in mc]

    def _apply(self, trk: STrack, det: np.ndarray, t: float) -> None:
        trk.mean, trk.cov = KalmanXYAH.update(trk.mean, trk.cov, xyxy_to_xyah(det[:4]))
        trk.box = det[:4].copy()
        trk.conf = float(det[4])
        trk.votes[int(det[5])] += 1
        trk.hits += 1
        trk.t_seen = t
        if trk.state == "lost" or trk.hits >= self.cfg.min_hits:
            trk.state = "tracked"

    # ------------------------------------------------------------------ main step
    def update(self, dets: np.ndarray, t: float) -> list[STrack]:
        """Advance to time t with detections (N, 6); return tracks matched in this frame."""
        cfg = self.cfg
        dets = np.asarray(dets, np.float32).reshape(-1, 6)
        dets = dets[np.isin(dets[:, 5].astype(int), list(COCO_GROUPS))]
        groups = np.array([COCO_GROUPS[int(c)] for c in dets[:, 5]], dtype=object)
        if len(dets):
            keep = _dedupe(dets, groups)
            dets, groups = dets[keep], groups[keep]

        for trk in self.tracks:
            trk.mean, trk.cov = KalmanXYAH.predict(trk.mean, trk.cov, t - trk.t_state)
            trk.t_state = t

        high = dets[:, 4] >= cfg.high_thresh
        hi_idx, lo_idx = np.where(high)[0], np.where(~high)[0]
        pool = [trk for trk in self.tracks]
        matched: list[STrack] = []

        m1, um_t, um_d = self._assign(pool, dets[hi_idx, :4], groups[hi_idx], cfg.match_iou)
        for r, c in m1:
            self._apply(pool[r], dets[hi_idx[c]], t)
            matched.append(pool[r])
        rest = [pool[i] for i in um_t]
        rest_hi = hi_idx[um_d]
        m1b, um_t2, um_d2 = self._assign_centres(rest, dets[rest_hi, :4], groups[rest_hi], t)
        for r, c in m1b:
            self._apply(rest[r], dets[rest_hi[c]], t)
            matched.append(rest[r])
        unmatched_tracks = [rest[i] for i in um_t2]
        new_idx = rest_hi[um_d2]

        live = [trk for trk in unmatched_tracks if trk.state == "tracked"]
        m2, um_t3, _ = self._assign(live, dets[lo_idx, :4], groups[lo_idx], cfg.low_match_iou)
        for r, c in m2:
            self._apply(live[r], dets[lo_idx[c]], t)
            matched.append(live[r])
        still = {id(live[i]) for i in um_t3} | {id(trk) for trk in unmatched_tracks if trk.state != "tracked"}

        survivors = []
        for trk in self.tracks:
            if id(trk) in still:
                if trk.state == "tentative":
                    continue
                trk.state = "lost"
                limit = cfg.max_lost_static if trk.speed < 0.2 else cfg.max_lost
                if t - trk.t_seen > limit:
                    continue
            survivors.append(trk)
        self.tracks = survivors

        for j in new_idx:
            if dets[j, 4] < cfg.new_track_thresh:
                continue
            mean, cov = KalmanXYAH.initiate(xyxy_to_xyah(dets[j, :4]))
            trk = STrack(self._next_id, groups[j], mean, cov, t, t, float(dets[j, 4]), dets[j, :4].copy())
            trk.votes[int(dets[j, 5])] += 1
            self._next_id += 1
            self.tracks.append(trk)
            matched.append(trk)
        return matched
