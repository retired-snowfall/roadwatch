"""Part B: causal accident-risk estimator.

step() sees frames in order and nothing else. Every k-th frame (k from the target
analysis rate) is detected and fed to an online tracker; the other frames return the
last score. Cues computed from the current tracks only:

* conflict  closing pairs of road users whose constant-velocity closest approach is
            within ~half a size inside the horizon; risk grows as time-to-collision shrinks
* braking   a vehicle decelerating hard while another road user is close
* wrong way a vehicle moving against the lane direction learned on the sample videos
* impact    boxes that touch while one party stops abruptly (the crash itself; those
            frames are ignored by the metric, but it keeps the alarm from flickering)
* soft      closing proximity of any pair, capped at 0.3: never an alarm, but it ranks
            approaching road users above empty or parallel traffic (the AP term)

Cues are fused as 1 - prod(1 - c) and smoothed with a fast attack and a slow, held release,
so a score >= 0.5 means "a collision is probably seconds away".
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from . import budget
from .config import CFG, WEIGHTS_DIR, Config
from .constants import MOTORISED, ROAD_USERS
from .detector import device_kind, get_detector
from .scene import SceneModel, angle_diff, thumbnail
from .tracker import ByteTracker, STrack
from .video import FrameSampler, work_size


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class CausalRisk:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.prior = SceneModel.load(WEIGHTS_DIR / "scene.json")
        self.detector = None

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.fps = float(meta.get("fps") or 25.0)
        kind = device_kind()
        rc = self.cfg.risk
        self.sampler = FrameSampler(self.fps, rc.analysis_fps_gpu if kind == "gpu" else rc.analysis_fps_cpu)
        if self.detector is None:
            self.detector = get_detector(kind)
        self.imgsz = min(self.detector.profile.imgsz, 960)
        self.tracker = ByteTracker(self.cfg.tracker)
        self.history: dict[int, deque] = {}
        self.idx = 0
        self.score = 0.0
        self.raw = 0.0
        self.hold_until = -1.0
        self.scene: SceneModel | None = None
        self.scene_checked = False
        self.last_cues: dict = {}
        self.work: tuple[int, int] | None = None   # set from the first analysed frame
        n, fps = int(meta.get("n_frames") or 0), self.fps
        self.duration = n / fps if n > 0 else 0.0
        self.deadline = budget.deadline(meta.get("video_id", ""), self.duration) if self.duration else None
        self._guard: tuple[float, float] | None = None   # (wall clock, video time) at the last check
        self.min_px = 0.0

    # ------------------------------------------------------------------ cues
    def _check_scene(self, frame: np.ndarray) -> None:
        """Use the prior lane directions only if this is the same camera view (first frame only)."""
        self.scene_checked = True
        p = self.prior
        if p is None or (p.width, p.height) != (frame.shape[1], frame.shape[0]):
            return
        probe = SceneModel(p.width, p.height, p.cfg)
        probe.thumb = thumbnail(frame)
        if probe.similar_to(p, min_corr=0.5):
            self.scene = p

    def _conflict(self, tracks: list[STrack]) -> tuple[float, tuple, float]:
        """(conflict cue, pair, soft proximity). The soft term ranks every closing pair, however
        far from a collision course, so frames before an accident outrank ordinary traffic in AP."""
        horizon = self.cfg.risk.horizon
        best, pair, soft = 0.0, (), 0.0
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                if a.group not in MOTORISED and b.group not in MOTORISED:
                    continue
                pa = np.array([(a.box[0] + a.box[2]) / 2, a.box[3]])
                pb = np.array([(b.box[0] + b.box[2]) / 2, b.box[3]])
                size = (a.size + b.size) / 2
                rel, rv = pb - pa, b.velocity - a.velocity
                closing = float(np.linalg.norm(rv) / size)
                if closing < 0.6:
                    continue
                rv2 = float(rv @ rv)
                tstar = -float(rel @ rv) / rv2
                if not (0.0 < tstar < horizon):
                    continue
                dstar = float(np.linalg.norm(rel + rv * tstar) / size)
                soft = max(soft, float(np.exp(-tstar / 2.5) * np.exp(-dstar / 1.2) * min(1.0, closing / 1.5)))
                if dstar > 0.9:
                    continue
                # time term: ~0.5 at 1.5 s, ~0.8 at 0.6 s; distance term favours head-on geometry
                c = _sigmoid((1.5 - tstar) / (0.4 * self.cfg.risk.ttc_scale)) * (1.0 - 0.6 * dstar / 0.9)
                c *= min(1.0, closing / 1.5)
                if c > best:
                    best, pair = c, (a.tid, b.tid)
        return best, pair, soft

    def _braking(self, tracks: list[STrack], t: float) -> float:
        best = 0.0
        for a in tracks:
            h = self.history.get(a.tid)
            if a.group != "vehicle" or h is None or len(h) < 4:
                continue
            past = [v for tt, v in h if t - 1.3 <= tt <= t - 0.6]
            if not past:
                continue
            drop = max(past) - a.speed
            if max(past) < 0.8 or drop < 0.5 * max(past):
                continue
            near = any(b.tid != a.tid and np.hypot(*(b.mean[:2] - a.mean[:2])) < 3.0 * a.size for b in tracks)
            if near:
                best = max(best, min(0.45, 0.25 * drop))
        return best

    def _wrong_way(self, tracks: list[STrack]) -> float:
        if self.scene is None:
            return 0.0
        best = 0.0
        for a in tracks:
            if a.group not in MOTORISED or a.speed < 0.8 or a.hits < 6:
                continue
            foot = np.array([(a.box[0] + a.box[2]) / 2, a.box[3]])
            lane, ok = self.scene.lane_direction(foot)
            if ok[0]:
                heading = np.degrees(np.arctan2(a.velocity[1], a.velocity[0]))
                if angle_diff(heading, lane[0]) > 130:
                    best = max(best, 0.25)
        return best

    def _impact(self, tracks: list[STrack], t: float) -> float:
        best = 0.0
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                ix = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
                iy = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
                if ix <= 0 or iy <= 0:
                    continue
                for x in (a, b):
                    h = self.history.get(x.tid)
                    if x.group in MOTORISED and h and len(h) >= 4:
                        past = [v for tt, v in h if t - 1.0 <= tt <= t - 0.4]
                        if past and max(past) > 1.2 and x.speed < 0.35 * max(past):
                            best = max(best, 0.6)
        return best

    def _time_guard(self, t: float) -> None:
        """Lower the detection rate if, at the recent pace (harness decoding included), Part B would
        finish too close to the harness deadline for Part A + Part B."""
        rc = self.cfg.risk
        if self.deadline is None or self._guard is None or t - self._guard[1] < rc.guard_every:
            return
        wall0, t0 = self._guard
        now = time.perf_counter()
        self._guard = (now, t)
        pace = (now - wall0) / (t - t0)                     # wall seconds per video second, recently
        if now + pace * max(0.0, self.duration - t) > self.deadline and self.sampler.rate > rc.min_fps:
            self.sampler.set_rate(max(rc.min_fps, 0.7 * self.sampler.rate))

    # ------------------------------------------------------------------ main
    def step(self, frame: np.ndarray, t: float) -> float:
        take = self.sampler.take(self.idx)
        self.idx += 1
        if not take:
            return self.score
        self._time_guard(t)
        if self.work is None:
            self.work = work_size(frame.shape[1], frame.shape[0])
            self.min_px = self.cfg.events.min_size * self.work[0]
        if (frame.shape[1], frame.shape[0]) != self.work:  # same working pixels as Part A and the scene prior
            frame = cv2.resize(frame, self.work, interpolation=cv2.INTER_AREA)
        if not self.scene_checked:
            self._check_scene(frame)
        dets = self.detector([frame], imgsz=self.imgsz)[0]
        self.tracker.update(dets, t)
        live = [x for x in self.tracker.tracks if x.state == "tracked" and x.t_seen >= t - 1e-6
                and x.group in ROAD_USERS and x.size >= self.min_px]
        for x in live:
            self.history.setdefault(x.tid, deque(maxlen=24)).append((t, x.speed))
        alive = {x.tid for x in self.tracker.tracks}
        for tid in [k for k in self.history if k not in alive]:
            del self.history[tid]

        conflict, pair, soft = self._conflict(live)
        cues = {"conflict": conflict, "braking": self._braking(live, t), "wrong_way": self._wrong_way(live),
                "impact": self._impact(live, t)}
        fused = 1.0 - float(np.prod([1.0 - c for c in cues.values()]))
        # the soft term only orders quiet frames: capped at 0.3, it can never raise an alarm by itself
        fused = max(fused, 0.3 * soft)
        self.last_cues = {**cues, "soft": soft, "pair": pair}
        rc = self.cfg.risk
        if fused >= self.raw:
            self.raw = fused                                  # fast attack
            if fused >= 0.5:
                self.hold_until = t + rc.hold
        elif t > self.hold_until:
            self.raw = rc.ema * self.raw + (1 - rc.ema) * fused  # slow release after the hold
        self.score = float(np.clip(self.raw, 0.0, 1.0))
        if self._guard is None:  # start the pace clock after the first frame (detector warm-up excluded)
            self._guard = (time.perf_counter(), t)
        return self.score


def risk_curve(video_path: str, progress=None, cfg: Config = CFG) -> list[list[float]]:
    """Stream every frame of a video through the estimator, exactly as run_submission.py does."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    est = CausalRisk(cfg)
    est.reset({"video_id": str(video_path), "fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "n_frames": n})
    curve, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        curve.append([round(t, 4), round(float(est.step(frame, t)), 4)])
        idx += 1
        if progress and idx % 50 == 0:
            progress("risk", min(1.0, idx / max(n, 1)))
    cap.release()
    return curve
