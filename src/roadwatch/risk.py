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

from . import budget, register
from .config import CFG, WEIGHTS_DIR, Config
from .constants import MOTORISED, ROAD_USERS
from .detector import device_kind, get_detector
from .events.base import size_factor
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
        self.H: np.ndarray | None = None          # this video's view -> the prior's reference view
        self.scene_checked = False
        self.last_cues: dict = {}
        self.streak = 0                            # consecutive analysed frames with fused >= 0.5
        self.ww_since: dict[int, float] = {}       # track id -> first time seen against its lane
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
        if p.background is not None:     # the camera may be re-aimed: register instead of comparing
            self.H = register.homography(frame, p.background)
            self.scene = p if self.H is not None else None
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
                size = min((a.size + b.size) / 2, 1.5 * min(a.size, b.size))   # a bus must not shrink the gap
                rel, rv = pb - pa, b.velocity - a.velocity
                closing = float(np.linalg.norm(rv) / size)
                if closing < 0.6:
                    continue
                rv2 = float(rv @ rv)
                tstar = -float(rel @ rv) / rv2
                if not (0.0 < tstar < horizon):
                    continue
                if self._following(a, b) and not (tstar < 1.0 and float(np.linalg.norm(rv)) >
                                                  0.5 * max(np.linalg.norm(a.velocity), np.linalg.norm(b.velocity))):
                    continue  # closing up on the car ahead in the same lane: ordinary traffic
                dstar = float(np.linalg.norm(rel + rv * tstar) / size)
                soft = max(soft, float(np.exp(-tstar / 2.5) * np.exp(-dstar / 1.2) * min(1.0, closing / 1.5)))
                gap = self.cfg.risk.max_cpa
                if dstar > gap:
                    continue  # passing in the next lane is not a collision course
                # time term: ~0.5 at the knee, higher as contact nears; distance term favours head-on geometry
                c = _sigmoid((self.cfg.risk.ttc_knee - tstar) / (0.4 * self.cfg.risk.ttc_scale)) * (1.0 - 0.6 * dstar / gap)
                c *= min(1.0, closing / 1.5)
                if c > best:
                    best, pair = c, (a.tid, b.tid)
        return best, pair, soft

    def _near_field(self, x: STrack, height: int) -> bool:
        """Below the far road (in the reference view when registered): up there boxes overlap in
        perspective without touching on the ground."""
        foot = np.array([[(x.box[0] + x.box[2]) / 2, x.box[3]]])
        if self.H is not None:
            foot = register.warp_points(self.H, foot)
        return float(foot[0, 1]) >= self.cfg.risk.min_y * height

    @staticmethod
    def _following(a: STrack, b: STrack) -> bool:
        """Both moving in nearly the same direction (one behind the other, or side by side)."""
        va, vb = a.velocity, b.velocity
        if min(np.linalg.norm(va) / a.size, np.linalg.norm(vb) / b.size) < 0.3:
            return False
        return angle_diff(np.degrees(np.arctan2(va[1], va[0])), np.degrees(np.arctan2(vb[1], vb[0]))) < 35

    @staticmethod
    def _partner_ahead(a: STrack, tracks: list[STrack]) -> bool:
        """Another road user in front of `a` (along its recent heading), within 3 sizes: something to brake for."""
        h = a.velocity
        n = float(np.hypot(*h))
        if n < 1e-6:
            return False
        u = h / n
        fa = np.array([(a.box[0] + a.box[2]) / 2, a.box[3]])
        for b in tracks:
            if b.tid == a.tid:
                continue
            rel = np.array([(b.box[0] + b.box[2]) / 2, b.box[3]]) - fa
            ahead, lateral = float(rel @ u), abs(float(rel @ np.array([-u[1], u[0]])))
            if 0.0 < ahead < 3.0 * a.size and lateral < 1.2 * a.size:
                return True
        return False

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
            if self._partner_ahead(a, tracks):
                best = max(best, min(self.cfg.risk.brake_cap, 0.25 * drop))
        return best

    def _wrong_way(self, tracks: list[STrack], t: float) -> float:
        """Driving against the learned lane direction, sustained (a turn across the lanes is not)."""
        if self.scene is None or (self.prior is not None and self.prior.background is not None and self.H is None):
            return 0.0
        best = 0.0
        flagged = set()
        for a in tracks:
            if a.group not in MOTORISED or a.speed < 0.8 or a.hits < 6:
                continue
            foot = np.array([(a.box[0] + a.box[2]) / 2, a.box[3]])
            ahead = foot + 0.5 * a.velocity
            if self.H is not None:
                foot, ahead = register.warp_points(self.H, np.array([foot, ahead]))
            lane, ok = self.scene.lane_direction(foot)
            if ok[0]:
                v = ahead - foot
                heading = np.degrees(np.arctan2(v[1], v[0]))
                if angle_diff(heading, lane[0]) > 130:
                    flagged.add(a.tid)
                    since = self.ww_since.setdefault(a.tid, t)
                    if t - since >= self.cfg.events.ww_min_duration:
                        best = max(best, 0.25)
        self.ww_since = {k: v for k, v in self.ww_since.items() if k in flagged}
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
            self.min_px = self.cfg.events.col_min_size * self.work[0]
        if (frame.shape[1], frame.shape[0]) != self.work:  # same working pixels as Part A and the scene prior
            frame = cv2.resize(frame, self.work, interpolation=cv2.INTER_AREA)
        if not self.scene_checked:
            self._check_scene(frame)
        dets = self.detector([frame], imgsz=self.imgsz)[0]
        self.tracker.update(dets, t)
        W, H = self.work
        live = [x for x in self.tracker.tracks if x.state == "tracked" and x.t_seen >= t - 1e-6
                and x.group in ROAD_USERS and x.size >= self.min_px * size_factor(x.group)
                and x.box[0] > 3 and x.box[1] > 3 and x.box[2] < W - 3 and x.box[3] < H - 3  # cut-off boxes jitter
                and self._near_field(x, H)]
        for x in live:
            self.history.setdefault(x.tid, deque(maxlen=24)).append((t, x.speed))
        alive = {x.tid for x in self.tracker.tracks}
        for tid in [k for k in self.history if k not in alive]:
            del self.history[tid]

        conflict, pair, soft = self._conflict(live)
        cues = {"conflict": conflict, "braking": self._braking(live, t), "wrong_way": self._wrong_way(live, t),
                "impact": self._impact(live, t)}
        fused = 1.0 - float(np.prod([1.0 - c for c in cues.values()]))
        # an alarm needs the strong cues on consecutive analysed frames: single-frame spikes (a box
        # jumping, a pair briefly predicted to meet) stay just below the threshold but keep their rank
        self.streak = self.streak + 1 if fused >= 0.5 else 0
        if fused >= 0.5 and self.streak < self.cfg.risk.persist:
            fused = 0.49
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
