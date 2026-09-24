"""Part A pipeline: video -> detections -> tracks -> scene -> rule candidates -> events."""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .appearance import AppearanceMonitor
from .config import CFG, WEIGHTS_DIR, Config
from .constants import COCO_GROUPS, TRAFFIC_LIGHT
from .detector import device_kind, get_detector
from .events import Candidate, Context, finalize, run_all
from .scene import SceneModel
from .signals import cluster_lights, light_color
from .tracker import ByteTracker
from .tracks import FrameIndex, Track, build_tracks
from .video import FrameSampler, VideoInfo, iter_frames, probe

log = logging.getLogger("roadwatch")
Progress = Callable[[str, float], None]


@dataclass
class Perception:
    """Everything read from the pixels in one pass; cacheable."""
    records: list
    light_obs: list
    appearance: AppearanceMonitor
    analysed: int
    seconds: float
    device: str


@dataclass
class Analysis:
    info: VideoInfo
    tracks: list[Track]
    scene: SceneModel
    candidates: list[Candidate]
    events: list[list]
    context: Context
    timings: dict = field(default_factory=dict)
    background: np.ndarray | None = None
    used_prior: bool = False
    appearance: AppearanceMonitor | None = None

    def to_json(self, max_track_points: int = 400) -> dict:
        """Compact description for the website: events with evidence, tracks for overlays, scene layers."""
        sc = self.scene
        d = sc.derived
        tracks = []
        for tr in self.tracks:
            stride = max(1, len(tr.t) // max_track_points)
            tracks.append(tr.to_json(stride))
        return {
            "video": self.info.name, "fps": self.info.fps, "width": self.info.width, "height": self.info.height,
            "duration": round(self.info.duration, 3), "n_frames": self.info.n_frames,
            "events": self.events, "candidates": [c.as_json() for c in self.candidates],
            "tracks": tracks, "timings": self.timings, "used_prior": self.used_prior,
            "scene": {"grid": [sc.cfg.grid_w, sc.cfg.grid_h], "road": d["road"].astype(int).tolist(),
                      "direction": np.round(d["direction"]).astype(int).tolist(),
                      "oriented": d["oriented"].astype(int).tolist(),
                      "stop_lines": [{"a": sl["a"].round(1).tolist(), "b": sl["b"].round(1).tolist(),
                                      "dir": round(sl["dir"], 1)} for sl in sc.stop_lines()],
                      "zones": sc.zones},
            "signals": {str(k): v for k, v in self.context.signals.items()},
        }


# ---------------------------------------------------------------------- perception pass
def _cache_file(info: VideoInfo, kind: str) -> Path | None:
    root = os.environ.get("ROADWATCH_CACHE")
    if not root:
        return None
    st = Path(info.path).stat()
    key = hashlib.sha1(f"{Path(info.path).resolve()}|{st.st_size}|{st.st_mtime_ns}|{kind}|v1".encode()).hexdigest()
    return Path(root) / f"{Path(info.path).stem}-{key[:12]}.pkl"


def perceive(info: VideoInfo, cfg: Config = CFG, progress: Progress | None = None) -> Perception:
    kind = device_kind()
    cache = _cache_file(info, kind)
    if cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    det = get_detector(kind)
    pc = cfg.perception
    sampler = FrameSampler(info.fps, pc.analysis_fps_gpu if kind == "gpu" else pc.analysis_fps_cpu)
    tracker = ByteTracker(cfg.tracker)
    appearance = AppearanceMonitor(info.width, info.height)
    records: list = []
    light_obs: list = []
    t0 = time.perf_counter()
    analysed = 0
    batch: list = []

    def flush() -> None:
        nonlocal analysed
        if not batch:
            return
        dets = det([b[2] for b in batch])
        for (idx, t, frame), d in zip(batch, dets):
            for x1, y1, x2, y2, conf, c in d[(d[:, 5] == TRAFFIC_LIGHT) & (d[:, 4] >= 0.3)]:
                crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                light_obs.append((t, np.array([x1, y1, x2, y2]), light_color(crop)))
            for trk in tracker.update(d, t):
                records.append((idx, t, trk.tid, *trk.box.tolist(), trk.conf, trk.cls, trk.group))
            if appearance.wants(t):
                keep = np.isin(d[:, 5].astype(int), list(COCO_GROUPS)) & (d[:, 4] >= 0.25)
                appearance.add(t, frame, d[keep, :4])
            analysed += 1
        # time guard: if detection runs far slower than planned, analyse fewer frames
        processed = batch[-1][1] + 1.0 / info.fps
        elapsed = time.perf_counter() - t0
        if processed > 20 and elapsed > pc.budget_share * processed:
            sampler.set_rate(max(2.0, info.fps / sampler.step * 0.75))
            log.warning("%s: slow perception (%.1fs for %.1fs of video); rate -> %.1f fps",
                        info.name, elapsed, processed, info.fps / sampler.step)
        if progress:
            progress("detect", min(1.0, processed / max(info.duration, 1e-6)))
        batch.clear()

    for idx, t, frame in iter_frames(info.path, sampler):
        batch.append((idx, t, frame))
        if len(batch) >= det.profile.batch:
            flush()
    flush()
    out = Perception(records, light_obs, appearance, analysed, time.perf_counter() - t0, kind)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(out, f)
    return out


# ---------------------------------------------------------------------- scene fusion
def load_prior() -> SceneModel | None:
    return SceneModel.load(WEIGHTS_DIR / "scene.json")


def build_scene(info: VideoInfo, tracks: list[Track], background: np.ndarray | None, cfg: Config,
                prior: SceneModel | None) -> tuple[SceneModel, bool]:
    current = SceneModel(info.width, info.height, cfg.scene)
    current.accumulate(tracks, info.duration, cfg.kin, background)
    if prior is not None and current.similar_to(prior):
        return prior.merged_with(current), True
    return current, False


# ---------------------------------------------------------------------- analysis
def analyze(video_path: str, cfg: Config = CFG, prior: SceneModel | None | bool = True,
            progress: Progress | None = None) -> Analysis:
    """Full Part A analysis. prior=True loads weights/scene.json; False/None disables it."""
    t_start = time.perf_counter()
    info = probe(video_path)
    per = perceive(info, cfg, progress)
    t_perc = time.perf_counter() - t_start

    if progress:
        progress("tracks", 0.0)
    tracks = build_tracks(per.records, cfg.kin, info.width, info.height)
    background = per.appearance.background()
    if background is not None:
        background = cv2.resize(background, (info.width, info.height))
    if prior is True:
        prior = load_prior()
    scene, used_prior = build_scene(info, tracks, background, cfg, prior or None)

    n_idx = max([int(r[0]) for r in per.records], default=0) + 1
    frame_times = np.arange(max(n_idx, info.n_frames) + 1) / info.fps
    ctx = Context(tracks, scene, info.duration, info.fps, info.width, info.height, FrameIndex(tracks), frame_times)
    road_work = cv2.resize(scene.derived["road"].astype(np.uint8), per.appearance.work_size,
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    ctx.extras = {
        "lights": cluster_lights(per.light_obs),
        "static_blobs": per.appearance.static_blobs(road_work, cfg.events.ob_min_area),
        "fire": per.appearance.fire_series(_moving_boxes(tracks, per.appearance.times), road_work),
        "smoke": per.appearance.smoke_series(road_work),
    }
    if progress:
        progress("rules", 0.0)
    candidates = run_all(ctx, cfg)
    events = finalize(candidates, info.duration, cfg.events.merge_gap, cfg.events.min_duration, cfg.enabled)
    timings = {"perception_sec": round(t_perc, 2), "total_sec": round(time.perf_counter() - t_start, 2),
               "analysed_frames": per.analysed, "device": per.device,
               "realtime_factor": round((time.perf_counter() - t_start) / max(info.duration, 1e-6), 3)}
    log.info("%s: %d tracks, %d candidates, %d events, %s", info.name, len(tracks), len(candidates),
             len(events), timings)
    if progress:
        progress("done", 1.0)
    return Analysis(info, tracks, scene, candidates, events, ctx, timings, background, used_prior, per.appearance)


def _moving_boxes(tracks: list[Track], times: list[float]) -> dict[int, list[np.ndarray]]:
    """Boxes of moving vehicles at each appearance sample (their tail lights are not fire)."""
    lookup = {round(t, 3): k for k, t in enumerate(times)}
    out: dict[int, list[np.ndarray]] = {}
    for tr in tracks:
        if tr.group not in ("vehicle", "two_wheeler"):
            continue
        for t, b, v in zip(tr.t, tr.box, tr.speed):
            k = lookup.get(round(float(t), 3))
            if k is not None and v > 0.3:
                out.setdefault(k, []).append(b)
    return out


def detect_events(video_path: str) -> list[list]:
    return analyze(video_path).events
