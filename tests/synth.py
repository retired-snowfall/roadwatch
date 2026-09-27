"""Synthetic trajectories and scenes for rule tests (no video, no detector)."""
from __future__ import annotations

import numpy as np

from roadwatch.config import CFG
from roadwatch.events.base import Context
from roadwatch.scene import SceneModel
from roadwatch.tracks import FrameIndex, Track

W, H, FPS, RATE = 1920, 1080, 25.0, 5.0     # video size, video fps, analysis rate


def make_track(tid: int, points: list[tuple[float, float, float]], group: str = "vehicle",
               size: tuple[float, float] = (120, 80), cls: int = 2) -> Track:
    """points: (t, x_foot, y_foot) key frames; linearly interpolated at the analysis rate."""
    key = np.array(points, float)
    t = np.arange(key[0, 0], key[-1, 0] + 1e-9, 1.0 / RATE)
    x = np.interp(t, key[:, 0], key[:, 1])
    y = np.interp(t, key[:, 0], key[:, 2])
    w, h = size
    box = np.c_[x - w / 2, y - h, x + w / 2, y]
    fidx = np.round(t * FPS).astype(int)
    tr = Track(tid, group, cls, fidx, t, box, np.full(len(t), 0.9))
    tr.compute_kinematics(CFG.kin, W, H)
    return tr


def two_way_road_scene(n_per_lane: int = 30) -> SceneModel:
    """Horizontal two-way road: y in [420, 530] drives right (+x), y in [560, 670] drives left."""
    tracks = []
    tid = 1000
    for k in range(n_per_lane):
        t0 = k * 4.0
        for y in (430, 460, 490, 520):
            tracks.append(make_track(tid, [(t0, 50, y), (t0 + 6, 1870, y)]))
            tid += 1
        for y in (570, 600, 630, 660):
            tracks.append(make_track(tid, [(t0, 1870, y), (t0 + 6, 50, y)]))
            tid += 1
    scene = SceneModel(W, H, CFG.scene)
    scene.accumulate(tracks, n_per_lane * 4.0, CFG.kin)
    return scene


def context(tracks: list[Track], scene: SceneModel, duration: float = 60.0) -> Context:
    frame_times = np.arange(int(duration * FPS) + 2) / FPS
    return Context(tracks, scene, duration, FPS, W, H, FrameIndex(tracks), frame_times)
