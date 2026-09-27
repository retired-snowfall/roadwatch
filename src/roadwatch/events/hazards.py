"""road_obstacle and fire_smoke."""
from __future__ import annotations

import numpy as np

from ..config import EventCfg
from .base import Candidate, Context, mask_to_intervals


def detect_obstacles(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    out = []
    for tr in ctx.by_group("animal"):
        on = ctx.scene.on_road(tr.foot) & ~tr.edge
        for s, e in mask_to_intervals(tr.t, on, gap=2.0, min_len=1.0):
            out.append(Candidate(s, e, "road_obstacle", 0.7, (tr.tid,), {"kind": tr.label}))
    # static foreground blobs: on a busy junction these are nearly always queued vehicles the detector
    # missed, so they only count when enabled (cfg.ob_static_blobs)
    for bl in ctx.extras.get("static_blobs", []) if cfg.ob_static_blobs else []:
        if bl.t1 - bl.t0 < cfg.ob_min_duration or bl.hits < 4:
            continue
        centre = np.array([(bl.box[0] + bl.box[2]) / 2, bl.box[3]])
        if not ctx.scene.on_road(centre, erode=1)[0]:
            continue
        end = bl.t1 if bl.t1 < ctx.duration - 2.0 else ctx.duration
        out.append(Candidate(bl.t0, end, "road_obstacle", 0.6, (), {"kind": "static object",
                                                                    "box": [round(v) for v in bl.box]}))
    return out


def _series_events(series: list[tuple[float, float]], label: str, min_len: float, duration: float,
                   kind: str) -> list[Candidate]:
    if not series:
        return []
    t = np.array([s[0] for s in series])
    on = np.array([s[1] > 0 for s in series])
    out = []
    for s, e in mask_to_intervals(t, on, gap=2.0, min_len=min_len):
        if e >= t[-1] - 1e-6:
            e = duration  # still visible when the video ends
        out.append(Candidate(s, e, label, 0.6, (), {"kind": kind}))
    return out


def detect_fire_smoke(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    fire = _series_events(ctx.extras.get("fire", []), "fire_smoke", cfg.fire_min_duration, ctx.duration, "fire")
    smoke = _series_events(ctx.extras.get("smoke", []), "fire_smoke", 2 * cfg.fire_min_duration + 2.0,
                           ctx.duration, "smoke")
    return fire + smoke


def detect(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    return detect_obstacles(ctx, cfg) + detect_fire_smoke(ctx, cfg)
