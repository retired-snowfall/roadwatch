"""jaywalking and failure_to_yield."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import EventCfg
from .base import Candidate, Context, mask_to_intervals


def detect_jaywalking(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    out = []
    scene = ctx.scene
    for tr in ctx.by_group("person"):
        # people walk on and beside the painted stripes: only clearly away from a crossing counts
        off_crossing = ~tr.edge & (scene.crossing_distance(tr.foot) > cfg.jw_crosswalk_margin * tr.size)
        on = off_crossing & scene.on_road(tr.foot)
        # boundaries come from the full carriageway mask, but the person must also get well
        # inside it (eroded mask) for a while: a curb-side wait does not count
        deep = off_crossing & scene.on_road(tr.foot, erode=cfg.jw_road_erode)
        # the event covers the walk onto and off the carriageway, not only the stretch off the stripes
        road_runs = mask_to_intervals(tr.t, scene.on_road(tr.foot) & ~tr.edge, gap=1.0, min_len=0.0)
        for s, e in mask_to_intervals(tr.t, on, gap=1.0, min_len=cfg.jw_min_duration):
            w = tr.window(s, e)
            if tr.t[w][deep[w]].size == 0 or np.ptp(tr.t[w][deep[w]]) < 0.5 * cfg.jw_min_duration:
                continue
            a, b = next(((a, b) for a, b in road_runs if a <= s and e <= b), (s, e))
            out.append(Candidate(max(a, s - cfg.jw_extend), min(b, e + cfg.jw_extend), "jaywalking", float(0.5 + 0.5 * deep[w].mean()), (tr.tid,)))
    return out


def _in_path(foot: np.ndarray, v: np.ndarray, size: float, p: np.ndarray, pv: np.ndarray, cfg: EventCfg) -> bool:
    """Is a pedestrian at p (velocity pv) walking across this vehicle's path, just ahead of or beside it?"""
    sp = float(np.hypot(*v))
    if sp < 1e-6:
        return False
    u = v / sp
    n = np.array([-u[1], u[0]])
    rel = p - foot
    ahead, lateral = float(rel @ u), abs(float(rel @ n))
    crossing = abs(float(pv @ n)) >= 0.5 * float(np.hypot(*pv))   # moving across the vehicle's direction
    return crossing and -0.5 * size <= ahead <= 2.0 * size and lateral <= cfg.fy_gap * size


def detect_failure_to_yield(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    scene = ctx.scene
    # frames where a pedestrian is walking across a crossing that lies on the carriageway
    # (people waiting at the kerb end of a crosswalk have not claimed it yet)
    ped_frames: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for tr in ctx.by_group("person"):
        on = (scene.crossing_distance(tr.foot) <= cfg.fy_crosswalk_margin * tr.size) & scene.on_road(tr.foot) \
            & (tr.speed >= cfg.fy_ped_speed)
        for f, p, v in zip(tr.fidx[on], tr.foot[on], tr.vel[on]):
            ped_frames[int(f)].append((p, v))
    if not ped_frames:
        return []
    out = []
    for tr in ctx.by_group("vehicle", "two_wheeler"):
        inside = scene.in_crossing(tr.foot) & ~tr.edge
        for s, e in mask_to_intervals(tr.t, inside, gap=0.5, min_len=0.2):
            w = tr.window(s, e)
            if np.median(tr.speed[w]) < cfg.fy_min_speed or tr.speed[w].min() < 0.15:
                continue  # it slowed or stopped for the crossing
            near = 0
            for f, foot, size, v in zip(tr.fidx[w], tr.foot[w], tr.size[w], tr.vel[w]):
                if any(_in_path(foot, v, size, p, pv, cfg) for p, pv in ped_frames.get(int(f), ())):
                    near += 1
            if near < 2:
                continue  # it must pass through the path of a crossing pedestrian, not just share a crosswalk
            e_out = e
            after = np.where((tr.t > e) & ~inside)[0]
            if len(after):
                e_out = float(tr.t[after[0]])
            # the event includes the approach to the crossing, not only the moment on the stripes
            out.append(Candidate(s - cfg.fy_lead, max(e_out, s + 0.5) + cfg.fy_tail, "failure_to_yield",
                                 float(0.5 + 0.5 * near / max(1, len(tr.fidx[w]))), (tr.tid,)))
    return out


def detect(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    return detect_jaywalking(ctx, cfg) + detect_failure_to_yield(ctx, cfg)
