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
        off_crossing = ~tr.edge & ~scene.in_crossing(tr.foot)
        on = off_crossing & scene.on_road(tr.foot)
        # boundaries come from the full carriageway mask, but the person must also get well
        # inside it (eroded mask) for a while: a curb-side wait does not count
        deep = off_crossing & scene.on_road(tr.foot, erode=cfg.jw_road_erode)
        for s, e in mask_to_intervals(tr.t, on, gap=1.0, min_len=cfg.jw_min_duration):
            w = tr.window(s, e)
            if tr.t[w][deep[w]].size == 0 or np.ptp(tr.t[w][deep[w]]) < 0.5 * cfg.jw_min_duration:
                continue
            out.append(Candidate(s, e, "jaywalking", float(0.5 + 0.5 * deep[w].mean()), (tr.tid,)))
    return out


def detect_failure_to_yield(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    scene = ctx.scene
    # frames where a pedestrian stands on a crossing that lies on the carriageway
    ped_frames: dict[int, list[np.ndarray]] = defaultdict(list)
    for tr in ctx.by_group("person"):
        on = scene.in_crossing(tr.foot) & scene.on_road(tr.foot)
        for f, p in zip(tr.fidx[on], tr.foot[on]):
            ped_frames[int(f)].append(p)
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
            for f, foot, size in zip(tr.fidx[w], tr.foot[w], tr.size[w]):
                if any(np.linalg.norm(p - foot) < 5.0 * size for p in ped_frames.get(int(f), ())):
                    near += 1
            if near == 0:
                continue
            e_out = e
            after = np.where((tr.t > e) & ~inside)[0]
            if len(after):
                e_out = float(tr.t[after[0]])
            out.append(Candidate(s, max(e_out, s + 0.5), "failure_to_yield",
                                 float(0.5 + 0.5 * near / max(1, len(tr.fidx[w]))), (tr.tid,)))
    return out


def detect(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    return detect_jaywalking(ctx, cfg) + detect_failure_to_yield(ctx, cfg)
