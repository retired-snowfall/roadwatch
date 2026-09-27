"""stopped_vehicle and congestion."""
from __future__ import annotations

import numpy as np

from ..config import EventCfg, KinematicsCfg
from ..tracks import Track
from .base import Candidate, Context, mask_to_intervals


def _near_stop_line(ctx: Context, foot: np.ndarray, size: float) -> bool:
    for sl in ctx.scene.stop_lines():
        u = np.array([np.cos(np.radians(sl["dir"])), np.sin(np.radians(sl["dir"]))])
        n = np.array([-u[1], u[0]])
        mid = (sl["a"] + sl["b"]) / 2
        half = np.linalg.norm(sl["b"] - sl["a"]) / 2
        rel = foot - mid
        if -8 * size <= rel @ u <= 1.5 * size and abs(rel @ n) <= half + size:
            return True
    return False


def _queued_share(ctx: Context, tr: Track, s: float, e: float, kin: KinematicsCfg) -> float:
    """Share of the stop during which another stationary vehicle sits right ahead or behind in the lane."""
    w = tr.window(s, e)
    lane, oriented = ctx.scene.lane_direction(tr.foot[w])
    hits = 0
    samples = list(zip(tr.fidx[w], tr.foot[w], tr.size[w], lane, oriented))
    for f, foot, size, d, ok in samples:
        u = np.array([np.cos(np.radians(d)), np.sin(np.radians(d))]) if ok else None
        for ti, si in ctx.index.by_frame.get(int(f), ()):
            other = ctx.tracks[ti]
            if other.tid == tr.tid or other.group != "vehicle" or other.speed[si] > kin.stationary_speed * 2:
                continue
            rel = other.foot[si] - foot
            dist = np.linalg.norm(rel) / size
            if u is not None:
                along, across = rel @ u / size, abs(rel @ np.array([-u[1], u[0]])) / size
                if abs(along) < 3.0 and across < 0.8:
                    hits += 1
                    break
            elif dist < 2.0:
                hits += 1
                break
    return hits / max(len(samples), 1)


def _no_stopping_stays(ctx: Context, cfg: EventCfg, kin: KinematicsCfg) -> list[Candidate]:
    """Vehicles standing in a drawn no-stopping zone (weights/zones.json), however routine the spot. A parked
    car keeps its place while passing traffic hides it, and comes back under a new track id: stays at the
    same spot are joined across such gaps. Boxes cut off by the frame border count: a parked car does not
    move, so a partial box is still the same car."""
    if not ctx.scene.zones.get("no_stopping"):
        return []
    stays = []
    for tr in ctx.by_group("vehicle"):
        still = tr.speed < kin.stationary_speed
        for s, e in mask_to_intervals(tr.t, still, gap=2.0, min_len=2.0):
            w = tr.window(s, e)
            foot = np.median(tr.foot[w], axis=0)
            if ctx.scene.in_zones("no_stopping", foot)[0]:
                stays.append((s, e, foot, float(np.median(tr.size[w])), tr.tid))
    chains: list[list] = []
    for s, e, foot, size, tid in sorted(stays, key=lambda x: x[0]):
        for ch in chains:
            if s <= ch[1] + cfg.sv_bridge_gap and np.linalg.norm(foot - ch[2]) <= 0.5 * ch[3]:
                ch[1] = max(ch[1], e)
                ch[4].append(tid)
                break
        else:
            chains.append([s, e, foot, size, [tid]])
    out = []
    for s, e, _, _, tids in chains:
        if e - s < cfg.sv_min_duration:
            continue
        s = 0.0 if s <= 1.0 else s                            # already there when the video starts
        e = ctx.duration if e >= ctx.duration - 1.0 else e    # still there when it ends
        out.append(Candidate(s, e, "stopped_vehicle", 1.0, tuple(tids), {"zone": "no_stopping"}))
    return out


def detect_stopped(ctx: Context, cfg: EventCfg, kin: KinematicsCfg, congestion: list[Candidate]) -> list[Candidate]:
    out = _no_stopping_stays(ctx, cfg, kin)
    scene = ctx.scene
    for tr in ctx.by_group("vehicle"):
        still = (tr.speed < kin.stationary_speed) & ~tr.edge
        for s, e in mask_to_intervals(tr.t, still, gap=2.0, min_len=cfg.sv_min_duration):
            w = tr.window(s, e)
            foot = np.median(tr.foot[w], axis=0)
            size = float(np.median(tr.size[w]))
            if not scene.on_road(foot)[0]:
                continue
            gy, gx = scene.cells(foot)
            if scene.derived["stop_norm"][gy[0], gx[0]] > 1.0 and scene.n_videos > 1:
                continue  # a place where vehicles routinely park or wait (learned on the samples)
            need = cfg.sv_min_duration
            if _near_stop_line(ctx, foot, size):
                need = cfg.sv_signal_zone_duration
            if any(c.start - 5 <= s and e <= c.end + 5 for c in congestion):
                continue
            if e - s < need:
                continue
            queued = _queued_share(ctx, tr, s, e, kin)
            if queued > 0.5:
                continue
            if tr.end - e < 1.0 and tr.end < ctx.duration - 1.0:
                e = tr.end          # track ends while still stopped: removed / occluded until the end
            elif tr.end >= ctx.duration - 1.0 and still[-3:].all():
                e = ctx.duration    # still there when the video ends
            out.append(Candidate(s, e, "stopped_vehicle", float(1.0 - 0.5 * queued), (tr.tid,),
                                 {"queued": queued}))
    return out


def detect_congestion(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    streams = ctx.scene.derived["streams"]
    n_streams = int(streams.max()) + 1 if streams.size else 0
    if n_streams == 0 or len(ctx.frame_times) == 0:
        return []
    frames = ctx.index.frames
    counts = np.zeros((n_streams, len(frames)))
    slow = np.zeros((n_streams, len(frames)))
    for k, f in enumerate(frames):
        items = [(ti, si) for ti, si in ctx.index.by_frame[int(f)] if ctx.tracks[ti].group == "vehicle"]
        if not items:
            continue
        feet = np.array([ctx.tracks[ti].foot[si] for ti, si in items])
        sp = np.array([ctx.tracks[ti].speed[si] for ti, si in items])
        sid = ctx.scene.stream_of(feet)
        for sd, v in zip(sid, sp):
            if sd >= 0:
                counts[sd, k] += 1
                slow[sd, k] += v < cfg.cg_slow_speed
    t = ctx.frame_times[frames]
    out = []
    for sd in range(n_streams):
        c = counts[sd]
        cap = np.percentile(c, 95) if c.any() else 0
        need = max(cfg.cg_min_vehicles, 0.5 * cap)
        share = np.where(c > 0, slow[sd] / np.maximum(c, 1), 0)
        jam = (c >= need) & (share >= cfg.cg_slow_share)
        # smooth over ~3 s so single frames of a moving car do not break a standstill
        kernel = max(1, int(round(3.0 / max(np.median(np.diff(t)) if len(t) > 1 else 1.0, 1e-3))))
        jam = np.convolve(jam.astype(float), np.ones(kernel) / kernel, mode="same") >= 0.6
        for s, e in mask_to_intervals(t, jam, gap=5.0, min_len=cfg.cg_min_duration):
            out.append(Candidate(s, e, "congestion", 0.8, (), {"stream": sd, "vehicles": float(need)}))
    return out


def detect(ctx: Context, cfg: EventCfg, kin: KinematicsCfg) -> list[Candidate]:
    congestion = detect_congestion(ctx, cfg)
    return congestion + detect_stopped(ctx, cfg, kin, congestion)
