"""red_light, stop_line and solid_line_crossing: rules on road markings and signal state."""
from __future__ import annotations

import numpy as np

from ..config import EventCfg, KinematicsCfg
from ..scene import runs
from ..signals import (StopLine, approaching, assign_light, front_points, red_intervals,
                       waiting_intervals)
from .base import Candidate, Context


def _stop_lines(ctx: Context) -> list[StopLine]:
    return [StopLine(np.asarray(sl["a"]), np.asarray(sl["b"]), sl["dir"]) for sl in ctx.scene.stop_lines()]


def _exit_time(ctx: Context, tr, k_cross: int, line: StopLine) -> float:
    """When the vehicle leaves the intersection (drawn polygon) or has driven well past the line."""
    zone = ctx.scene.zones.get("intersection")
    if zone:
        inside = ctx.scene.in_zones("intersection", tr.foot[k_cross:])
        entered = np.where(inside)[0]
        if len(entered):
            left = np.where(~inside[entered[0]:])[0]
            if len(left):
                return float(tr.t[k_cross + entered[0] + left[0]])
        return tr.end
    past = line.along(tr.foot[k_cross:]) > 6.0 * tr.size[k_cross:]
    k = np.where(past)[0]
    return float(tr.t[k_cross + k[0]]) if len(k) else tr.end


def detect_signal_events(ctx: Context, cfg: EventCfg, kin: KinematicsCfg) -> list[Candidate]:
    lights = ctx.extras.get("lights", [])
    vehicles = ctx.by_group("vehicle", "two_wheeler")
    out = []
    for li, line in enumerate(_stop_lines(ctx)):
        waits = waiting_intervals(ctx.tracks, line, kin.stationary_speed)
        light = assign_light(waits, lights, ctx.duration)
        reds = red_intervals(waits, light)
        ctx.signals[li] = {"red": reds, "light": None if light is None else light.box.tolist()}
        if not reds:
            continue
        for tr in vehicles:
            fr = front_points(tr, line)
            al, ac = line.along(fr), line.across(fr)
            lateral = np.abs(ac) < line.half + 0.3 * tr.size
            ok = lateral & approaching(tr, line, 50.0) & ~tr.edge
            # --- red_light: the front crosses the line while moving, inside a red period
            cross = np.where(ok[1:] & (al[:-1] < 0) & (al[1:] >= 0))[0] + 1
            for k in cross:
                tx = float(tr.t[k])
                if tr.speed[k] < cfg.rl_min_speed:
                    continue
                light_red = light.red_at(tx) if light is not None else None
                others = [(s, e, tid) for s, e, tid in waits if tid != tr.tid and s + 1.0 <= tx <= e - 1.0]
                if light_red is False or (light_red is None and not others):
                    continue
                exit_t = _exit_time(ctx, tr, k, line)
                out.append(Candidate(tx, max(exit_t, tx + 0.5), "red_light", 0.9 if light_red else 0.7,
                                     (tr.tid,), {"line": li, "waiting": len(others), "light": light_red}))
            # --- stop_line: stationary with the front past the line, not in the intersection, on red
            past = ok & (al > 0.3 * tr.size) & (al < 2.5 * tr.size) & (tr.speed < kin.stationary_speed)
            for s, e in runs(past):
                ts, te = float(tr.t[s]), float(tr.t[e])
                if te - ts < cfg.sl_min_stop:
                    continue
                red = [(a, b) for a, b in reds if a <= ts + 1.0 and b >= ts]
                if not red and not (light is not None and light.red_at(ts)):
                    continue
                green = min([b for _, b in red] + [te]) if red else te
                out.append(Candidate(ts, max(green, ts + 1.0), "stop_line", 0.7, (tr.tid,), {"line": li}))
    return out


def _signed_distance(poly: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of points to a polyline and whether the projection falls inside it."""
    best = np.full(len(pts), np.inf)
    sign = np.zeros(len(pts))
    inside = np.zeros(len(pts), bool)
    for p, q in zip(poly[:-1], poly[1:]):
        d = q - p
        L2 = max(d @ d, 1e-9)
        tpar = ((pts - p) @ d) / L2
        proj = p + np.clip(tpar, 0, 1)[:, None] * d
        dist = np.linalg.norm(pts - proj, axis=1)
        closer = dist < best
        best = np.where(closer, dist, best)
        cross = d[0] * (pts[:, 1] - p[1]) - d[1] * (pts[:, 0] - p[0])
        sign = np.where(closer, np.sign(cross), sign)
        inside = np.where(closer, (tpar > 0.02) & (tpar < 0.98), inside)
    return best * sign, inside


def detect_solid_line_crossings(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    lines = [ctx.scene.denorm(sl["points"] if isinstance(sl, dict) else sl)
             for sl in ctx.scene.zones.get("solid_lines", [])]
    if not lines:
        return []
    out = []
    for tr in ctx.by_group("vehicle", "two_wheeler"):
        left = np.c_[tr.box[:, 0], tr.box[:, 3]]
        right = np.c_[tr.box[:, 2], tr.box[:, 3]]
        moving = (tr.speed > 0.3) & ~tr.edge
        for li, poly in enumerate(lines):
            dl, in_l = _signed_distance(poly, left)
            dr, in_r = _signed_distance(poly, right)
            valid = moving & in_l & in_r
            side = np.where(valid, np.where((dl > 0) & (dr > 0), 1, np.where((dl < 0) & (dr < 0), -1, 0)), 99)
            clean = np.where(np.isin(side, (-1, 1)))[0]
            for k0, k1 in zip(clean[:-1], clean[1:]):
                if side[k0] == side[k1] or tr.t[k1] - tr.t[k0] > 6.0:
                    continue
                # straddling samples in between: the wheel crosses at the first, fully over after the last
                mid = np.arange(k0 + 1, k1)
                t_start = float(tr.t[mid[0]]) if len(mid) else float(tr.t[k0])
                t_end = float(tr.t[k1])
                out.append(Candidate(t_start, max(t_end, t_start + 0.4), "solid_line_crossing", 0.7,
                                     (tr.tid,), {"line": li}))
    return out


def detect(ctx: Context, cfg: EventCfg, kin: KinematicsCfg) -> list[Candidate]:
    return detect_signal_events(ctx, cfg, kin) + detect_solid_line_crossings(ctx, cfg)
