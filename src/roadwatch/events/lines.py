"""red_light, stop_line and solid_line_crossing: rules on road markings and signal state."""
from __future__ import annotations

import numpy as np

from ..config import EventCfg, KinematicsCfg
from ..scene import runs
from ..signals import (StopLine, approaching, assign_light, front_points, red_intervals,
                       waiting_intervals)
from .base import Candidate, Context, trustworthy


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
        geo = []
        for tr in vehicles:
            fr = front_points(tr, line)
            al, ac = line.along(fr), line.across(fr)
            lateral = np.abs(ac) < line.half + 0.3 * tr.size
            ok = lateral & approaching(tr, line, 50.0) & ~tr.edge
            cross = np.where(ok[1:] & (al[:-1] < 0) & (al[1:] >= 0))[0] + 1
            geo.append((tr, al, ok, cross))
        # every moving crossing of this line: a steady stream of them means the signal is green
        crossings = np.array(sorted(float(tr.t[k]) for tr, _, _, cross in geo for k in cross
                                    if tr.speed[k] >= cfg.rl_min_speed))
        for tr, al, ok, cross in geo:
            # --- red_light: the front crosses the line while moving, inside a red period
            for k in cross:
                tx = float(tr.t[k])
                if tr.speed[k] < cfg.rl_min_speed:
                    continue
                light_red = light.red_at(tx) if light is not None else None
                others = [(s, e, tid) for s, e, tid in waits if tid != tr.tid and s + 1.0 <= tx <= e - 1.0]
                stream = int(((crossings >= tx - cfg.rl_stream_window) & (crossings <= tx + cfg.rl_stream_window)).sum())
                if light_red is False or stream > cfg.rl_max_stream:
                    continue
                if light_red is None and len({tid for _, _, tid in others}) < cfg.rl_min_waiting:
                    continue  # without a visible light, one waiting car (e.g. turning) is not proof of red
                exit_t = _exit_time(ctx, tr, k, line)
                out.append(Candidate(tx, max(exit_t, tx + 0.5), "red_light", 0.9 if light_red else 0.7,
                                     (tr.tid,), {"line": li, "waiting": len(others), "light": light_red}))
            # --- stop_line: stationary with the front past the line, not in the intersection, on red
            past = ok & (al > 0.3 * tr.size) & (al < 2.5 * tr.size) & (tr.speed < kin.stationary_speed)
            reds_others = red_intervals([w for w in waits if w[2] != tr.tid], light)
            for s, e in runs(past):
                ts, te = float(tr.t[s]), float(tr.t[e])
                if te - ts < cfg.sl_min_stop or not tr.arrived_moving(s):
                    continue  # too short, or a parked car that happens to stand there
                red = [(a, b) for a, b in reds_others if a <= ts + 1.0 and b >= ts]
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
    """A vehicle whose ground point moves from one side of a drawn solid line to the other. In this oblique
    view a box's bottom corners straddle the neighbouring lane lines even when the car drives straight, so
    the box centre is used, with a dead band around the line (a share of the box width) against jitter.
    A vehicle that stops astride the line (caught by the signal mid-manoeuvre) is in violation for as long
    as it stands there."""
    lines = [ctx.scene.denorm(sl["points"] if isinstance(sl, dict) else sl)
             for sl in ctx.scene.zones.get("solid_lines", [])]
    if not lines:
        return []
    out = []
    for tr in ctx.by_group("vehicle", "two_wheeler"):
        width = tr.box[:, 2] - tr.box[:, 0]
        ok = ~tr.edge
        for li, poly in enumerate(lines):
            d, along = _signed_distance(poly, tr.foot)
            band = cfg.slc_band * width
            side = np.where(ok & along & (d > band), 1, np.where(ok & along & (d < -band), -1, 0))
            clean = np.where(side != 0)[0]
            for k0, k1 in zip(clean[:-1], clean[1:]):
                if side[k0] == side[k1] or tr.t[k1] - tr.t[k0] > cfg.slc_max_gap:
                    continue
                if not trustworthy(ctx, tr, k0, k1, cfg):
                    continue  # jittering far-away boxes and identity switches in queues
                t0, t1 = float(tr.t[k0]), float(tr.t[k1])
                mid = (t0 + t1) / 2
                s, e = min(t0, mid - cfg.slc_min_len / 2), max(t1, mid + cfg.slc_min_len / 2)
                astride = (tr.speed[k0:k1 + 1] < 0.12).mean() if k1 > k0 else 0.0
                out.append(Candidate(s, e, "solid_line_crossing", 0.7, (tr.tid,),
                                     {"line": li, "stopped_astride": float(astride)}))
    return out


def detect_crosswalk_blocking(ctx: Context, cfg: EventCfg, kin: KinematicsCfg) -> list[Candidate]:
    """stop_line: vehicles that drove in and stopped on a pedestrian crossing, i.e. past the stop line
    in front of it (typically stuck in the junction behind a queue). The event lasts while any of them
    stands there. Crossings are the drawn crosswalks that lie on the carriageway."""
    scene = ctx.scene
    polys = [p for p in scene.zones.get("crosswalks", [])
             if scene.on_road(np.asarray(scene.denorm(p["points"] if isinstance(p, dict) else p)).mean(0))[0]]
    if not polys:
        return []
    t = np.arange(0.0, ctx.duration, 0.5)
    count = np.zeros(len(t))
    tids: set = set()
    for tr in ctx.by_group("vehicle"):
        on = np.zeros(len(tr.t), bool)
        for p in polys:
            on |= scene.in_polygon(p, tr.foot)
        still = on & (tr.speed < kin.stationary_speed) & ~tr.edge
        for a, b in runs(still):
            # it drove in at some point (in a jam the last metres are a crawl); parked cars never did
            if tr.t[b] - tr.t[a] < cfg.sl_min_stop or tr.travelled(tr.start, float(tr.t[a])) < 1.5:
                continue
            count += (t >= tr.t[a]) & (t <= tr.t[b])
            tids.add(tr.tid)
    out = []
    for a, b in runs(count >= cfg.sl_block_min_vehicles):
        s, e = float(t[a]), float(t[b])
        if out and s - out[-1][1] <= cfg.sl_block_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [Candidate(s, e, "stop_line", 0.6, tuple(sorted(tids)), {"blocking": True})
            for s, e in out if e - s >= cfg.sl_block_min_duration]


def detect(ctx: Context, cfg: EventCfg, kin: KinematicsCfg) -> list[Candidate]:
    return (detect_signal_events(ctx, cfg, kin) + detect_crosswalk_blocking(ctx, cfg, kin)
            + detect_solid_line_crossings(ctx, cfg))
