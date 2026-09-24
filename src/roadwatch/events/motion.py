"""wrong_way, illegal_u_turn, illegal_turn: single-track trajectory rules against the scene."""
from __future__ import annotations

import numpy as np

from ..config import EventCfg
from ..scene import angle_diff, movement_key, runs
from ..tracks import Track
from .base import Candidate, Context, mask_to_intervals


def _motorised(ctx: Context) -> list[Track]:
    return ctx.by_group("vehicle", "two_wheeler")


def detect_wrong_way(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    out = []
    scene = ctx.scene
    for tr in _motorised(ctx):
        lane, oriented = scene.lane_direction(tr.foot)
        valid = ~tr.edge & (tr.speed > cfg.ww_min_speed) & oriented & scene.on_road(tr.foot)
        against = valid & (angle_diff(tr.heading, lane) > cfg.ww_angle)
        for s, e in mask_to_intervals(tr.t, against, gap=1.0, min_len=cfg.ww_min_duration):
            w = tr.window(s, e)
            travelled = np.linalg.norm(tr.foot[w][-1] - tr.foot[w][0]) / np.median(tr.size[w])
            share = against[w].mean()
            if travelled < 1.5 or share < 0.6:
                continue
            if tr.end - e < 1.0:        # still against the flow when it disappears: it left the frame
                e = tr.end
            out.append(Candidate(s, e, "wrong_way", float(0.5 + 0.5 * share), (tr.tid,),
                                 {"travelled": float(travelled)}))
    return out


def _unwrapped_heading(tr: Track, min_speed: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    ok = (tr.speed > min_speed) & ~tr.edge
    idx = np.where(ok)[0]
    return idx, np.degrees(np.unwrap(np.radians(tr.heading[idx])))


def _turn_segments(tr: Track, min_turn: float, max_turn: float, max_duration: float) -> list[tuple[int, int, float]]:
    """(start_idx, end_idx, signed turn deg) of turning manoeuvres of the track."""
    idx, hd = _unwrapped_heading(tr)
    if len(idx) < 5:
        return []
    t = tr.t[idx]
    rate = np.gradient(hd, t)
    turning = np.abs(rate) > 8.0            # deg/s
    # a turn broken by a short straight stretch (noise) stays one manoeuvre if it keeps its sense
    merged: list[list[int]] = []
    for s, e in runs(turning):
        same_sense = merged and np.sign(hd[e] - hd[s]) == np.sign(hd[merged[-1][1]] - hd[merged[-1][0]])
        if same_sense and t[s] - t[merged[-1][1]] < 1.0:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        turn = hd[e] - hd[s]
        steps = np.abs(np.diff(hd[s:e + 1]))
        if min_turn <= abs(turn) <= max_turn and t[e] - t[s] <= max_duration and (steps.max(initial=0) < 60):
            out.append((int(idx[s]), int(idx[e]), float(turn)))
    return out


def detect_u_turns(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    out = []
    for tr in _motorised(ctx):
        for s, e, turn in _turn_segments(tr, cfg.ut_min_turn, 250.0, cfg.ut_max_duration):
            mid = tr.foot[(s + e) // 2]
            if ctx.scene.zones.get("u_turn_allowed") and ctx.scene.in_zones("u_turn_allowed", mid)[0]:
                continue
            if not ctx.scene.on_road(tr.foot[[s, e]]).any():
                continue  # manoeuvre in a car park or yard, not on the carriageway
            out.append(Candidate(float(tr.t[s]), float(tr.t[e]), "illegal_u_turn",
                                 float(np.clip(abs(turn) / 180, 0, 1)), (tr.tid,), {"turn": turn}))
    return out


def detect_illegal_turns(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    """Turns matching a drawn prohibited movement, or (with a large prior) movements never seen before."""
    out = []
    scene = ctx.scene
    rules = scene.zones.get("no_turn") or []
    total = sum(scene.movements.values())
    for tr in ctx.by_group("vehicle"):
        turns = _turn_segments(tr, 45.0, 140.0, 12.0)
        if not turns:
            continue
        for s, e, turn in turns:
            hit = None
            for rule in rules:
                a = scene.in_polygon(rule["from"], tr.foot[: s + 1])
                b = scene.in_polygon(rule["to"], tr.foot[e:])
                if a.any() and b.any():
                    hit = rule.get("name", "no_turn")
                    break
            if hit is None and total >= cfg.it_min_movements and not rules:
                key = movement_key(tr, ctx.width, ctx.height)
                if key is None:
                    continue
                share = scene.movements.get(key, 0) / total
                if share <= cfg.it_max_share and tr.duration > 4.0:
                    hit = f"rare movement {key} ({share:.3%})"
            if hit:
                out.append(Candidate(float(tr.t[s]), float(tr.t[e]), "illegal_turn", 0.6, (tr.tid,),
                                     {"rule": hit, "turn": turn}))
    return out


def detect(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    return detect_wrong_way(ctx, cfg) + detect_u_turns(ctx, cfg) + detect_illegal_turns(ctx, cfg)
