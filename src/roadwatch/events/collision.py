"""accident and near_miss: pairwise interaction analysis of road users.

For every analysed frame the pairs of nearby road users are measured (ground-point
distance in object sizes, box overlap, time and distance of closest approach under
constant velocity). Each pair's series is then searched for two signatures:

accident   contact (feet within ~1 size, boxes touching) where at least one party
           was moving, followed within ~1 s by an abrupt speed drop or deflection
           (not a gradual queue stop) and then by both parties coming to rest near
           each other or leaving. Pedestrian falls (box aspect flip) count as impact.
near_miss  a predicted conflict (closest approach < gate within nm_ttc seconds) met
           by hard braking or a swerve, with no contact, after which the pair clears.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import EventCfg
from ..constants import MOTORISED, ROAD_USERS
from ..scene import angle_diff
from ..tracks import Track
from .base import Candidate, Context


def pair_series(ctx: Context, min_size: float, max_dist: float = 4.0, horizon: float = 3.0) -> dict:
    """{(ti, tj): {"t": [...], "dist": [...], "iomin": [...], "tstar": [...], "dstar": [...]} }

    Only road users larger than min_size (share of image width) and away from the image border:
    far-away boxes jitter by a large fraction of their size, which reads as violent acceleration.
    """
    series: dict = defaultdict(lambda: defaultdict(list))
    tracks = ctx.tracks
    min_px = min_size * ctx.width
    for f in ctx.index.frames:
        items = [(ti, si) for ti, si in ctx.index.by_frame[int(f)]
                 if tracks[ti].group in ROAD_USERS and tracks[ti].size[si] >= min_px and not tracks[ti].edge[si]]
        if len(items) < 2:
            continue
        ti_arr = np.array([ti for ti, _ in items])
        foot = np.array([tracks[ti].foot[si] for ti, si in items])
        vel = np.array([tracks[ti].vel[si] for ti, si in items])
        size = np.array([tracks[ti].size[si] for ti, si in items])
        box = np.array([tracks[ti].box[si] for ti, si in items])
        motor = np.array([tracks[ti].group in MOTORISED for ti in ti_arr])

        sbar = (size[:, None] + size[None, :]) / 2
        rel = foot[None, :, :] - foot[:, None, :]
        dist = np.linalg.norm(rel, axis=2) / sbar
        rv = vel[None, :, :] - vel[:, None, :]
        rv2 = (rv ** 2).sum(axis=2)
        tstar = np.where(rv2 > 1e-6, -(rel * rv).sum(axis=2) / np.maximum(rv2, 1e-6), np.inf)
        cpa = rel + rv * np.clip(tstar, 0, horizon)[..., None]
        dstar = np.linalg.norm(cpa, axis=2) / sbar

        ix = np.clip(np.minimum(box[:, None, 2], box[None, :, 2]) - np.maximum(box[:, None, 0], box[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(box[:, None, 3], box[None, :, 3]) - np.maximum(box[:, None, 1], box[None, :, 1]), 0, None)
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        iomin = ix * iy / np.maximum(np.minimum(area[:, None], area[None, :]), 1.0)

        near = (dist < max_dist) | ((tstar > 0) & (tstar < horizon) & (dstar < 1.0) & (dist < 10))
        near &= motor[:, None] | motor[None, :]
        ii, jj = np.where(np.triu(near, 1))
        t = float(ctx.frame_times[int(f)])
        for i, j in zip(ii, jj):
            a, b = int(ti_arr[i]), int(ti_arr[j])
            key = (a, b) if a < b else (b, a)
            s = series[key]
            s["t"].append(t)
            s["dist"].append(dist[i, j])
            s["iomin"].append(iomin[i, j])
            s["tstar"].append(tstar[i, j])
            s["dstar"].append(dstar[i, j])
    return {k: {n: np.asarray(v) for n, v in s.items()} for k, s in series.items()}


# --------------------------------------------------------------------------- accident
def _impact(tr: Track, tc: float, cfg: EventCfg) -> tuple[float, dict]:
    """How abruptly this road user's motion changed at time tc (0..1) and why."""
    v_early = tr.median_in(tr.speed_fast, tc - 2.5, tc - 1.2)
    v_pre = tr.median_in(tr.speed_fast, tc - 1.0, tc - 0.05)
    v_post = tr.median_in(tr.speed_fast, tc + 0.4, tc + 1.4)
    feats = {"v_pre": v_pre, "v_post": v_post}
    if np.isnan(v_pre) or np.isnan(v_post):
        return 0.0, feats
    drop = v_pre - v_post
    score = np.clip((drop - 0.5) / 1.5, 0, 1) if v_pre >= cfg.acc_min_prior_speed else 0.0
    if not np.isnan(v_early) and v_early > 1.5 * max(v_pre, 0.1):
        score *= 0.4  # already braking hard before contact: a controlled stop, not an impact
    pre = tr.window(tc - 1.0, tc)
    post = tr.window(tc + 0.2, tc + 1.2)
    if v_pre >= 0.8 and len(tr.heading[pre]) and len(tr.heading[post]) and v_post >= 0.4:
        turn = float(angle_diff(np.median(tr.heading[post]), np.median(tr.heading[pre])))
        feats["deflection"] = turn
        score = max(score, np.clip((turn - 30) / 40, 0, 1))
    if tr.group == "person":
        asp = tr.aspect()
        a_pre = np.median(asp[pre]) if len(asp[pre]) else np.nan
        a_post = np.median(asp[tr.window(tc + 0.5, tc + 2.5)]) if len(asp[tr.window(tc + 0.5, tc + 2.5)]) else np.nan
        if not np.isnan(a_pre) and not np.isnan(a_post) and a_pre > 1.6 and a_post < 1.1:
            feats["fall"] = True
            score = max(score, 0.9)
    feats["impact"] = float(score)
    return float(score), feats


def _rest_time(tr: Track, t0: float, cfg: EventCfg, limit: float) -> tuple[float, bool]:
    """First time after t0 the track stays below rest speed for 1 s, or its end if it leaves."""
    w = tr.window(t0, t0 + limit)
    t, v = tr.t[w], tr.speed[w]
    for k in range(len(t)):
        stay = (t >= t[k]) & (t <= t[k] + 1.0)
        if v[k] < cfg.acc_rest_speed and np.all(v[stay] < cfg.acc_rest_speed * 1.5):
            return float(t[k]), True
    if tr.end < t0 + limit:
        return tr.end, False
    return float(t0 + limit), False


def detect_accidents(ctx: Context, series: dict, cfg: EventCfg) -> list[Candidate]:
    out: list[Candidate] = []
    for (a, b), s in series.items():
        ta, tb = ctx.tracks[a], ctx.tracks[b]
        contact = (s["dist"] < 1.0) & (s["iomin"] > 0.02)
        if not contact.any():
            continue
        idx = np.where(contact & ~np.r_[False, contact[:-1]])[0]  # first frame of each contact run
        for k in idx:
            tc = float(s["t"][k])
            ia, fa = _impact(ta, tc, cfg)
            ib, fb = _impact(tb, tc, cfg)
            impact = max(ia, ib)
            if impact < 0.35:
                continue
            closing = k > 0 and s["dist"][max(0, k - 4)] > s["dist"][k] + 0.2
            ra, rest_a = _rest_time(ta, tc, cfg, cfg.acc_max_len)
            rb, rest_b = _rest_time(tb, tc, cfg, cfg.acc_max_len)
            after = (s["t"] > tc) & (s["t"] <= tc + 4.0)
            stay_close = bool(after.any() and np.median(s["dist"][after]) < 1.6)
            post = 1.0 if (rest_a or rest_b) and stay_close else 0.6 if (rest_a or rest_b) else 0.2
            score = 0.55 * impact + 0.30 * post + 0.15 * float(closing)
            if score < 0.6:
                continue
            end = min(max(ra, rb, tc + 1.0), tc + cfg.acc_max_len)
            out.append(Candidate(tc, end, "accident", score, (ta.tid, tb.tid),
                                 {"impact": impact, "post": post, "closing": closing, "a": fa, "b": fb}))
    return _dedupe(out)


# --------------------------------------------------------------------------- near miss
def _evasion(tr: Track, t0: float, t1: float, cfg: EventCfg) -> tuple[float | None, str]:
    """Onset time of hard braking or a swerve inside [t0, t1], if any.

    Braking must be sustained (speed falls by >= 40 % within 1.5 s) and the track must have been
    observed for >= 1.5 s before, so fragment starts and detector jitter do not count.
    """
    w = tr.window(max(t0, tr.start + 1.5), t1)
    t, acc, v, hd = tr.t[w], tr.accel_fast[w], tr.speed_fast[w], tr.heading[w]
    hard = np.where((acc < -cfg.nm_brake) & (v > 0.5))[0]
    for k in hard:
        while k > 0 and acc[k - 1] < -0.4 * cfg.nm_brake:
            k -= 1
        before = tr.median_in(tr.speed_fast, t[k] - 0.4, t[k])
        after = tr.median_in(tr.speed_fast, t[k] + 0.8, t[k] + 1.5)
        if before > 0.8 and after < 0.6 * before:
            return float(t[k]), "brake"
    for k in range(len(t)):
        later = np.where((t > t[k]) & (t <= t[k] + 1.0) & (v > 0.8))[0]
        if v[k] > 0.8 and len(later) and angle_diff(hd[later], hd[k]).max() > cfg.nm_swerve_deg:
            return float(t[k]), "swerve"
    return None, ""


def detect_near_misses(ctx: Context, series: dict, accidents: list[Candidate], cfg: EventCfg) -> list[Candidate]:
    out: list[Candidate] = []
    crashed = {tid for c in accidents for tid in c.tracks}
    for (a, b), s in series.items():
        ta, tb = ctx.tracks[a], ctx.tracks[b]
        if ta.tid in crashed and tb.tid in crashed:
            continue
        conflict = (s["tstar"] > 0) & (s["tstar"] < cfg.nm_ttc) & (s["dstar"] < cfg.nm_min_gap + 0.3) \
            & (s["dist"] < 8.0)
        conflict &= np.r_[False, conflict[:-1]]      # predicted on two consecutive analysed frames
        if not conflict.any():
            continue
        k = int(np.argmax(conflict)) - 1
        tk = float(s["t"][k])
        onsets = [(o, kind, tr) for tr in (ta, tb) if tr.group in MOTORISED
                  for o, kind in [_evasion(tr, tk - 1.0, tk + 1.5, cfg)] if o is not None]
        if not onsets:
            continue
        onset, kind, actor = min(onsets, key=lambda x: x[0])
        span = (s["t"] >= onset) & (s["t"] <= onset + 6.0)
        if not span.any():
            continue
        min_dist = float(s["dist"][span].min())
        if min_dist < 0.6 and float(s["iomin"][span].max()) > 0.05:
            continue  # they touched: that is an accident candidate, not a near miss
        k_min = int(np.where(span)[0][np.argmin(s["dist"][span])])
        clear = np.where((s["t"] > s["t"][k_min]) & (s["dist"] > max(1.5, min_dist + 0.8)))[0]
        end = float(s["t"][clear[0]]) if len(clear) else min(float(s["t"][span][-1]), onset + 4.0)
        if end - onset < 0.5:
            end = onset + 0.5
        score = 0.6 + 0.2 * float(kind == "brake") + 0.2 * float(np.clip(1.0 - min_dist, 0, 1))
        out.append(Candidate(onset, end, "near_miss", score, (ta.tid, tb.tid),
                             {"kind": kind, "actor": actor.tid, "min_dist": min_dist}))
    return _dedupe(out)


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    """One event per incident: overlapping candidates sharing a track keep the best-scoring one."""
    cands = sorted(cands, key=lambda c: -c.score)
    kept: list[Candidate] = []
    for c in cands:
        if any(set(c.tracks) & set(k.tracks) and c.start < k.end + 1.0 and k.start < c.end + 1.0 for k in kept):
            continue
        kept.append(c)
    return kept


def detect(ctx: Context, cfg: EventCfg) -> list[Candidate]:
    series = pair_series(ctx, cfg.min_size)
    accidents = detect_accidents(ctx, series, cfg)
    return accidents + detect_near_misses(ctx, series, accidents, cfg)
