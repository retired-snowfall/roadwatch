"""Each rule fires on a hand-built scenario with the expected boundaries, and stays quiet on normal traffic."""
from __future__ import annotations

import numpy as np
import pytest

from roadwatch.config import CFG
from roadwatch.events import collision, lines, motion, pedestrian, run_all, stationary
from roadwatch.events.base import finalize
from synth import context, make_track, two_way_road_scene


@pytest.fixture(scope="module")
def scene():
    return two_way_road_scene()


def normal_traffic(t_end: float = 60.0) -> list:
    out, tid = [], 1
    for t0 in np.arange(0, t_end - 6, 5.0):
        out.append(make_track(tid, [(t0, 50, 470), (t0 + 6, 1870, 470)]))
        out.append(make_track(tid + 1, [(t0 + 2, 1870, 630), (t0 + 8, 50, 630)]))
        tid += 2
    return out


def labels(cands):
    return sorted({c.label for c in cands})


def test_normal_traffic_is_quiet(scene):
    ctx = context(normal_traffic(), scene)
    assert run_all(ctx, CFG) == []


def test_wrong_way(scene):
    # drives left (x decreasing) in the right-bound lane from t=10 to t=16, leaves the frame
    bad = make_track(1, [(10, 1870, 470), (16, 50, 470)])
    cands = motion.detect_wrong_way(context([bad], scene), CFG.events)
    assert labels(cands) == ["wrong_way"]
    assert cands[0].start == pytest.approx(10, abs=0.8) and cands[0].end == pytest.approx(16, abs=0.5)


def test_u_turn(scene):
    pts = [(0, 300, 470), (4, 1000, 470)]
    for k, a in enumerate(np.linspace(0, np.pi, 9)[1:]):          # half circle of radius 80 px, 4 s
        pts.append((4 + 0.5 * (k + 1), 1000 + 80 * np.sin(a), 550 - 80 * np.cos(a)))
    pts.append((12, 300, 630))
    tr = make_track(1, pts)
    cands = motion.detect_u_turns(context([tr], scene), CFG.events)
    assert labels(cands) == ["illegal_u_turn"]
    assert 3.0 <= cands[0].start <= 5.0 and 7.0 <= cands[0].end <= 9.5


def test_stopped_vehicle(scene):
    passing = normal_traffic()
    stopper = make_track(99, [(5, 50, 470), (8, 900, 470), (30, 900, 470), (33, 1870, 470)])
    ctx = context(passing + [stopper], scene)
    cands = stationary.detect(ctx, CFG.events, CFG.kin)
    sv = [c for c in cands if c.label == "stopped_vehicle"]
    assert len(sv) == 1 and 99 in sv[0].tracks
    assert sv[0].start == pytest.approx(8, abs=1.0) and sv[0].end == pytest.approx(30, abs=1.0)


def test_jaywalking(scene):
    person = make_track(1, [(5, 1000, 300), (8, 1000, 400), (14, 1000, 700), (17, 1000, 850)],
                        group="person", size=(30, 80), cls=0)
    cands = pedestrian.detect_jaywalking(context([person], scene), CFG.events)
    assert labels(cands) == ["jaywalking"]
    assert cands[0].start == pytest.approx(8, abs=1.0) and cands[0].end == pytest.approx(14, abs=1.0)


def test_accident(scene):
    a = make_track(1, [(0, 100, 470), (5, 1000, 470), (5.4, 1030, 480), (30, 1030, 480)])
    b = make_track(2, [(0, 1100, 150), (5, 1100, 460), (5.4, 1090, 470), (30, 1090, 470)])
    ctx = context([a, b], scene)
    cands = collision.detect(ctx, CFG.events)
    acc = [c for c in cands if c.label == "accident"]
    assert len(acc) == 1, [c.as_json() for c in cands]
    assert acc[0].start == pytest.approx(5, abs=0.8) and acc[0].end < 12


def test_queue_stop_is_not_an_accident(scene):
    lead = make_track(1, [(0, 900, 470), (30, 900, 470)])
    follower = make_track(2, [(0, 100, 470), (4, 600, 470), (7, 780, 470), (30, 780, 470)])
    cands = collision.detect(context([lead, follower], scene), CFG.events)
    assert [c for c in cands if c.label == "accident"] == []


def test_near_miss(scene):
    # the car drives at ~3.5 sizes/s, a pedestrian steps into its lane, the car stops within ~1.2 s
    car = make_track(1, [(0, 100, 470), (2, 800, 470), (2.4, 925, 470), (2.8, 1010, 470),
                         (3.2, 1040, 470), (3.6, 1045, 470), (7, 1045, 470), (10, 1800, 470)])
    walker = make_track(2, [(0, 1250, 330), (2, 1250, 420), (4, 1250, 520), (7, 1250, 750)],
                        group="person", size=(30, 80), cls=0)
    cands = collision.detect(context([car, walker], scene), CFG.events)
    assert "near_miss" in labels(cands), [c.as_json() for c in cands]
    assert "accident" not in labels(cands)


def test_congestion(scene):
    jam = []
    for k in range(12):   # 12 cars crawling in the right-bound lane for 90 s
        x0 = 150 + 140 * k
        jam.append(make_track(10 + k, [(0, x0, 470), (90, x0 + 60, 470)]))
    cands = stationary.detect_congestion(context(jam, scene, duration=90), CFG.events)
    assert labels(cands) == ["congestion"]
    assert cands[0].start < 5 and cands[0].end > 85


def test_solid_line_crossing(scene):
    scene.zones["solid_lines"] = [[[0.05, 545 / 1080], [0.95, 545 / 1080]]]
    try:
        tr = make_track(1, [(0, 100, 470), (4, 800, 470), (6, 1100, 630), (9, 1700, 630)])
        cands = lines.detect_solid_line_crossings(context([tr], scene), CFG.events)
        assert labels(cands) == ["solid_line_crossing"]
        assert 4.0 <= cands[0].start <= 5.5 and 5.0 <= cands[0].end <= 6.5
    finally:
        scene.zones["solid_lines"] = []


def test_red_light(scene):
    # stop line at x=1200 across the right-bound lanes; one car waits, another runs it
    scene.zones["stop_lines"] = [{"points": [[1200 / 1920, 420 / 1080], [1200 / 1920, 520 / 1080]], "dir": 0}]
    try:
        waiter = make_track(1, [(0, 900, 440), (4, 1150, 440), (40, 1150, 440), (46, 1870, 440)])
        runner = make_track(2, [(8, 100, 505), (14, 1870, 505)])
        ctx = context([waiter, runner], scene)
        cands = lines.detect_signal_events(ctx, CFG.events, CFG.kin)
        rl = [c for c in cands if c.label == "red_light"]
        assert len(rl) == 1 and rl[0].tracks == (2,)
        assert rl[0].start == pytest.approx(8 + 6 * 1100 / 1770, abs=0.6)
    finally:
        scene.zones["stop_lines"] = []


def test_finalize_merges_and_clips():
    from roadwatch.events.base import Candidate
    cands = [Candidate(1.0, 3.0, "jaywalking"), Candidate(3.5, 6.0, "jaywalking"),
             Candidate(58.0, 70.0, "congestion"), Candidate(10.0, 10.3, "wrong_way")]
    out = finalize(cands, 60.0, merge_gap=1.0, min_duration=0.8, enabled=CFG.enabled)
    assert out == [[1.0, 6.0, "jaywalking"], [58.0, 60.0, "congestion"]]
