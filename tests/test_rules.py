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
    car = make_track(1, [(0, 100, 470), (2, 800, 470), (2.4, 925, 470), (2.8, 1060, 470),
                         (3.2, 1105, 470), (3.6, 1112, 470), (7, 1112, 470), (10, 1800, 470)])
    walker = make_track(2, [(0, 1230, 330), (2, 1230, 420), (4, 1230, 520), (7, 1230, 750)],
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
        # centred on the crossing (about t = 5 s), at least slc_min_len long
        assert cands[0].start <= 5.0 <= cands[0].end and 1.9 <= cands[0].end - cands[0].start <= 3.0
        # caught by the signal astride the line: the violation lasts while it stands there
        astride = make_track(2, [(0, 100, 470), (4, 800, 470), (5, 950, 545), (35, 950, 545), (37, 1200, 630),
                                 (40, 1700, 630)])
        cands = lines.detect_solid_line_crossings(context([astride], scene), CFG.events)
        assert labels(cands) == ["solid_line_crossing"]
        assert cands[0].start <= 6.0 and cands[0].end >= 35.0
    finally:
        scene.zones["solid_lines"] = []


@pytest.mark.parametrize("n_waiting, expected", [(2, 1), (1, 0)])
def test_red_light(scene, n_waiting, expected):
    # stop line at x=1200 across the right-bound lanes; cars wait at it, another runs it.
    # Without a visible light one waiting car (it may be turning) is not enough evidence of red.
    scene.zones["stop_lines"] = [{"points": [[1200 / 1920, 420 / 1080], [1200 / 1920, 520 / 1080]], "dir": 0}]
    try:
        waiters = [make_track(1 + k, [(0, 900, y), (4, 1150, y), (40, 1150, y), (46, 1870, y)])
                   for k, y in enumerate((440, 470)[:n_waiting])]
        runner = make_track(9, [(8, 100, 505), (14, 1870, 505)])
        ctx = context(waiters + [runner], scene)
        cands = lines.detect_signal_events(ctx, CFG.events, CFG.kin)
        rl = [c for c in cands if c.label == "red_light"]
        assert len(rl) == expected and all(c.tracks == (9,) for c in rl)
        if rl:
            assert rl[0].start == pytest.approx(8 + 6 * 1100 / 1770, abs=0.6)
    finally:
        scene.zones["stop_lines"] = []


def test_finalize_merges_and_clips():
    from roadwatch.events.base import Candidate
    cands = [Candidate(1.0, 3.0, "jaywalking"), Candidate(3.5, 6.0, "jaywalking"),
             Candidate(58.0, 70.0, "stopped_vehicle"), Candidate(10.0, 10.3, "wrong_way"),
             Candidate(20.0, 30.0, "congestion")]
    out = finalize(cands, 60.0, merge_gap=1.0, min_duration=0.8, enabled=CFG.enabled)
    # merged, clipped to the video, blips dropped, classes we do not report left out
    assert out == [[1.0, 6.0, "jaywalking"], [58.0, 60.0, "stopped_vehicle"]]


def test_parked_cars_create_no_stop_lines_or_violations(scene):
    parked = [make_track(50 + k, [(0, 300 + 250 * k, 430), (120, 300 + 250 * k, 430)]) for k in range(4)]
    from roadwatch.scene import SceneModel
    fresh = SceneModel(1920, 1080, CFG.scene)
    fresh.accumulate(parked + normal_traffic(120), 120.0, CFG.kin)
    assert fresh.queue_heads == [] and fresh.stop_lines() == []
    ctx = context(parked + normal_traffic(120), scene, duration=120)
    assert [c for c in run_all(ctx, CFG) if c.label in ("stop_line", "red_light")] == []


def test_signal_queues_are_learned_as_a_stop_line():
    from roadwatch.scene import SceneModel
    tracks, tid = [], 1
    for cycle in range(4):                       # four red phases, a different car waits at x~1180 each time
        t0 = 60.0 * cycle
        tracks.append(make_track(tid, [(t0, 200, 470), (t0 + 4, 1150, 470), (t0 + 30, 1150, 470), (t0 + 34, 1870, 470)]))
        tid += 1
    fresh = SceneModel(1920, 1080, CFG.scene)
    fresh.accumulate(tracks, 240.0, CFG.kin)
    lines = fresh.stop_lines()
    assert len(lines) == 1 and abs(lines[0]["dir"]) < 15
    assert 1150 < (lines[0]["a"][0] + lines[0]["b"][0]) / 2 < 1260


def test_identity_switch_in_a_queue_is_not_an_accident(scene):
    # a stopped car's track jumps onto the stopped car ahead (a detector/tracker glitch), then stays
    lead = make_track(1, [(0, 900, 470), (30, 900, 470)])
    hop = make_track(2, [(0, 600, 470), (5.0, 600, 470), (5.2, 860, 470), (30, 860, 470)])
    cands = collision.detect(context([lead, hop], scene), CFG.events)
    assert [c for c in cands if c.label == "accident"] == [], [c.as_json() for c in cands]


def test_turning_past_a_stopped_car_is_not_an_accident(scene):
    parked = make_track(1, [(0, 1000, 560), (30, 1000, 560)])
    turner = make_track(2, [(0, 200, 470), (4, 900, 470), (4.6, 960, 500), (5.2, 990, 580), (8, 1000, 900)])
    cands = collision.detect(context([parked, turner], scene), CFG.events)
    assert "accident" not in labels(cands), [c.as_json() for c in cands]


@pytest.mark.parametrize("ped_x, y_end, expected", [(1230, 420, True), (1650, 640, False)])
def test_failure_to_yield_needs_a_pedestrian_next_to_the_car(scene, ped_x, y_end, expected):
    """The walker crosses into the car's lanes (True) or stays in the opposite lanes (False)."""
    import copy

    sc = copy.deepcopy(scene)
    # a crosswalk across the whole road, x 1150..1700, y 400..690 (normalised coordinates)
    sc.zones["crosswalks"] = [{"points": [[1150 / 1920, 400 / 1080], [1700 / 1920, 400 / 1080],
                                          [1700 / 1920, 690 / 1080], [1150 / 1920, 690 / 1080]]}]
    car = make_track(1, [(0, 100, 470), (6, 1900, 470)])                     # ~3 sizes/s, never slows
    walker = make_track(2, [(0, ped_x, 690), (8, ped_x, y_end)], group="person", size=(30, 80), cls=0)
    cands = pedestrian.detect_failure_to_yield(context([car, walker], sc), CFG.events)
    assert (labels(cands) == ["failure_to_yield"]) is expected, [c.as_json() for c in cands]


def test_parked_car_in_a_no_stopping_zone_spans_identity_switches(scene):
    # a car cut off by the right frame border stands in a drawn no-stopping zone the whole minute;
    # passing traffic hides it for 5 s and it comes back under a new track id
    saved = scene.zones.get("no_stopping", [])
    scene.zones["no_stopping"] = [[[0.9, 0.1], [1.0, 0.1], [1.0, 0.3], [0.9, 0.3]]]
    try:
        first = make_track(1, [(0, 1880, 250), (25, 1880, 250)])
        second = make_track(2, [(30, 1880, 252), (60, 1880, 252)])
        assert first.edge.all()
        cands = stationary.detect(context([first, second], scene), CFG.events, CFG.kin)
        sv = [c for c in cands if c.label == "stopped_vehicle"]
        assert len(sv) == 1 and set(sv[0].tracks) == {1, 2}
        assert sv[0].start == 0.0 and sv[0].end == 60.0
    finally:
        scene.zones["no_stopping"] = saved


def test_vehicle_stuck_on_a_crossing_is_a_stop_line_violation(scene):
    saved = scene.zones.get("crosswalks", [])
    scene.zones["crosswalks"] = [[[0.49, 0.36], [0.55, 0.36], [0.55, 0.64], [0.49, 0.64]]]
    try:
        stuck = make_track(1, [(0, 100, 470), (5, 1000, 470), (35, 1000, 470), (40, 1870, 470)])
        passing = make_track(2, [(10, 100, 490), (16, 1870, 490)])
        cands = lines.detect_crosswalk_blocking(context([stuck, passing], scene), CFG.events, CFG.kin)
        assert labels(cands) == ["stop_line"]
        assert cands[0].start == pytest.approx(5, abs=1.0) and cands[0].end == pytest.approx(35, abs=1.0)
        # a car that parked there before the video started never drove in
        parked = make_track(3, [(0, 1000, 470), (40, 1000, 470)])
        assert lines.detect_crosswalk_blocking(context([parked], scene), CFG.events, CFG.kin) == []
    finally:
        scene.zones["crosswalks"] = saved
