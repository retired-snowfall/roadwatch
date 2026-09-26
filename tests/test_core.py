"""Tracker, kinematics, stitching, scene persistence and the solution.py contract."""
from __future__ import annotations

import json

import numpy as np
import pytest

from roadwatch.config import CFG
from roadwatch.scene import SceneModel
from roadwatch.tracker import ByteTracker
from roadwatch.tracks import Track, local_linear, stitch
from synth import make_track, two_way_road_scene


def test_local_linear_recovers_line():
    t = np.linspace(0, 10, 51)
    x = 3.0 + 2.5 * t
    val, slope = local_linear(t, x, 0.5)
    assert np.allclose(val, x) and np.allclose(slope, 2.5)


def test_tracker_keeps_ids_for_fast_objects_at_low_rate():
    trk = ByteTracker(CFG.tracker)
    ids = set()
    for k in range(20):                      # car moving 90 px per analysed frame, plus a pedestrian
        t = k * 0.2
        car = [100 + 90 * k, 400, 220 + 90 * k, 480, 0.9, 2]
        ped = [900, 300 + 5 * k, 930, 380 + 5 * k, 0.8, 0]
        out = trk.update(np.array([car, ped], np.float32), t)
        if k >= 3:
            ids |= {(o.group, o.tid) for o in out}
    assert len({i for g, i in ids if g == "vehicle"}) == 1
    assert len({i for g, i in ids if g == "person"}) == 1


def test_tracker_never_mixes_groups():
    trk = ByteTracker(CFG.tracker)
    trk.update(np.array([[100, 100, 200, 200, 0.9, 2]], np.float32), 0.0)
    out = trk.update(np.array([[100, 100, 200, 200, 0.9, 0]], np.float32), 0.2)   # same box, now a person
    assert all(o.group == "person" for o in out) and out[0].tid != 1


def test_stitch_joins_occluded_fragments():
    a = make_track(1, [(0, 100, 470), (4, 700, 470)])
    b = make_track(2, [(5, 850, 470), (9, 1450, 470)])
    other = make_track(3, [(5, 850, 800), (9, 1450, 800)], group="person", size=(30, 80), cls=0)
    raw = [Track(t.tid, t.group, t.cls, t.fidx, t.t, t.box, t.conf) for t in (a, b, other)]
    out = stitch(raw, CFG.kin)
    assert sorted(len(t.t) for t in out) == sorted([len(a.t) + len(b.t), len(other.t)])


def test_scene_learns_lane_directions_and_round_trips(tmp_path):
    scene = two_way_road_scene()
    d, ok = scene.lane_direction(np.array([[960, 470], [960, 630]]))
    assert ok.all()
    assert abs(d[0]) < 15 and abs(abs(d[1]) - 180) < 15
    assert scene.on_road(np.array([[960, 470], [960, 900]])).tolist() == [True, False]
    scene.save(tmp_path / "scene.json")
    back = SceneModel.load(tmp_path / "scene.json")
    assert np.array_equal(back.derived["road"], scene.derived["road"])
    json.loads((tmp_path / "scene.json").read_text())


def test_solution_interface_and_causality():
    import solution
    from evaluate import OFFICIAL_CLASSES
    assert set(solution.CLASSES) <= set(OFFICIAL_CLASSES)
    est = solution.RiskEstimator()
    est.reset({"video_id": "x.mp4", "fps": 25.0, "width": 640, "height": 360, "n_frames": 50})
    rng = np.random.default_rng(0)
    scores = [est.step(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), i / 25) for i in range(50)]
    assert all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in scores)
    assert not hasattr(est._impl, "video_path")      # never given, never opened


@pytest.mark.parametrize("n", [0, 3])
def test_finalized_events_are_valid_for_the_harness(n):
    from run_submission import clean_events
    import solution
    events = [[1.0 + i, 2.0 + i, "jaywalking"] for i in range(0, 2 * n, 2)]
    kept, problems = clean_events(events, solution.CLASSES, 60.0)
    assert kept == events and problems == []


def test_risk_time_guard_lowers_rate_only_when_behind():
    import time

    from roadwatch.risk import CausalRisk
    from roadwatch.video import FrameSampler

    def estimator(seconds_left: float) -> CausalRisk:
        r = CausalRisk.__new__(CausalRisk)   # no detector needed for the guard
        r.cfg, r.duration = CFG, 100.0
        r.sampler = FrameSampler(30.0, 6.0)
        r.deadline = time.perf_counter() + seconds_left
        r._guard = (time.perf_counter() - 3.0, 10.0)   # recent pace: 1 wall second per video second
        r._time_guard(13.0)
        return r

    assert estimator(seconds_left=20.0).sampler.rate < 6.0      # 87 s of video left, 20 s of budget
    assert estimator(seconds_left=500.0).sampler.rate == pytest.approx(6.0)


def test_candidate_evidence_is_json_serialisable():
    from roadwatch.events.base import Candidate

    c = Candidate(1.0, 2.0, "accident", np.float32(0.7), (np.int64(3), 4),
                  {"closing": np.bool_(True), "a": {"v_pre": np.float64(1.23456), "gap": np.nan}, "k": np.int32(2),
                   "pts": np.array([1.0, 2.0])})
    d = json.loads(json.dumps(c.as_json()))
    assert d["info"] == {"closing": True, "a": {"v_pre": 1.235, "gap": None}, "k": 2, "pts": [1.0, 2.0]}


def test_oblique_stop_line_measures_distance_along_travel():
    from roadwatch.signals import StopLine

    line = StopLine(np.array([0.0, 0.0]), np.array([100.0, 50.0]), 90.0)   # slanted line, traffic moves down
    assert line.along([[50.0, 25.0]])[0] == pytest.approx(0.0, abs=1e-6)    # on the line
    assert line.along([[100.0, 60.0]])[0] == pytest.approx(10.0, abs=1e-6)  # 10 px below its right end
    assert line.along([[0.0, -5.0]])[0] == pytest.approx(-5.0, abs=1e-6)    # before the line
    assert abs(line.across([[100.0, 50.0]])[0]) == pytest.approx(line.half, abs=1e-6)
