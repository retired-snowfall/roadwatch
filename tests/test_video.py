"""The frame reader: right frames, right timestamps, working size, OpenCV fallback."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roadwatch import video
from roadwatch.video import FrameSampler, VideoInfo, iter_frames, work_size


@pytest.fixture(scope="module")
def counter_video(tmp_path_factory):
    """2 s at 30 fps, 640x360; frame i is uniformly grey at level 4*i, so every frame is identifiable."""
    path = tmp_path_factory.mktemp("vid") / "counter.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
    for i in range(60):
        w.write(np.full((360, 640, 3), 4 * i, np.uint8))
    w.release()
    return str(path)


def test_work_size_caps_width_and_keeps_aspect():
    assert work_size(3840, 2160) == (1920, 1080)
    assert work_size(1920, 1080) == (1920, 1080)
    assert work_size(1280, 720) == (1280, 720)
    assert VideoInfo("x.mp4", 29.97, 3840, 2160, 100).work == (1920, 1080)


def _check(frames, rate):
    assert abs(len(frames) - 2 * rate) <= 1
    for idx, t, img in frames:
        assert img.shape == (90, 160, 3)
        assert t == pytest.approx(idx / 30.0)
        assert abs(float(img.mean()) - 4 * idx) < 4, (idx, img.mean())   # the image is frame idx
    ts = [t for _, t, _ in frames]
    assert ts == sorted(ts)


@pytest.mark.parametrize("rate", [5.0, 10.0])
def test_reader_returns_the_frames_it_claims(counter_video, rate):
    _check(list(iter_frames(counter_video, FrameSampler(30.0, rate), (160, 90))), rate)


def test_reader_falls_back_to_opencv(counter_video, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no decoder")
    monkeypatch.setattr(video, "_decode_pyav", broken)
    _check(list(iter_frames(counter_video, FrameSampler(30.0, 5.0), (160, 90))), 5.0)


def test_reader_stops_cleanly_when_abandoned(counter_video):
    it = iter_frames(counter_video, FrameSampler(30.0, 30.0), (160, 90), prefetch=2)
    next(it)
    it.close()   # must not hang on the full queue
