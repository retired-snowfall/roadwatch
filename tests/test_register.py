"""View registration: a re-aimed camera is mapped back onto the reference view."""
from __future__ import annotations

import cv2
import numpy as np

from roadwatch import register


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (135, 240, 3), dtype=np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST)
    for _ in range(60):   # some structure: lines and blocks like road markings and buildings
        x, y = rng.integers(0, 1900), rng.integers(0, 1060)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(20, 200)), int(y + rng.integers(5, 60))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_recovers_a_shift_and_zoom_and_rejects_another_scene():
    ref = _texture(1)
    true = np.array([[1.02, 0.0, -40.0], [0.0, 1.02, 25.0], [0.0, 0.0, 1.0]])     # reference -> video
    video = cv2.warpPerspective(ref, true, (1920, 1080))
    video = np.clip(video.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)       # "dusk"
    H = register.homography(video, ref)
    assert H is not None
    pts = np.array([[300.0, 200.0], [1500.0, 900.0], [960.0, 540.0]])
    back = register.warp_points(H, register.warp_points(true, pts))
    assert np.abs(back - pts).max() < 3.0
    assert register.homography(_texture(2), ref) is None


def test_warp_boxes_bounds_the_mapped_corners():
    H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]])
    assert np.allclose(register.warp_boxes(H, [[0, 0, 100, 50]]), [[10, -5, 110, 45]])
