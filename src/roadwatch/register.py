"""Register a video's view to the reference view of the scene model.

The organisers' camera is re-aimed between recordings: the sample videos differ by tens of
pixels of shift and a few percent of zoom. Everything learned about the junction (lanes, stop
lines, movements, drawn zones) lives in one reference view; a video is mapped onto it by a
homography estimated from its static background (SIFT on contrast-equalised grey images, so a
dusk recording still matches a daytime reference), and the rules run in reference coordinates.
"""
from __future__ import annotations

import copy
import logging

import cv2
import numpy as np

from .config import KinematicsCfg

log = logging.getLogger("roadwatch")
MATCH_WIDTH = 960          # features are computed at this width
MIN_INLIERS = 40
_REF_CACHE: dict[int, tuple] = {}


def _prep(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    s = MATCH_WIDTH / bgr.shape[1]
    g = cv2.cvtColor(cv2.resize(bgr, (MATCH_WIDTH, int(round(bgr.shape[0] * s))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g), s


def _features(bgr: np.ndarray):
    g, s = _prep(bgr)
    kp, des = cv2.SIFT_create(4000).detectAndCompute(g, None)
    return np.float32([k.pt for k in kp]) / s, des


def homography(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray | None:
    """3x3 H mapping src pixels to reference pixels, or None if the views do not match well enough
    to trust (another camera, or too little shared background)."""
    key = id(ref_bgr)
    if key not in _REF_CACHE or _REF_CACHE[key][0] is not ref_bgr:
        _REF_CACHE.clear()
        _REF_CACHE[key] = (ref_bgr, _features(ref_bgr))
    p_ref, d_ref = _REF_CACHE[key][1]
    p_src, d_src = _features(src_bgr)
    if d_src is None or d_ref is None or len(d_src) < MIN_INLIERS or len(d_ref) < MIN_INLIERS:
        return None
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    pairs = matcher.knnMatch(d_src, d_ref, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    a = p_src[[m.queryIdx for m in good]]
    b = p_ref[[m.trainIdx for m in good]]
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 8.0)
    if H is None or int(inl.sum()) < MIN_INLIERS:
        return None
    # a re-aimed fixed camera: modest zoom, little rotation, almost no perspective change
    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    rot = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    if not (0.75 <= scale <= 1.33) or abs(rot) > 10 or np.abs(H[2, :2]).max() > 2e-4:
        log.info("registration rejected: scale %.3f rotation %.1f", scale, rot)
        return None
    log.info("registered to the reference view: %d inliers, shift (%.0f, %.0f) px, scale %.3f",
             int(inl.sum()), H[0, 2], H[1, 2], scale)
    return H


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts[None], H)[0]


def warp_boxes(H: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Axis-aligned boxes around the mapped corners."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    if len(b) == 0:
        return b
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [0, 3]], b[:, [2, 3]]], axis=1)
    m = warp_points(H, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.c_[m[:, :, 0].min(1), m[:, :, 1].min(1), m[:, :, 0].max(1), m[:, :, 1].max(1)]


def warp_tracks(tracks: list, H: np.ndarray, kin: KinematicsCfg, width: int, height: int) -> list:
    """Copies of the tracks in reference coordinates (same ids), kinematics recomputed."""
    out = []
    for tr in tracks:
        w = copy.copy(tr)
        w.box = warp_boxes(H, tr.box).astype(tr.box.dtype)
        w.compute_kinematics(kin, width, height)
        w.edge = tr.edge.copy()   # cut off by the video's own frame border, not the reference's
        out.append(w)
    return out


def warp_image(img: np.ndarray, H: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    return cv2.warpPerspective(img, H, size, flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
