"""
solution.py — entry point imported by the organizers' harness (run_submission.py).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The implementation lives in src/roadwatch (see README.md for the pipeline):
detector + tracker -> trajectories -> scene model of the fixed camera -> rules per class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from roadwatch.constants import CLASSES as _OFFICIAL  # noqa: E402
from roadwatch.pipeline import detect_events as _detect_events  # noqa: E402
from roadwatch.risk import CausalRisk  # noqa: E402

TEAM_NAME = "roadwatch"

# Official class ids (14). All are kept: which ones we actually report is decided in
# src/roadwatch/config.py (Config.enabled).
CLASSES: list[str] = list(_OFFICIAL)

# Anticipation horizon used by the metric (seconds).
RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection: [[start_sec, end_sec, label], ...] for one .mp4."""
    return _detect_events(video_path)


class RiskEstimator:
    """Part B — causal accident anticipation.

    Uses only the frames passed to step(), in order; it never opens the video file and
    never reads Part A results.
    """

    def __init__(self) -> None:
        self._impl = CausalRisk()

    def reset(self, meta: dict) -> None:
        """meta = {"video_id", "fps", "width", "height", "n_frames"}"""
        self._impl.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within the next RISK_HORIZON_SEC s) in [0, 1]."""
        return self._impl.step(frame, t_sec)
