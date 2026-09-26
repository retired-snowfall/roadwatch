"""Shared wall clock for the harness's per-video time budget.

run_submission.py allows TIME_FACTOR x duration for Part A + Part B *together* and scores a
video as empty (events and risk) when it runs over. Part A records when a video started;
Part B reads that and lowers its detection rate if its projected finish would come too close
to the deadline. On the target hardware neither guard triggers; they only make a slow
machine degrade gracefully instead of losing the whole video.
"""
from __future__ import annotations

import time
from pathlib import Path

TIME_FACTOR = 3.0      # official budget multiplier
SAFETY = 0.85          # plan to finish by this share of the budget

_started: dict[str, float] = {}


def mark_start(video_path: str) -> None:
    """Called when Part A starts on a video (the harness starts its clock just before)."""
    _started[Path(video_path).name] = time.perf_counter()


def deadline(video_id: str, duration: float) -> float:
    """Wall-clock time by which Part B should be done; if Part A did not run in this process,
    Part B alone gets the budget from now."""
    start = _started.get(Path(str(video_id)).name, time.perf_counter())
    return start + SAFETY * TIME_FACTOR * max(duration, 1.0)
