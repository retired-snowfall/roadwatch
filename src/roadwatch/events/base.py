"""Candidate events and segment post-processing shared by all rule modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene import SceneModel
from ..tracks import FrameIndex, Track


@dataclass
class Candidate:
    """One detected event before post-processing. `score` in [0, 1] ranks confidence."""
    start: float
    end: float
    label: str
    score: float = 1.0
    tracks: tuple = ()
    info: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2), "label": self.label,
                "score": round(float(self.score), 3), "tracks": [int(t) for t in self.tracks],
                "info": _plain(self.info)}


def _plain(v):
    """JSON-safe copy of rule evidence (NumPy scalars and arrays, nested dicts, NaN -> None)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 3)
    return v


@dataclass
class Context:
    """Everything a rule module may look at for one video."""
    tracks: list[Track]
    scene: SceneModel
    duration: float
    fps: float
    width: int
    height: int
    index: FrameIndex
    frame_times: np.ndarray           # analysed frame index -> seconds
    signals: dict = field(default_factory=dict)       # per stop line: red intervals
    extras: dict = field(default_factory=dict)        # per-frame side channels (fire, obstacles, lights)

    def by_group(self, *groups: str) -> list[Track]:
        return [tr for tr in self.tracks if tr.group in groups]


def merge_intervals(iv: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def mask_to_intervals(t: np.ndarray, mask: np.ndarray, gap: float, min_len: float) -> list[tuple[float, float]]:
    """Turn a boolean series sampled at times t into merged [start, end] intervals."""
    from ..scene import runs
    iv = [(float(t[s]), float(t[e])) for s, e in runs(mask)]
    return [(s, e) for s, e in merge_intervals(iv, gap) if e - s >= min_len]


def finalize(cands: list[Candidate], duration: float, merge_gap: float, min_duration: float,
             enabled: tuple) -> list[list]:
    """Per class: clip to the video, merge overlapping/adjacent segments, drop blips."""
    out = []
    by_label: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.label in enabled:
            by_label.setdefault(c.label, []).append(c)
    for label, cs in by_label.items():
        iv = [(max(0.0, c.start), min(duration, c.end)) for c in cs if c.end > c.start]
        for s, e in merge_intervals(iv, merge_gap):
            if e - s >= min_duration:
                out.append([round(s, 2), round(e, 2), label])
    out.sort(key=lambda x: (x[0], x[2]))
    return out
