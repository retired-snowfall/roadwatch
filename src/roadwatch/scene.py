"""Scene model of the fixed camera, learned from tracks.

The camera never moves, so everything that depends on the road layout is learned
once from the sample videos (the *prior*, stored in weights/scene.json) and
refined with the statistics of the video being analysed:

* carriageway mask: cells that many distinct cars pass through;
* lane direction field: per cell histogram of vehicle headings (one vote per vehicle);
* stop likelihood: seconds of stationary vehicles per cell (signal queues, parking);
* queue heads: where the first vehicle of a queue stops (-> stop lines);
* movements: entry/exit statistics for turn analysis;
* zones: polygons/polylines (crosswalks, stop lines, solid lines, ...) that are
  either auto-suggested or drawn in the scene editor; coordinates are normalised to [0, 1].
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import KinematicsCfg, SceneCfg
from .tracks import Track

ZONE_KINDS = ("crosswalks", "stop_lines", "solid_lines", "intersection", "u_turn_allowed",
              "no_turn", "ignore", "lights")
THUMB = (96, 54)


def empty_zones() -> dict:
    return {k: [] for k in ZONE_KINDS}


def angle_diff(a, b):
    """Absolute difference of angles in degrees, in [0, 180]."""
    d = (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
    return np.abs(d)


def region_of(x: float, y: float, w: int, h: int, n: int = 4) -> int:
    return int(min(n - 1, max(0, y / h * n))) * n + int(min(n - 1, max(0, x / w * n)))


def movement_key(tr: Track, w: int, h: int) -> tuple | None:
    """(entry region, entry heading bin, exit region, exit heading bin) of a moving vehicle track."""
    moving = np.where(tr.speed > 0.5)[0]
    if len(moving) < 4:
        return None
    a, b = moving[: max(2, len(moving) // 6)], moving[-max(2, len(moving) // 6):]
    h0 = np.degrees(np.arctan2(tr.vel[a, 1].mean(), tr.vel[a, 0].mean()))
    h1 = np.degrees(np.arctan2(tr.vel[b, 1].mean(), tr.vel[b, 0].mean()))
    return (region_of(*tr.foot[a[0]], w, h), int(((h0 + 360) % 360) // 45),
            region_of(*tr.foot[b[-1]], w, h), int(((h1 + 360) % 360) // 45))


@dataclass
class SceneModel:
    width: int
    height: int
    cfg: SceneCfg = field(default_factory=SceneCfg)
    seconds: float = 0.0
    n_videos: int = 0
    veh_count: np.ndarray = None
    veh_dir: np.ndarray = None
    ped_count: np.ndarray = None
    wheel_count: np.ndarray = None
    stop_time: np.ndarray = None
    queue_heads: list = field(default_factory=list)   # [x_n, y_n, dir_deg, size_n]
    movements: Counter = field(default_factory=Counter)
    zones: dict = field(default_factory=empty_zones)
    thumb: np.ndarray = None                           # grey background thumbnail (scene identity)

    def __post_init__(self):
        gh, gw, nb = self.cfg.grid_h, self.cfg.grid_w, self.cfg.dir_bins
        if self.veh_count is None:
            self.veh_count = np.zeros((gh, gw))
            self.veh_dir = np.zeros((gh, gw, nb))
            self.ped_count = np.zeros((gh, gw))
            self.wheel_count = np.zeros((gh, gw))
            self.stop_time = np.zeros((gh, gw))
        self._derived = None

    # ------------------------------------------------------------------ geometry
    def cells(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        gx = np.clip((pts[:, 0] / self.width * self.cfg.grid_w).astype(int), 0, self.cfg.grid_w - 1)
        gy = np.clip((pts[:, 1] / self.height * self.cfg.grid_h).astype(int), 0, self.cfg.grid_h - 1)
        return gy, gx

    def cell_px(self) -> float:
        return self.width / self.cfg.grid_w

    def denorm(self, pts) -> np.ndarray:
        return np.asarray(pts, np.float64).reshape(-1, 2) * [self.width, self.height]

    # ------------------------------------------------------------------ learning
    def accumulate(self, tracks: list[Track], duration: float, kin: KinematicsCfg,
                   background: np.ndarray | None = None) -> None:
        nb = self.cfg.dir_bins
        frame_feet: dict[int, list] = {}
        for tr in tracks:
            if tr.group in ("vehicle", "two_wheeler"):
                for i, f in enumerate(tr.fidx):
                    frame_feet.setdefault(int(f), []).append((tr.tid, tr.foot[i], tr.size[i]))
        for tr in tracks:
            ok = ~tr.edge
            if tr.group == "person":
                gy, gx = self.cells(tr.foot[ok])
                u = np.unique(gy * self.cfg.grid_w + gx)
                np.add.at(self.ped_count, np.unravel_index(u, self.ped_count.shape), 1)
                continue
            if tr.group not in ("vehicle", "two_wheeler"):
                continue
            moving = ok & (tr.speed > kin.moving_speed)
            pts, vels = self._densify(tr, moving)
            gy, gx = self.cells(pts)
            flat = gy * self.cfg.grid_w + gx
            for c in np.unique(flat):
                sel = flat == c
                v = vels[sel].mean(axis=0)
                cy, cx = divmod(int(c), self.cfg.grid_w)
                if tr.group == "vehicle":
                    self.veh_count[cy, cx] += 1
                    b = int(((np.degrees(np.arctan2(v[1], v[0])) + 360) % 360) / (360 / nb)) % nb
                    self.veh_dir[cy, cx, b] += 1
                else:
                    self.wheel_count[cy, cx] += 1
            if tr.group == "vehicle":
                still = ok & (tr.speed < kin.stationary_speed)
                dt = np.clip(np.diff(tr.t, append=tr.t[-1]), 0, 1.0)
                gy, gx = self.cells(tr.foot[still])
                np.add.at(self.stop_time, (gy, gx), dt[still])
                self.queue_heads.extend(self._queue_heads(tr, frame_feet, kin))
                key = movement_key(tr, self.width, self.height)
                if key is not None:
                    self.movements[key] += 1
        self.seconds += duration
        self.n_videos += 1
        if background is not None:
            self.thumb = thumbnail(background)
        self._derived = None

    def _densify(self, tr: Track, mask: np.ndarray, max_gap: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """Foot points (and velocities) along the path, at most half a cell apart.

        A car near the camera moves several cells between analysed frames; voting only at
        the samples would leave holes in the carriageway and direction maps.
        """
        step = self.cell_px() / 2
        pts, vels = [tr.foot[mask]], [tr.vel[mask]]
        idx = np.where(mask[:-1] & mask[1:] & (np.diff(tr.t) <= max_gap))[0]
        for i in idx:
            a, b = tr.foot[i], tr.foot[i + 1]
            n = int(np.linalg.norm(b - a) // step)
            if n >= 1:
                f = (np.arange(1, n + 1) / (n + 1))[:, None]
                pts.append(a + f * (b - a))
                vels.append(np.repeat(((tr.vel[i] + tr.vel[i + 1]) / 2)[None], n, axis=0))
        return np.concatenate(pts), np.concatenate(vels)

    def _queue_heads(self, tr: Track, frame_feet: dict, kin: KinematicsCfg) -> list:
        """Front points where this vehicle waited 3-180 s with nobody right ahead.

        It must have driven in and must drive off again: parked cars also stand with nobody
        ahead, but never arrive or leave, and would otherwise invent stop lines.
        Rows: [x, y, travel dir, size, t, video index, track id] (x, y, size normalised).
        """
        heads = []
        still = tr.speed < kin.stationary_speed
        for s, e in runs(still):
            wait = tr.t[e] - tr.t[s]
            if not (3.0 <= wait <= 180.0) or not tr.arrived_moving(s) or not tr.departs_moving(e):
                continue
            before = tr.window(tr.t[s] - 3.0, tr.t[s])
            v = tr.foot[before][-1] - tr.foot[before][0]
            d = v / (np.linalg.norm(v) + 1e-9)
            m = (s + e) // 2
            foot, size = tr.foot[m], tr.size[m]
            blocked = False
            for tid, f2, _ in frame_feet.get(int(tr.fidx[m]), ()):
                if tid == tr.tid:
                    continue
                rel = f2 - foot
                along, across = rel @ d, abs(rel @ np.array([-d[1], d[0]]))
                if 0 < along < 2.5 * size and across < 0.6 * size:
                    blocked = True
                    break
            if not blocked:
                front = foot + d * 0.5 * size
                heads.append([front[0] / self.width, front[1] / self.height,
                              float(np.degrees(np.arctan2(d[1], d[0]))), size / self.width,
                              float(tr.t[m]), self.n_videos, tr.tid])
        return heads

    def merged_with(self, other: "SceneModel") -> "SceneModel":
        out = SceneModel(self.width, self.height, self.cfg)
        for name in ("veh_count", "veh_dir", "ped_count", "wheel_count", "stop_time"):
            setattr(out, name, getattr(self, name) + getattr(other, name))
        out.seconds = self.seconds + other.seconds
        out.n_videos = self.n_videos + other.n_videos
        out.queue_heads = self.queue_heads + other.queue_heads
        out.movements = self.movements + other.movements
        out.zones = {k: (self.zones.get(k) or other.zones.get(k) or []) for k in ZONE_KINDS}
        out.thumb = self.thumb if self.thumb is not None else other.thumb
        return out

    # ------------------------------------------------------------------ derived maps
    @property
    def derived(self) -> dict:
        if self._derived is None:
            self._derived = self._derive()
        return self._derived

    def _derive(self) -> dict:
        cfg = self.cfg
        road = (self.veh_count >= cfg.road_min_vehicles).astype(np.uint8)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        hist = self.veh_dir
        smooth = hist + 0.5 * (np.roll(hist, 1, axis=2) + np.roll(hist, -1, axis=2))
        mode = smooth.argmax(axis=2)
        nb = cfg.dir_bins
        idx = np.stack([(mode - 1) % nb, mode, (mode + 1) % nb], axis=2)
        near = np.take_along_axis(hist, idx, axis=2)
        total = hist.sum(axis=2)
        purity = np.where(total > 0, near.sum(axis=2) / np.maximum(total, 1e-9), 0.0)
        centers = (idx + 0.5) * (360 / nb)
        vx = (near * np.cos(np.radians(centers))).sum(axis=2)
        vy = (near * np.sin(np.radians(centers))).sum(axis=2)
        direction = np.degrees(np.arctan2(vy, vx))
        oriented = (total >= cfg.oriented_min_vehicles) & (purity >= cfg.oriented_min_purity)
        # How often vehicles wait in a cell compared with how many pass: signal queues and
        # parking spots are "normal places to stop"; the rest of the carriageway is not.
        per_hour = self.stop_time / max(self.seconds / 3600.0, 1e-6)
        stop_norm = per_hour / np.maximum(self.veh_count / max(self.seconds / 3600.0, 1e-6) * 2.0, 60.0)
        return {"road": road.astype(bool), "direction": direction, "purity": purity,
                "oriented": oriented, "stop_norm": stop_norm, "streams": self._streams(oriented, direction)}

    @staticmethod
    def _streams(oriented: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Label connected groups of oriented cells that share a travel direction (-1 = none)."""
        lab = -np.ones(oriented.shape, int)
        n = 0
        gh, gw = oriented.shape
        for y0 in range(gh):
            for x0 in range(gw):
                if not oriented[y0, x0] or lab[y0, x0] >= 0:
                    continue
                stack, members = [(y0, x0)], []
                lab[y0, x0] = n
                while stack:
                    y, x = stack.pop()
                    members.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy, x + dx
                            if (0 <= yy < gh and 0 <= xx < gw and oriented[yy, xx] and lab[yy, xx] < 0
                                    and angle_diff(direction[yy, xx], direction[y, x]) < 40):
                                lab[yy, xx] = n
                                stack.append((yy, xx))
                if len(members) < 6:
                    for y, x in members:
                        lab[y, x] = -2
                else:
                    n += 1
        lab[lab == -2] = -1
        return lab

    # ------------------------------------------------------------------ queries
    def on_road(self, pts: np.ndarray, erode: int = 0) -> np.ndarray:
        road = self.derived["road"].astype(np.uint8)
        if erode:
            road = cv2.erode(road, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
        gy, gx = self.cells(pts)
        ok = road[gy, gx].astype(bool)
        if self.zones.get("ignore"):
            ok &= ~self.in_zones("ignore", pts)
        return ok

    def lane_direction(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(direction deg, valid) of the learned traffic flow at each point."""
        d = self.derived
        gy, gx = self.cells(pts)
        return d["direction"][gy, gx], d["oriented"][gy, gx]

    def stream_of(self, pts: np.ndarray) -> np.ndarray:
        gy, gx = self.cells(pts)
        return self.derived["streams"][gy, gx]

    def in_polygon(self, poly, pts: np.ndarray) -> np.ndarray:
        """Points (pixels) inside one normalised polygon (list of [x, y] or {"points": [...]})."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        p = poly["points"] if isinstance(poly, dict) else poly
        cnt = self.denorm(p).astype(np.float32).reshape(-1, 1, 2)
        if len(cnt) < 3:
            return np.zeros(len(pts), bool)
        return np.array([cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0 for x, y in pts], bool)

    def in_zones(self, kind: str, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        inside = np.zeros(len(pts), bool)
        for poly in self.zones.get(kind, []):
            inside |= self.in_polygon(poly, pts)
        return inside

    def in_crossing(self, pts: np.ndarray) -> np.ndarray:
        """Pedestrian crossings: drawn crosswalks, else carriageway cells most pedestrians use."""
        if self.zones.get("crosswalks"):
            return self.in_zones("crosswalks", pts)
        road = self.derived["road"]
        peds = np.where(road, self.ped_count, 0)
        if peds.max() < 5:
            return np.zeros(len(np.asarray(pts).reshape(-1, 2)), bool)
        cells = (peds >= max(5.0, 0.3 * peds.max())).astype(np.uint8)
        cells = cv2.dilate(cells, np.ones((3, 3), np.uint8)).astype(bool)
        gy, gx = self.cells(pts)
        return cells[gy, gx]

    def stop_lines(self) -> list[dict]:
        """Stop lines in pixels: [{"a": (x, y), "b": (x, y), "dir": deg}]; drawn ones win over learned ones."""
        drawn = self.zones.get("stop_lines") or []
        if drawn:
            out = []
            for sl in drawn:
                a, b = self.denorm(sl["points"][:2])
                out.append({"a": a, "b": b, "dir": float(sl.get("dir", _perp_dir(a, b)))})
            return out
        return self._learned_stop_lines()

    def _learned_stop_lines(self, min_heads: int = 4) -> list[dict]:
        if len(self.queue_heads) < min_heads:
            return []
        heads = np.array([q for q in self.queue_heads if len(q) == 7], np.float64)
        if len(heads) < min_heads:
            return []
        pts = heads[:, :2] * [self.width, self.height]
        dirs = heads[:, 2]
        sizes = heads[:, 3] * self.width
        used = np.zeros(len(heads), bool)
        lines = []
        for i in np.argsort(-sizes):
            if used[i]:
                continue
            d = np.radians(dirs[i])
            u = np.array([np.cos(d), np.sin(d)])
            n = np.array([-u[1], u[0]])
            rel = pts - pts[i]
            member = (~used) & (angle_diff(dirs, dirs[i]) < 30) & (np.abs(rel @ u) < 2.0 * sizes[i]) \
                & (np.abs(rel @ n) < 6.0 * sizes[i])
            # a stop line is where different vehicles wait in different signal cycles
            vehicles = {(int(v), int(k)) for v, k in heads[member][:, 5:7]}
            cycles = {(int(v), int(t // 20.0)) for t, v in heads[member][:, 4:6]}
            if member.sum() < min_heads or len(vehicles) < 3 or len(cycles) < 3:
                continue
            used |= member
            along = rel[member] @ u
            across = rel[member] @ n
            s0 = np.percentile(along, 80)
            half = max(np.ptp(across) / 2 + 0.6 * np.median(sizes[member]), sizes[i])
            mid = pts[i] + u * s0 + n * (across.max() + across.min()) / 2
            lines.append({"a": mid - n * half, "b": mid + n * half, "dir": float(np.degrees(d)),
                          "support": int(member.sum())})
        return lines

    def similar_to(self, other: "SceneModel", min_corr: float = 0.6) -> bool:
        """Same camera and angle? Resolution must match and background thumbnails correlate."""
        if (self.width, self.height) != (other.width, other.height):
            return False
        if self.thumb is None or other.thumb is None:
            return True
        a = _edges(self.thumb)
        b = _edges(other.thumb)
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) >= min_corr

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        return {
            "width": self.width, "height": self.height, "seconds": self.seconds, "n_videos": self.n_videos,
            "grid": [self.cfg.grid_w, self.cfg.grid_h, self.cfg.dir_bins],
            "veh_count": self.veh_count.round(2).tolist(), "veh_dir": self.veh_dir.round(2).tolist(),
            "ped_count": self.ped_count.round(2).tolist(), "wheel_count": self.wheel_count.round(2).tolist(),
            "stop_time": self.stop_time.round(2).tolist(),
            "queue_heads": [[round(v, 5) for v in q] for q in self.queue_heads],
            "movements": [[list(k), v] for k, v in sorted(self.movements.items())],
            "zones": self.zones,
            "thumb": None if self.thumb is None else self.thumb.astype(int).tolist(),
        }

    @classmethod
    def from_json(cls, d: dict, cfg: SceneCfg | None = None) -> "SceneModel":
        gw, gh, nb = d["grid"]
        cfg = cfg or SceneCfg()
        cfg = SceneCfg(**{**cfg.__dict__, "grid_w": gw, "grid_h": gh, "dir_bins": nb})
        zones = empty_zones()
        zones.update(d.get("zones") or {})
        return cls(d["width"], d["height"], cfg, d.get("seconds", 0.0), d.get("n_videos", 0),
                   np.array(d["veh_count"]), np.array(d["veh_dir"]), np.array(d["ped_count"]),
                   np.array(d["wheel_count"]), np.array(d["stop_time"]), list(d.get("queue_heads", [])),
                   Counter({tuple(k): v for k, v in d.get("movements", [])}), zones,
                   None if d.get("thumb") is None else np.array(d["thumb"], np.uint8))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path: str | Path) -> "SceneModel | None":
        """Learned statistics from `path`; hand-drawn zones from zones.json next to it override."""
        p = Path(path)
        if not p.exists():
            return None
        scene = cls.from_json(json.loads(p.read_text()))
        drawn = p.with_name("zones.json")
        if drawn.exists():
            for kind, items in json.loads(drawn.read_text()).items():
                if kind in ZONE_KINDS and items:
                    scene.zones[kind] = items
        return scene


# ---------------------------------------------------------------------- helpers
def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def thumbnail(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA)


def _edges(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def _perp_dir(a: np.ndarray, b: np.ndarray) -> float:
    """Default travel direction for a drawn stop line: perpendicular, pointing up the image."""
    v = b - a
    n = np.array([-v[1], v[0]])
    if n[1] > 0:
        n = -n
    return float(np.degrees(np.arctan2(n[1], n[0])))


def median_background(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
