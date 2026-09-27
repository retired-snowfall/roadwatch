"""Exploratory statistics and pictures of a video (used by the website's EDA section)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pipeline import Analysis
from .scene import SceneModel

GROUPS = ("vehicle", "two_wheeler", "person", "animal")


def video_stats(an: Analysis, bin_sec: float = 10.0) -> dict:
    info = an.info
    per_bin = int(np.ceil(info.duration / bin_sec)) or 1
    edges = np.arange(per_bin + 1) * bin_sec
    counts = {g: np.zeros(per_bin, int) for g in GROUPS}
    for tr in an.tracks:
        if tr.group in counts:
            b0, b1 = int(tr.start // bin_sec), int(min(tr.end, info.duration - 1e-6) // bin_sec)
            counts[tr.group][b0:b1 + 1] += 1
    frames = an.context.index.frames
    simultaneous = {g: [] for g in GROUPS}
    for f in frames:
        c = {g: 0 for g in GROUPS}
        for ti, _ in an.context.index.by_frame[int(f)]:
            g = an.tracks[ti].group
            if g in c:
                c[g] += 1
        for g in GROUPS:
            simultaneous[g].append(c[g])
    veh = [tr for tr in an.tracks if tr.group == "vehicle"]
    moving = np.concatenate([tr.speed[(tr.speed > 0.3) & ~tr.edge] for tr in veh]) if veh else np.zeros(0)
    still_time = sum(float(np.sum(np.diff(tr.t)[(tr.speed[:-1] < 0.12)])) for tr in veh if len(tr.t) > 1)
    total_time = sum(tr.duration for tr in veh) or 1.0
    hist, hedges = np.histogram(moving, bins=np.linspace(0, 6, 25))
    appearance = an.appearance
    light = []
    if appearance is not None:
        light = [[round(t, 1), round(float(g.mean()), 1)] for t, g in zip(appearance.times, appearance.grays)]
    bright = np.mean([v for _, v in light]) if light else float("nan")
    return {
        "video": info.name, "width": info.width, "height": info.height, "fps": round(info.fps, 3),
        "duration": round(info.duration, 2), "n_frames": info.n_frames,
        "size_mb": round(Path(info.path).stat().st_size / 1e6, 1),
        "bitrate_mbps": round(Path(info.path).stat().st_size * 8 / 1e6 / max(info.duration, 1e-6), 2),
        # grey mean of whole frames; asphalt keeps a sunlit junction around 90-100
        "lighting": "night" if bright < 30 else "dusk/overcast" if bright < 80 else "day",
        "mean_brightness": None if np.isnan(bright) else round(float(bright), 1),
        "brightness": light,
        "bins": edges[:-1].tolist(), "bin_sec": bin_sec,
        "tracks_per_bin": {g: c.tolist() for g, c in counts.items()},
        "unique_tracks": {g: int(sum(1 for tr in an.tracks if tr.group == g)) for g in GROUPS},
        "mean_simultaneous": {g: round(float(np.mean(v)), 2) if v else 0.0 for g, v in simultaneous.items()},
        "max_simultaneous": {g: int(np.max(v)) if v else 0 for g, v in simultaneous.items()},
        "speed_hist": {"edges": np.round(hedges, 2).tolist(), "counts": hist.tolist()},
        "vehicle_stationary_share": round(still_time / total_time, 3),
        "events_per_class": {lab: sum(1 for e in an.events if e[2] == lab) for lab in sorted({e[2] for e in an.events})},
        "timings": an.timings,
    }


# ---------------------------------------------------------------------- pictures
def _dim(bg: np.ndarray, factor: float = 0.45) -> np.ndarray:
    return (bg.astype(np.float32) * factor).astype(np.uint8)


def heatmap(bg: np.ndarray, points: np.ndarray, sigma_frac: float = 0.012) -> np.ndarray:
    """Density of ground points as a colour overlay on the dimmed background."""
    h, w = bg.shape[:2]
    acc = np.zeros((h, w), np.float32)
    if len(points):
        p = np.round(points).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        np.add.at(acc, (p[ok, 1], p[ok, 0]), 1.0)
    s = max(3, int(sigma_frac * w))
    acc = cv2.GaussianBlur(acc, (0, 0), s)
    if acc.max() > 0:
        acc = np.sqrt(acc / acc.max())
    col = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(acc * 1.4, 0, 1)[..., None]
    return (_dim(bg) * (1 - alpha) + col * alpha).astype(np.uint8)


def trajectories(bg: np.ndarray, tracks: list, group: str = "vehicle") -> np.ndarray:
    """Every trajectory, coloured by its travel direction (hue wheel)."""
    img = _dim(bg, 0.35)
    for tr in tracks:
        if tr.group != group or len(tr.t) < 3:
            continue
        mv = tr.speed > 0.3
        if mv.sum() < 2:
            continue
        v = tr.vel[mv].mean(axis=0)
        hue = int(((np.degrees(np.arctan2(v[1], v[0])) + 360) % 360) / 2)
        color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(img, [tr.foot.astype(np.int32)], False, color, 2, cv2.LINE_AA)
    return img


def direction_field(bg: np.ndarray, scene: SceneModel) -> np.ndarray:
    """Learned lane direction per grid cell (arrows) over the carriageway mask."""
    img = _dim(bg, 0.5)
    d = scene.derived
    h, w = bg.shape[:2]
    gh, gw = d["road"].shape
    cw, ch = w / gw, h / gh
    road = cv2.resize(d["road"].astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[road > 0] = (overlay[road > 0] * 0.6 + np.array([90, 60, 20]) * 0.4).astype(np.uint8)
    img = overlay
    for y in range(gh):
        for x in range(gw):
            if not d["oriented"][y, x]:
                continue
            a = np.radians(d["direction"][y, x])
            c = np.array([(x + 0.5) * cw, (y + 0.5) * ch])
            tip = c + 0.45 * cw * np.array([np.cos(a), np.sin(a)])
            hue = int(((np.degrees(a) + 360) % 360) / 2)
            color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.arrowedLine(img, tuple((2 * c - tip).astype(int)), tuple(tip.astype(int)), color, 2,
                            cv2.LINE_AA, tipLength=0.45)
    for sl in scene.stop_lines():
        cv2.line(img, tuple(np.asarray(sl["a"]).astype(int)), tuple(np.asarray(sl["b"]).astype(int)),
                 (60, 60, 255), 3, cv2.LINE_AA)
    return img


def write_pictures(an: Analysis, out_dir: Path, width: int = 960) -> dict[str, str]:
    if an.background is None:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = an.background
    scale = width / bg.shape[1]
    small = cv2.resize(bg, (width, int(bg.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    def pts(group: str, moving: bool | None = None) -> np.ndarray:
        arr = []
        for tr in an.tracks:
            if tr.group != group:
                continue
            keep = ~tr.edge
            if moving is not None:
                keep &= (tr.speed > 0.3) if moving else (tr.speed < 0.12)
            arr.append(tr.foot[keep])
        return np.concatenate(arr) * scale if arr else np.zeros((0, 2))

    from .tracks import Track
    scaled = []
    for tr in an.tracks:
        c = Track(tr.tid, tr.group, tr.cls, tr.fidx, tr.t, tr.box * scale, tr.conf)
        c.foot, c.vel, c.speed = tr.foot * scale, tr.vel, tr.speed
        scaled.append(c)
    pictures = {
        "background": small,
        "heat_vehicles": heatmap(small, pts("vehicle", moving=True)),
        "heat_stops": heatmap(small, pts("vehicle", moving=False)),
        "heat_people": heatmap(small, pts("person")),
        "trajectories": trajectories(small, scaled),
        "directions": direction_field(bg, an.scene),
    }
    out = {}
    for name, img in pictures.items():
        if img.shape[1] != width:
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out[name] = path.name
    return out
