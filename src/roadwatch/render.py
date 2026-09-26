"""Annotated video rendering (boxes, trails, scene layers, active events, risk gauge)."""
from __future__ import annotations

import bisect
from pathlib import Path

import cv2
import numpy as np

from .constants import CLASSES

GROUP_COLORS = {"vehicle": (255, 170, 60), "two_wheeler": (60, 200, 255), "person": (80, 230, 80),
                "animal": (200, 80, 255)}
# one colour per event class (BGR), shared with the website palette
EVENT_COLORS = {c: tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(i * 180 / len(CLASSES)), 200, 255]]]),
                                                      cv2.COLOR_HSV2BGR)[0, 0]) for i, c in enumerate(CLASSES)}


class TrackLookup:
    """Box of every track at an arbitrary time (linear interpolation between analysed frames)."""

    def __init__(self, tracks: list[dict]):
        self.tracks = [(tr["id"], tr["group"], np.asarray(tr["t"]), np.asarray(tr["box"], float)) for tr in tracks]

    def at(self, t: float, max_gap: float = 0.6):
        for tid, group, ts, boxes in self.tracks:
            if len(ts) == 0 or t < ts[0] - 0.05 or t > ts[-1] + 0.05:
                continue
            k = bisect.bisect_left(ts.tolist(), t)
            if k == 0 or k >= len(ts):
                k = min(max(k, 0), len(ts) - 1)
                yield tid, group, boxes[k], ts, boxes, k
                continue
            t0, t1 = ts[k - 1], ts[k]
            if t1 - t0 > max_gap:
                continue
            a = (t - t0) / max(t1 - t0, 1e-6)
            yield tid, group, boxes[k - 1] * (1 - a) + boxes[k] * a, ts, boxes, k


def draw_scene(img: np.ndarray, scene: dict, scale: float) -> None:
    for sl in scene.get("stop_lines", []):
        a = (np.asarray(sl["a"]) * scale).astype(int)
        b = (np.asarray(sl["b"]) * scale).astype(int)
        cv2.line(img, tuple(a), tuple(b), (60, 60, 255), 2)
    for kind, color in (("crosswalks", (255, 255, 255)), ("intersection", (200, 200, 200))):
        for poly in scene.get("zones", {}).get(kind, []) or []:
            pts = poly["points"] if isinstance(poly, dict) else poly
            p = (np.asarray(pts) * [img.shape[1], img.shape[0]]).astype(np.int32)
            cv2.polylines(img, [p], True, color, 1)
    for line in scene.get("zones", {}).get("solid_lines", []) or []:
        pts = line["points"] if isinstance(line, dict) else line
        p = (np.asarray(pts) * [img.shape[1], img.shape[0]]).astype(np.int32)
        cv2.polylines(img, [p], False, (0, 220, 255), 2)


def draw_frame(frame: np.ndarray, t: float, lookup: TrackLookup, events: list, risk: float | None,
               highlight: dict, scene: dict | None = None, scale: float = 1.0, resize: bool = True) -> np.ndarray:
    """Draw overlays; `scale` maps working pixels to the output. resize=False: frame is already output-sized."""
    img = frame if scale == 1.0 or not resize else cv2.resize(frame, None, fx=scale, fy=scale,
                                                             interpolation=cv2.INTER_AREA)
    if scene:
        draw_scene(img, scene, scale)
    active = [e for e in events if e[0] <= t <= e[1]]
    hot = {tid: lab for lab, tids in highlight.items() for tid in tids if any(e[2] == lab for e in active)}
    for tid, group, box, ts, boxes, k in lookup.at(t):
        color = EVENT_COLORS[hot[tid]] if tid in hot else GROUP_COLORS.get(group, (200, 200, 200))
        x1, y1, x2, y2 = (box * scale).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if tid in hot else 1)
        cv2.putText(img, str(tid), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        lo = max(0, k - 12)
        trail = np.c_[(boxes[lo:k + 1, 0] + boxes[lo:k + 1, 2]) / 2, boxes[lo:k + 1, 3]] * scale
        if len(trail) > 1:
            cv2.polylines(img, [trail.astype(np.int32)], False, color, 1)
    y = 22
    cv2.putText(img, f"t={t:6.1f}s", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    for s, e, lab in active:
        y += 22
        cv2.putText(img, lab.replace("_", " ").upper(), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    EVENT_COLORS[lab], 2, cv2.LINE_AA)
    if risk is not None:
        h, w = img.shape[:2]
        bar = int((w - 20) * risk)
        cv2.rectangle(img, (10, h - 16), (w - 10, h - 8), (60, 60, 60), -1)
        col = (0, 0, 255) if risk >= 0.5 else (0, 200, 255) if risk >= 0.2 else (0, 200, 0)
        cv2.rectangle(img, (10, h - 16), (10 + bar, h - 8), col, -1)
        cv2.putText(img, f"accident risk {risk:.2f}", (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
    return img


def render_video(video_path: str, result: dict, out_path: str | Path, risk: list | None = None,
                 max_width: int = 960, fps_out: float | None = None) -> Path:
    """Write an H.264 mp4 (browser-playable) with overlays, using the ffmpeg bundled by imageio-ffmpeg."""
    import imageio_ffmpeg

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, max_width / w)
    size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
    scale = size[0] / float(result.get("width") or w)  # result coordinates are in working pixels
    fps_out = fps_out or fps
    step = max(1, int(round(fps / fps_out)))
    writer = imageio_ffmpeg.write_frames(str(out_path), size, fps=fps / step, codec="libx264",
                                         pix_fmt_in="bgr24", output_params=["-crf", "28", "-preset", "veryfast",
                                                                            "-movflags", "+faststart"])
    writer.send(None)
    lookup = TrackLookup(result["tracks"])
    highlight: dict = {}
    for c in result.get("candidates", []):
        highlight.setdefault(c["label"], set()).update(c["tracks"])
    risk_t = np.array([r[0] for r in risk]) if risk else None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            r = None
            if risk_t is not None and len(risk_t):
                r = risk[min(int(np.searchsorted(risk_t, t)), len(risk) - 1)][1]
            img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            img = draw_frame(img, t, lookup, result["events"], r, highlight, result.get("scene"), scale, resize=False)
            writer.send(np.ascontiguousarray(img))
        idx += 1
    writer.close()
    cap.release()
    return Path(out_path)


def transcode_preview(src: str | Path, dst: str | Path, max_width: int = 960) -> Path:
    """Browser-safe H.264 copy without overlays (the website draws overlays on a canvas)."""
    import subprocess

    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"scale='min({max_width},iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
           "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return Path(dst)
