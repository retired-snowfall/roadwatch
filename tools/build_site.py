"""Build the website data for the sample videos.

    python tools/build_site.py --videos samples [--out web/static/data]

For every video: Part A analysis (events + evidence + tracks + scene layers), the Part B
risk curve streamed exactly like run_submission.py, EDA statistics and pictures, a clean
H.264 preview for the interactive player and a burned-in annotated render. Writes
<out>/index.json listing everything, and <out>/scene/*.jpg for the calibrated scene.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roadwatch import eda  # noqa: E402
from roadwatch.config import WEIGHTS_DIR  # noqa: E402
from roadwatch.pipeline import analyze, load_prior  # noqa: E402
from roadwatch.render import render_video, transcode_preview  # noqa: E402
from roadwatch.risk import risk_curve  # noqa: E402


def thin(curve: list, max_points: int = 3000) -> list:
    step = max(1, len(curve) // max_points)
    return curve[::step]


def build_video(path: Path, out_root: Path, skip_media: bool) -> dict:
    out = out_root / "samples" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    an = analyze(str(path))
    result = an.to_json()
    risk = risk_curve(str(path))
    result["risk"] = thin(risk)
    result["eda"] = eda.video_stats(an)
    result["pictures"] = eda.write_pictures(an, out)
    (out / "result.json").write_text(json.dumps(result))
    if not skip_media:
        transcode_preview(path, out / "preview.mp4")
        render_video(str(path), result, out / "annotated.mp4", risk=risk)
    poster = out / "poster.jpg"
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(50, max(0, an.info.n_frames - 1)))
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(str(poster), cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1]))),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"id": path.stem, "video": path.name, "duration": result["duration"], "fps": result["fps"],
            "width": result["width"], "height": result["height"], "events": result["events"],
            "n_events": len(result["events"]), "lighting": result["eda"]["lighting"],
            "max_risk": max((r[1] for r in risk), default=0.0), "timings": result["timings"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default=str(ROOT / "web" / "static" / "data"))
    ap.add_argument("--skip-media", action="store_true", help="JSON and pictures only (fast)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ.setdefault("ROADWATCH_CACHE", str(ROOT / ".cache" / "perception"))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(Path(args.videos).glob("*.mp4"))
    index = {"videos": [build_video(p, out_root, args.skip_media) for p in videos]}

    prior = load_prior()
    if prior is not None:
        scene_dir = out_root / "scene"
        scene_dir.mkdir(exist_ok=True)
        bg_path = WEIGHTS_DIR / "scene_background.jpg"
        if bg_path.exists():
            bg = cv2.imread(str(bg_path))
            cv2.imwrite(str(scene_dir / "directions.jpg"), cv2.resize(eda.direction_field(bg, prior), (960, int(
                bg.shape[0] * 960 / bg.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 85])
            shutil.copy(bg_path, scene_dir / "background.jpg")
        index["scene"] = {"videos": prior.n_videos, "seconds": prior.seconds,
                          "stop_lines": len(prior.stop_lines()), "zones": prior.zones}
    preds = ROOT / "predictions_samples.json"
    if preds.exists():
        shutil.copy(preds, out_root / "predictions_samples.json")
    (out_root / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {out_root / 'index.json'} ({len(videos)} videos)")


if __name__ == "__main__":
    main()
