"""Learn the scene prior of the fixed camera from the sample videos.

    python tools/calibrate.py --videos samples

Writes weights/scene.json (learned statistics: carriageway, lane directions, stop zones,
queue heads, movements) and weights/scene_background.jpg (background of the reference view:
registration target, scene editor and website). The camera is re-aimed between recordings,
so every video is first registered to the reference view (first video, or --reference) and
its trajectories are accumulated in reference coordinates. Hand-drawn zones live in
weights/zones.json (reference view) and are never touched here.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roadwatch import register  # noqa: E402
from roadwatch.config import CFG, WEIGHTS_DIR  # noqa: E402
from roadwatch.pipeline import perceive  # noqa: E402
from roadwatch.scene import SceneModel  # noqa: E402
from roadwatch.tracks import build_tracks  # noqa: E402
from roadwatch.video import probe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder with the sample .mp4 files")
    ap.add_argument("--reference", help="file name of the reference-view video (default: the first)")
    ap.add_argument("--out", default=str(WEIGHTS_DIR / "scene.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ.setdefault("ROADWATCH_CACHE", str(ROOT / ".cache" / "perception"))

    videos = sorted(q for q in Path(args.videos).iterdir() if q.suffix.lower() == ".mp4")
    if not videos:
        sys.exit(f"no .mp4 in {args.videos}")
    if args.reference:
        videos.sort(key=lambda q: q.name != args.reference)   # reference first
    scene: SceneModel | None = None
    ref_bg: np.ndarray | None = None
    for p in videos:
        info = probe(str(p))
        per = perceive(info, CFG, lambda stage, f: None)
        tracks = build_tracks(per.records, CFG.kin, info.work_width, info.work_height)
        bg = per.appearance.background()
        if bg is None:
            logging.warning("%s: no background, skipped", p.name)
            continue
        bg = cv2.resize(bg, info.work)
        if scene is None:
            scene, ref_bg = SceneModel(info.work_width, info.work_height, CFG.scene), bg
            H = None
        else:
            if (scene.width, scene.height) != info.work:
                logging.warning("%s: working size %dx%d differs from %dx%d, skipped", p.name, *info.work,
                                scene.width, scene.height)
                continue
            H = register.homography(bg, ref_bg)
            if H is None:
                logging.warning("%s: could not register to the reference view, skipped", p.name)
                continue
            tracks = register.warp_tracks(tracks, H, CFG.kin, info.work_width, info.work_height)
            bg = register.warp_image(bg, H, info.work)
        scene.accumulate(tracks, info.duration, CFG.kin, bg)
        logging.info("%s: %.0fs, %d tracks, %d analysed frames, %s", p.name, info.duration, len(tracks),
                     per.analysed, "reference view" if H is None else
                     f"registered (shift {H[0, 2]:.0f}, {H[1, 2]:.0f} px)")

    cv2.imwrite(str(Path(args.out).with_name("scene_background.jpg")), ref_bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
    from roadwatch.scene import thumbnail
    scene.thumb = thumbnail(ref_bg)
    scene.save(args.out)
    d = scene.derived
    print(f"wrote {args.out}: {scene.n_videos} videos, {scene.seconds:.0f}s, "
          f"carriageway cells {int(d['road'].sum())}, oriented {int(d['oriented'].sum())}, "
          f"streams {int(d['streams'].max()) + 1}, queue heads {len(scene.queue_heads)}, "
          f"learned stop lines {len(scene._learned_stop_lines())}, movements {sum(scene.movements.values())}")


if __name__ == "__main__":
    main()
