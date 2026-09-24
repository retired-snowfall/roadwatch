"""Learn the scene prior of the fixed camera from the sample videos.

    python tools/calibrate.py --videos samples

Writes weights/scene.json (learned statistics: carriageway, lane directions, stop zones,
queue heads, movements) and weights/scene_background.jpg (median background, used by the
scene editor and the website). Hand-drawn zones live in weights/zones.json and are never
touched here.
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

from roadwatch.config import CFG, WEIGHTS_DIR  # noqa: E402
from roadwatch.pipeline import perceive  # noqa: E402
from roadwatch.scene import SceneModel  # noqa: E402
from roadwatch.tracks import build_tracks  # noqa: E402
from roadwatch.video import probe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder with the sample .mp4 files")
    ap.add_argument("--out", default=str(WEIGHTS_DIR / "scene.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ.setdefault("ROADWATCH_CACHE", str(ROOT / ".cache" / "perception"))

    videos = sorted(Path(args.videos).glob("*.mp4"))
    if not videos:
        sys.exit(f"no .mp4 in {args.videos}")
    scene: SceneModel | None = None
    backgrounds = []
    for p in videos:
        info = probe(str(p))
        per = perceive(info, CFG, lambda stage, f: None)
        tracks = build_tracks(per.records, CFG.kin, info.width, info.height)
        bg = per.appearance.background()
        bg_full = None if bg is None else cv2.resize(bg, (info.width, info.height))
        if bg_full is not None:
            backgrounds.append(bg_full)
        if scene is None:
            scene = SceneModel(info.width, info.height, CFG.scene)
        elif (scene.width, scene.height) != (info.width, info.height):
            logging.warning("%s: resolution %dx%d differs from %dx%d, skipped", p.name, info.width,
                            info.height, scene.width, scene.height)
            continue
        scene.accumulate(tracks, info.duration, CFG.kin, bg_full)
        logging.info("%s: %.0fs, %d tracks, %d analysed frames", p.name, info.duration, len(tracks), per.analysed)

    if backgrounds:
        bg = np.median(np.stack(backgrounds), axis=0).astype(np.uint8)
        cv2.imwrite(str(Path(args.out).with_name("scene_background.jpg")), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        from roadwatch.scene import thumbnail
        scene.thumb = thumbnail(bg)
    scene.save(args.out)
    d = scene.derived
    print(f"wrote {args.out}: {scene.n_videos} videos, {scene.seconds:.0f}s, "
          f"carriageway cells {int(d['road'].sum())}, oriented {int(d['oriented'].sum())}, "
          f"streams {int(d['streams'].max()) + 1}, queue heads {len(scene.queue_heads)}, "
          f"learned stop lines {len(scene._learned_stop_lines())}, movements {sum(scene.movements.values())}")


if __name__ == "__main__":
    main()
