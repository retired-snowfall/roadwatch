"""Turn plain-text labels into the official ground-truth JSON.

    python tools/labels_from_text.py labels/team_labels.txt --videos samples/1080p --out labels/dev_labels.json

Input: a video name on its own line ("C3896"), then one event per line, "m:ss - m:ss class"
(seconds may have decimals; anything after the class is ignored). Same-class segments that
overlap in one video are merged into one, as the official format requires; segments that only
touch stay separate. Ends are clipped to the video duration, read from the videos in --videos.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluate import OFFICIAL_CLASSES  # noqa: E402  (official metric, unchanged)
from roadwatch.video import probe  # noqa: E402

VIDEO = re.compile(r"^\s*[CС](\d{3,})(?:\.mp4)?\s*$", re.IGNORECASE)   # Latin or Cyrillic C
TIME = r"(\d+):(\d{1,2}(?:\.\d+)?)"
EVENT = re.compile(rf"^\s*{TIME}\s*[-–—]\s*{TIME}\s+([a-z_]+)")


def parse(text: str) -> tuple[dict[str, list], list[str]]:
    """{video name: [[start, end, class], ...]} and a list of lines that could not be read."""
    videos: dict[str, list] = {}
    problems: list[str] = []
    current = None
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = VIDEO.match(line)
        if m:
            current = f"C{m.group(1)}.MP4"
            videos.setdefault(current, [])
            continue
        m = EVENT.match(line)
        if m is None or current is None:
            problems.append(f"line {n}: not understood: {line.strip()!r}")
            continue
        s = int(m.group(1)) * 60 + float(m.group(2))
        e = int(m.group(3)) * 60 + float(m.group(4))
        label = m.group(5)
        if label not in OFFICIAL_CLASSES:
            problems.append(f"line {n}: unknown class {label!r}")
        elif e <= s:
            problems.append(f"line {n}: ends before it starts")
        else:
            videos[current].append([s, e, label])
    return videos, problems


def merge_same_class(events: list) -> tuple[list, int]:
    """Merge overlapping segments of one class; returns the events and how many merges were made."""
    out, merged = [], 0
    for label in sorted({lab for _, _, lab in events}):
        segs = sorted((s, e) for s, e, lab in events if lab == label)
        cur = list(segs[0])
        for s, e in segs[1:]:
            if s < cur[1]:
                cur[1] = max(cur[1], e)
                merged += 1
            else:
                out.append([cur[0], cur[1], label])
                cur = [s, e]
        out.append([cur[0], cur[1], label])
    return sorted(out), merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--videos", required=True, help="folder with the videos (duration and fps)")
    ap.add_argument("--out", default=str(ROOT / "labels" / "dev_labels.json"))
    args = ap.parse_args()

    videos, problems = parse(Path(args.text).read_text(encoding="utf-8"))
    for p in problems:
        print("!", p)
    files = {q.name.upper(): q for q in Path(args.videos).iterdir() if q.suffix.lower() == ".mp4"}
    gt = {}
    for name, events in videos.items():
        if name.upper() not in files:
            sys.exit(f"{name}: not found in {args.videos}")
        info = probe(str(files[name.upper()]))
        events = [[s, min(e, info.duration), lab] for s, e, lab in events if s < info.duration]
        events, merged = merge_same_class(events) if events else ([], 0)
        gt[files[name.upper()].name] = {"duration": round(info.duration, 3), "fps": round(info.fps, 3),
                                        "events": [[round(s, 2), round(e, 2), lab] for s, e, lab in events]}
        counts: dict[str, int] = {}
        for _, _, lab in events:
            counts[lab] = counts.get(lab, 0) + 1
        print(f"{name}: {len(events)} events, {merged} same-class overlaps merged: {counts}")
    Path(args.out).write_text(json.dumps(gt, indent=1) + "\n")
    print(f"wrote {args.out}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
