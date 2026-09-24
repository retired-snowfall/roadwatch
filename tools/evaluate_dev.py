"""Score the pipeline on our own labels of the sample videos, with optional parameter overrides.

    python tools/evaluate_dev.py --videos samples --labels labels/dev_labels.json
    python tools/evaluate_dev.py --videos samples --labels labels/dev_labels.json \\
        --set events.nm_brake=2.0 --set events.jw_min_duration=1.5 --risk

Perception is cached (.cache/perception), so re-scoring after a rule or threshold change
takes seconds. Uses the official evaluate.py for the numbers, then prints which predicted
class overlapped each missed ground-truth event (class confusion).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402  (official metric, unchanged)
from roadwatch.config import CFG  # noqa: E402
from roadwatch.pipeline import analyze  # noqa: E402
from roadwatch.risk import risk_curve  # noqa: E402


def apply_overrides(cfg, items: list[str]):
    cfg = copy.deepcopy(cfg)
    for item in items:
        key, value = item.split("=", 1)
        section, name = key.split(".")
        target = getattr(cfg, section)
        old = getattr(target, name)
        setattr(target, name, type(old)(value) if not isinstance(old, bool) else value.lower() == "true")
    return cfg


def confusion(gt: dict, pred: dict) -> dict:
    """For every ground-truth event: the predicted label with the best temporal IoU (any class)."""
    table: dict = {}
    for vid, g in gt.items():
        pevents = pred["videos"].get(vid, {}).get("events", [])
        for s, e, lab in g["events"]:
            best, best_iou = "(none)", 0.0
            for ps, pe, plab in pevents:
                iou = evaluate.tiou((s, e), (ps, pe))
                if iou > best_iou:
                    best, best_iou = plab, iou
            row = table.setdefault(lab, {})
            row[best] = row.get(best, 0) + 1
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--set", action="append", default=[], help="section.name=value config override")
    ap.add_argument("--risk", action="store_true", help="also stream Part B (slow)")
    ap.add_argument("--out", help="write the predictions here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    os.environ.setdefault("ROADWATCH_CACHE", str(ROOT / ".cache" / "perception"))

    cfg = apply_overrides(CFG, args.set)
    gt = json.loads(Path(args.labels).read_text())
    pred = {"team": "dev", "videos": {}}
    for name in gt:
        path = Path(args.videos) / name
        entry = {"events": analyze(str(path), cfg).events}
        entry["risk"] = risk_curve(str(path), cfg=cfg) if args.risk else []
        pred["videos"][name] = entry
    errors, _ = evaluate.validate(pred, gt)
    if errors:
        sys.exit("\n".join(errors))
    if args.out:
        Path(args.out).write_text(json.dumps(pred))
    evaluate.print_report(evaluate.evaluate(gt, pred, per_video=True))
    print("\nbest-overlapping prediction per ground-truth event (rows = truth):")
    for lab, row in sorted(confusion(gt, pred).items()):
        print(f"  {lab:<20} " + ", ".join(f"{k}:{v}" for k, v in sorted(row.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
