"""Rule modules, one per family of event classes."""
from . import collision, hazards, lines, motion, pedestrian, stationary
from .base import Candidate, Context, finalize

__all__ = ["Candidate", "Context", "finalize", "run_all"]


def run_all(ctx: Context, cfg) -> list[Candidate]:
    ev, kin = cfg.events, cfg.kin
    return (collision.detect(ctx, ev) + motion.detect(ctx, ev) + stationary.detect(ctx, ev, kin)
            + pedestrian.detect(ctx, ev) + lines.detect(ctx, ev, kin) + hazards.detect(ctx, ev))
