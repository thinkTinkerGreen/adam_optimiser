"""Learning rate schedules: Linear Warmup, Cosine Decay, and Warmup-Stable-Decay (WSD)."""

from __future__ import annotations

import math
from collections.abc import Callable


def get_schedule_fn(
    schedule_name: str,
    total_steps: int,
    warmup_steps: int = 30,
    decay_steps: int | None = None,
    f_min: float = 0.1,
) -> Callable[[int], float]:
    """Return a function f(step) -> lr_multiplier in [0, 1].

    Args:
        schedule_name: 'cosine', 'wsd', or 'constant'.
        total_steps: T, the horizon (e.g., 300 or 200).
        warmup_steps: W, the warmup duration (e.g., 30 or 0).
        decay_steps: D for WSD. If None, round(0.2 * total_steps).
        f_min: minimum LR floor factor for decaying schedules (default 0.1).

    Returns:
        Callable mapping step (1-based integer) to multiplier.
    """
    schedule_name = schedule_name.lower()

    if decay_steps is None:
        decay_steps = round(0.2 * total_steps)

    if schedule_name == "cosine":

        def schedule(s: int) -> float:
            if warmup_steps > 0 and s <= warmup_steps:
                return float(s) / float(warmup_steps)
            if s >= total_steps:
                return f_min
            p = (s - warmup_steps) / float(total_steps - warmup_steps)
            return f_min + (1.0 - f_min) * 0.5 * (1.0 + math.cos(math.pi * p))

        return schedule

    elif schedule_name == "wsd":
        stable_end = total_steps - decay_steps

        def schedule(s: int) -> float:
            if warmup_steps > 0 and s <= warmup_steps:
                return float(s) / float(warmup_steps)
            elif s <= stable_end:
                return 1.0
            elif s >= total_steps:
                return f_min
            else:
                q = (s - stable_end) / float(decay_steps)
                return f_min + (1.0 - f_min) * (1.0 - math.sqrt(q))

        return schedule

    elif schedule_name == "constant":

        def schedule(s: int) -> float:
            if warmup_steps > 0 and s <= warmup_steps:
                return float(s) / float(warmup_steps)
            return 1.0

        return schedule

    else:
        raise ValueError(
            f"Unknown schedule: {schedule_name}. Must be 'cosine', 'wsd', or 'constant'."
        )
