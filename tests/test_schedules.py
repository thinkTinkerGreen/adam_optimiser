"""Tests for learning rate schedules: Cosine, WSD, and Constant."""

from __future__ import annotations

import math

from adamlab.schedules import get_schedule_fn


def test_schedule_endpoints() -> None:
    """Verify endpoints and boundary behaviors for Cosine and WSD."""
    T = 300
    W = 30
    D = 60
    f_min = 0.1

    cos_fn = get_schedule_fn("cosine", total_steps=T, warmup_steps=W, f_min=f_min)
    wsd_fn = get_schedule_fn("wsd", total_steps=T, warmup_steps=W, decay_steps=D, f_min=f_min)

    # Endpoints at step T
    assert math.isclose(cos_fn(T), f_min, rel_tol=1e-5)
    assert math.isclose(wsd_fn(T), f_min, rel_tol=1e-5)

    # Warmup checks
    assert math.isclose(cos_fn(15), 0.5, rel_tol=1e-5)
    assert math.isclose(wsd_fn(15), 0.5, rel_tol=1e-5)
    assert math.isclose(cos_fn(30), 1.0, rel_tol=1e-5)
    assert math.isclose(wsd_fn(30), 1.0, rel_tol=1e-5)

    # WSD stable plateau: W < s <= T - D
    stable_end = T - D  # 240
    for s in [31, 50, 100, 200, stable_end]:
        assert math.isclose(wsd_fn(s), 1.0, rel_tol=1e-5), f"WSD not 1.0 at step {s}"

    # Cosine at step 200 (of 300 total with W=30): expect ~0.37
    # p = (200 - 30) / (300 - 30) = 170 / 270 = 0.6296
    # 0.1 + 0.9 * 0.5 * (1 + cos(pi * 0.6296)) = 0.1 + 0.45 * (1 - 0.3926) = 0.3733
    cos_200 = cos_fn(200)
    assert 0.36 < cos_200 < 0.38, f"Cosine at 200 should be ~0.37, got {cos_200}"


def test_wsd_monotonic_decay() -> None:
    """Verify that WSD decay phase (s > T - D) is strictly monotonic decreasing."""
    T = 300
    W = 30
    D = 60
    wsd_fn = get_schedule_fn("wsd", total_steps=T, warmup_steps=W, decay_steps=D)

    prev_val = 1.0
    for s in range(T - D + 1, T + 1):
        curr_val = wsd_fn(s)
        assert curr_val < prev_val, f"WSD did not decrease at step {s}: {curr_val} >= {prev_val}"
        prev_val = curr_val
