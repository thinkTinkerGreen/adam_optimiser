"""Tests for statistical routines: quadratic vertex fit, bootstrap, and scaling law."""

from __future__ import annotations

import math

import numpy as np

from adamlab.stats import (
    bootstrap_ci,
    compute_paired_difference,
    fit_quadratic_vertex,
    fit_scaling_law,
)


def test_quadratic_vertex_recovers_known_parabola() -> None:
    """Quadratic fit on y = 2*(x - 3)^2 + 1 recovers x* = 3.0."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * (x - 3.0) ** 2 + 1.0

    x_star, is_fallback, (a, b, c) = fit_quadratic_vertex(x, y)
    assert not is_fallback
    assert math.isclose(x_star, 3.0, abs_tol=1e-6)
    assert math.isclose(a, 2.0, abs_tol=1e-6)


def test_quadratic_vertex_fallback_non_convex() -> None:
    """Concave parabola (a < 0) correctly triggers discrete argmin fallback."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -2.0 * (x - 3.0) ** 2 + 10.0  # Minima at boundaries

    x_star, is_fallback, _ = fit_quadratic_vertex(x, y)
    assert is_fallback
    assert x_star in (1.0, 5.0)


def test_bootstrap_covers_true_mean() -> None:
    """Bootstrap 95% CI covers the true population mean."""
    rng = np.random.default_rng(42)
    true_mean = 5.0
    data = rng.normal(loc=true_mean, scale=1.0, size=50)

    est, ci_low, ci_high = bootstrap_ci(
        data, stat_fn=lambda d: float(np.mean(d)), n_resamples=500, seed=42
    )
    assert ci_low <= true_mean <= ci_high


def test_fit_scaling_law() -> None:
    """Scaling law OLS fits line and extrapolates to 4096."""
    widths = [256, 512, 1024]
    # Synthetic optimal log2 LRs: slope = -0.5
    # log2(w): 8, 9, 10
    # y = -0.5 * log2(w) - 5 => for 8: -9, for 9: -9.5, for 10: -10
    log2_lrs = [-9.0, -9.5, -10.0]

    res = fit_scaling_law(widths, log2_lrs)
    assert math.isclose(res["slope_b"], -0.5, abs_tol=1e-5)
    # Extrapolation to 4096 (log2(w) = 12): -0.5 * 12 - 5 = -11.0
    assert math.isclose(res["pred_log2_4096"], -11.0, abs_tol=1e-5)
    assert math.isclose(res["pred_lr_4096"], 2.0**-11.0, rel_tol=1e-5)


def test_compute_paired_difference() -> None:
    """Test paired difference calculation."""
    a = [2.0, 2.1, 1.9]
    b = [1.8, 1.9, 1.7]
    diff = compute_paired_difference(a, b)
    assert math.isclose(diff["mean_diff"], 0.2, abs_tol=1e-5)
    assert diff["abs_diff"] == diff["mean_diff"]
