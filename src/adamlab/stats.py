"""Statistical modeling: quadratic vertex fitting, bootstrap confidence intervals, and scaling laws."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np


def fit_quadratic_vertex(
    x_points: list[float] | np.ndarray,
    y_points: list[float] | np.ndarray,
) -> tuple[float, bool, tuple[float, float, float]]:
    """Fit quadratic y = a*x^2 + b*x + c to 5 points centered on argmin.

    Returns:
        tuple (x_star, is_fallback, (a, b, c))
        If a <= 0 or x_star is outside [min(x_points), max(x_points)],
        falls back to argmin and sets is_fallback=True.
    """
    x = np.asarray(x_points, dtype=np.float64)
    y = np.asarray(y_points, dtype=np.float64)

    argmin_idx = int(np.argmin(y))
    x_argmin = float(x[argmin_idx])

    if len(x) < 3:
        return x_argmin, True, (0.0, 0.0, y[argmin_idx])

    # Fit quadratic polynomial: y = a*x^2 + b*x + c
    poly = np.polyfit(x, y, deg=2)
    a, b, c = float(poly[0]), float(poly[1]), float(poly[2])

    if a <= 1e-12:  # Not strictly convex
        return x_argmin, True, (a, b, c)

    x_star = -b / (2.0 * a)

    # Check if vertex lies within the window
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if not (x_min <= x_star <= x_max):
        return x_argmin, True, (a, b, c)

    return float(x_star), False, (a, b, c)


def compute_basin_half_width(
    x_grid: list[float] | np.ndarray,
    y_grid: list[float] | np.ndarray,
    threshold_nats: float = 0.02,
) -> float:
    """Compute basin half-width (in log2 units) where y is within threshold_nats of min(y)."""
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)

    min_y = np.min(y)
    cutoff = min_y + threshold_nats

    # Indices where loss <= cutoff
    within = np.where(y <= cutoff)[0]
    if len(within) <= 1:
        return 0.5  # default half-grid spacing

    x_within = x[within]
    span = float(np.max(x_within) - np.min(x_within))
    return span / 2.0


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_resamples: int = 1000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute empirical bootstrap confidence interval for a scalar statistic.

    Returns:
        (estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    estimate = stat_fn(data)
    ci_lower = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))
    return float(estimate), ci_lower, ci_upper


def fit_scaling_law(
    widths: list[int],
    opt_log2_lrs: list[float],
) -> dict[str, Any]:
    """Fit log2(lr*) = a + b * log2(w) via OLS and extrapolate to width 4096."""
    w = np.array(widths, dtype=np.float64)
    log2_w = np.log2(w)
    y = np.array(opt_log2_lrs, dtype=np.float64)

    # OLS fit: y = b * log2_w + a
    poly = np.polyfit(log2_w, y, deg=1)
    b, a = float(poly[0]), float(poly[1])

    # Extrapolate to 4096 (log2(4096) = 12.0)
    pred_log2_4096 = float(a + b * 12.0)
    pred_lr_4096 = float(2.0**pred_log2_4096)

    # 1/width constrained value: lr_1024 / 4
    idx_1024 = widths.index(1024)
    lr_1024 = 2.0 ** opt_log2_lrs[idx_1024]
    constrained_lr_4096 = float(lr_1024 / 4.0)

    return {
        "slope_b": b,
        "intercept_a": a,
        "pred_log2_4096": pred_log2_4096,
        "pred_lr_4096": pred_lr_4096,
        "constrained_lr_4096": constrained_lr_4096,
    }


def compute_paired_difference(
    losses_a: list[float] | np.ndarray,
    losses_b: list[float] | np.ndarray,
) -> dict[str, float]:
    """Compute paired difference delta = mean(A - B), SE, and 95% CI."""
    a = np.asarray(losses_a, dtype=np.float64)
    b = np.asarray(losses_b, dtype=np.float64)
    diff = a - b

    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = float(std_diff / math.sqrt(n)) if n > 0 else 0.0
    ci_95 = 1.96 * se

    return {
        "mean_diff": mean_diff,
        "se": se,
        "ci_lower": mean_diff - ci_95,
        "ci_upper": mean_diff + ci_95,
        "abs_diff": abs(mean_diff),
    }
