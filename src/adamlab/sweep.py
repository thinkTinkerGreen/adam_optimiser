"""Hyperparameter grid construction and deterministic edge-extension logic."""

from __future__ import annotations

import numpy as np

from adamlab.io import RunSpec


def get_base_lr_grid() -> list[float]:
    """Return base learning rate grid: 2^k for k in [-14, -13.5, ..., -6] (17 points)."""
    powers = np.arange(-14.0, -5.5, 0.5)
    return [float(2.0**k) for k in powers]


def check_and_extend_grid(
    current_grid: list[float],
    mean_losses: list[float],
) -> tuple[bool, list[float], str]:
    """Check if the grid argmin is at the boundary; if so, extend by 2 octaves (4 points).

    Returns:
        (extended: bool, new_grid: list[float], direction: 'none' | 'lower' | 'upper')
    """
    assert len(current_grid) == len(mean_losses)
    argmin_idx = int(np.argmin(mean_losses))

    # Log2 powers of the current grid
    powers = [float(np.log2(lr)) for lr in current_grid]

    if argmin_idx == 0:
        # Extend lower: 4 points downwards at step 0.5
        min_p = powers[0]
        new_powers = [min_p - 0.5 * i for i in range(4, 0, -1)]
        extended_powers = new_powers + powers
        return True, [float(2.0**p) for p in extended_powers], "lower"

    elif argmin_idx == len(current_grid) - 1:
        # Extend upper: 4 points upwards at step 0.5
        max_p = powers[-1]
        new_powers = [max_p + 0.5 * i for i in range(1, 5)]
        extended_powers = powers + new_powers
        return True, [float(2.0**p) for p in extended_powers], "upper"

    return False, current_grid, "none"


def build_sweep_specs(
    experiment: str,
    widths: list[int],
    lrs: list[float],
    seeds: list[int],
    schedule: str = "cosine",
    total_steps: int = 300,
    warmup_steps: int = 30,
    device: str = "cpu",
    use_synthetic: bool = False,
) -> list[RunSpec]:
    """Generate deterministic ordered list of RunSpecs for grid evaluation."""
    specs: list[RunSpec] = []
    for w in sorted(widths):
        for lr in sorted(lrs):
            for s in sorted(seeds):
                spec = RunSpec(
                    experiment=experiment,
                    width=w,
                    lr=lr,
                    schedule=schedule,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    seed=s,
                    device=device,
                    use_synthetic=use_synthetic,
                )
                specs.append(spec)
    return specs
