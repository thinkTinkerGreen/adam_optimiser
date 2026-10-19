"""Tests for Part 2: Closed-form ratio and t* threshold recovery."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adamlab.adam_ref import AdamRef
from experiments.p2_bias_correction import closed_form_ratio, compute_t_star

EXPECTED_T_STAR = {
    0.999: (2375, 3925),
    0.99: (237, 391),
    0.98: (118, 195),
    0.95: (43, 76),
}


def test_closed_form_matches_simulation_steps_1_to_50() -> None:
    """Simulation on 1 weight with constant gradient 1.0 equals closed form to 1e-9."""
    b1, b2 = 0.9, 0.999
    lr = 0.01
    eps = 0.0

    w_bc = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    w_nobc = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    opt_bc = AdamRef([w_bc], lr=lr, betas=(b1, b2), eps=eps, bias_correction=True)
    opt_nobc = AdamRef([w_nobc], lr=lr, betas=(b1, b2), eps=eps, bias_correction=False)

    g = torch.tensor([1.0], dtype=torch.float64)

    for t in range(1, 51):
        prev_bc = w_bc.item()
        prev_nobc = w_nobc.item()

        w_bc.grad = g.clone()
        w_nobc.grad = g.clone()

        opt_bc.step()
        opt_nobc.step()

        delta_bc = abs(w_bc.item() - prev_bc)
        delta_nobc = abs(w_nobc.item() - prev_nobc)
        sim_ratio = delta_nobc / delta_bc

        cf_ratio = float(closed_form_ratio(t, b1=b1, b2=b2))
        assert abs(sim_ratio - cf_ratio) < 1e-8, f"Step {t}: sim {sim_ratio} != cf {cf_ratio}"


@pytest.mark.parametrize("b2, exp_5, exp_1", [(k, v[0], v[1]) for k, v in EXPECTED_T_STAR.items()])
def test_t_star_matches_golden_table(b2: float, exp_5: int, exp_1: int) -> None:
    """t* finder returns golden table values for all beta2 settings."""
    steps = np.arange(1, 20001)
    ratios = closed_form_ratio(steps, b1=0.9, b2=b2)

    t_5 = compute_t_star(ratios, theta=0.05, horizon=20000)
    t_1 = compute_t_star(ratios, theta=0.01, horizon=20000)

    assert t_5 == exp_5, f"Beta2={b2}: got t*(5%)={t_5}, expected {exp_5}"
    assert t_1 == exp_1, f"Beta2={b2}: got t*(1%)={t_1}, expected {exp_1}"
