"""Tests for Part 1: Golden table values and Decimal oracle vs PyTorch agreement."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from adamlab.adam_ref import adam_by_hand_decimal
from experiments.p1_adam_by_hand import run_part1

# Golden table values from Section 3.2
GOLDEN_VALUES = [
    {
        "t": 1,
        "m": 0.010000000000,
        "v": 0.000010000000,
        "m_hat": 0.100000000000,
        "v_hat": 0.010000000000,
        "step": 0.009999999000,
        "w": 0.990000001000,
    },
    {
        "t": 2,
        "m": -0.011000000000,
        "v": 0.000049990000,
        "m_hat": -0.057894736842,
        "v_hat": 0.025007503752,
        "step": -0.003661035039,
        "w": 0.993661036039,
    },
    {
        "t": 3,
        "m": 0.020100000000,
        "v": 0.000139940010,
        "m_hat": 0.074169741697,
        "v_hat": 0.046693347783,
        "step": 0.003432410644,
        "w": 0.990228625395,
    },
    {
        "t": 4,
        "m": -0.021910000000,
        "v": 0.000299800070,
        "m_hat": -0.063710380925,
        "v_hat": 0.075062536258,
        "step": -0.002325404822,
        "w": 0.992554030216,
    },
    {
        "t": 5,
        "m": 0.030281000000,
        "v": 0.000549500270,
        "m_hat": 0.073944470221,
        "v_hat": 0.110120074002,
        "step": 0.002228293746,
        "w": 0.990325736470,
    },
]


def test_decimal_oracle_matches_golden_table() -> None:
    """Verify that Decimal oracle matches the pre-computed golden values to 12 decimal places."""
    grads = [0.1, -0.2, 0.3, -0.4, 0.5]
    history = adam_by_hand_decimal(grads, w0=1.0, lr=0.01, b1=0.9, b2=0.999, eps=1e-8)

    for i, golden in enumerate(GOLDEN_VALUES):
        step_res = history[i]
        assert step_res["t"] == golden["t"]
        for key in ["m", "v", "m_hat", "v_hat", "step", "w"]:
            val = float(step_res[key])
            expected = golden[key]
            assert abs(val - expected) < 1e-11, f"Step {i + 1} {key}: {val} != {expected}"


def test_pytorch_agrees_with_oracle(tmp_path: pytest.TempPathFactory) -> None:
    """Pass criteria from Section 3.4: FP64 error <= 1e-12, FP32 error <= 2e-6."""
    res = run_part1(results_dir=str(tmp_path))
    assert res["overall_max_err_fp64"] <= 1e-12
    assert res["overall_max_err_fp32"] <= 2e-6


@settings(max_examples=25, deadline=None)
@given(
    w0=st.floats(min_value=-10.0, max_value=10.0),
    lr=st.floats(min_value=1e-5, max_value=0.1),
    b1=st.floats(min_value=0.5, max_value=0.95),
    b2=st.floats(min_value=0.9, max_value=0.9999),
    eps=st.floats(min_value=1e-9, max_value=1e-6),
    grads=st.lists(st.floats(min_value=-5.0, max_value=5.0), min_size=5, max_size=20),
)
def test_hypothesis_oracle_vs_pytorch_fp64(
    w0: float, lr: float, b1: float, b2: float, eps: float, grads: list[float]
) -> None:
    """Hypothesis property test: oracle and PyTorch FP64 agree across arbitrary inputs."""
    oracle_hist = adam_by_hand_decimal(grads, w0=w0, lr=lr, b1=b1, b2=b2, eps=eps)

    w = torch.tensor([w0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, betas=(b1, b2), eps=eps, foreach=False)

    for t, g in enumerate(grads, start=1):
        w.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()

        o_w = float(oracle_hist[t - 1]["w"])
        p_w = w.item()

        abs_err = abs(p_w - o_w)
        rel_err = abs_err / (abs(o_w) + 1e-9)
        assert rel_err < 1e-9 or abs_err < 1e-10
