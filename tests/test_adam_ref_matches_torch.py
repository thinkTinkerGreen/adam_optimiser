"""Gate test: AdamRef(bias_correction=True) matches torch.optim.Adam over 1000 steps."""

from __future__ import annotations

import torch

from adamlab.adam_ref import AdamRef


def test_adam_ref_matches_torch_1000_steps() -> None:
    """Verify that AdamRef(bias_correction=True) is bitwise/fp64 identical to torch.optim.Adam."""
    torch.manual_seed(42)
    dim = 20
    n_steps = 1000

    w_ref = torch.randn(dim, dtype=torch.float64, requires_grad=True)
    w_torch = w_ref.detach().clone().requires_grad_(True)

    lr = 0.003
    betas = (0.9, 0.999)
    eps = 1e-8

    opt_ref = AdamRef([w_ref], lr=lr, betas=betas, eps=eps, bias_correction=True)
    opt_torch = torch.optim.Adam([w_torch], lr=lr, betas=betas, eps=eps, foreach=False)

    for _ in range(n_steps):
        g = torch.randn(dim, dtype=torch.float64)

        w_ref.grad = g.clone()
        w_torch.grad = g.clone()

        opt_ref.step()
        opt_torch.step()

        diff = torch.max(torch.abs(w_ref - w_torch)).item()
        assert diff <= 1e-12, f"Discrepancy {diff} exceeded 1e-12 at step"
