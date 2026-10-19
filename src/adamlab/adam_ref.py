"""Arbitrary precision Decimal Adam oracle and PyTorch AdamRef optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, getcontext
from typing import Any

import torch
from torch.optim import Optimizer


def adam_by_hand_decimal(
    g_list: list[float | str | Decimal],
    w0: float | str | Decimal = "1.0",
    lr: float | str | Decimal = "0.01",
    b1: float | str | Decimal = "0.9",
    b2: float | str | Decimal = "0.999",
    eps: float | str | Decimal = "1e-8",
) -> list[dict[str, Any]]:
    """Simulate Adam optimizer steps using 40-digit arbitrary precision decimal arithmetic.

    Does not import or depend on PyTorch. Acts as the pure mathematical oracle.
    """
    getcontext().prec = 40

    w = Decimal(str(w0))
    lr_d = Decimal(str(lr))
    b1_d = Decimal(str(b1))
    b2_d = Decimal(str(b2))
    eps_d = Decimal(str(eps))

    m = Decimal("0.0")
    v = Decimal("0.0")

    history: list[dict[str, Any]] = []

    for t, g_val in enumerate(g_list, start=1):
        g = Decimal(str(g_val))
        m = b1_d * m + (Decimal("1") - b1_d) * g
        v = b2_d * v + (Decimal("1") - b2_d) * (g * g)

        bc1 = Decimal("1") - (b1_d**t)
        bc2 = Decimal("1") - (b2_d**t)

        m_hat = m / bc1
        v_hat = v / bc2
        v_hat_sqrt = v_hat.sqrt()

        denom = v_hat_sqrt + eps_d
        step = lr_d * m_hat / denom
        w = w - step

        history.append(
            {
                "t": t,
                "g": g,
                "m": m,
                "v": v,
                "bc1": bc1,
                "bc2": bc2,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "v_hat_sqrt": v_hat_sqrt,
                "step": step,
                "w": w,
            }
        )

    return history


class AdamRef(Optimizer):
    """From-scratch reference Adam optimizer with bias-correction toggle.

    When bias_correction=True, mathematically equivalent to torch.optim.Adam.
    When bias_correction=False, update is lr * m_t / (sqrt(v_t) + eps).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "bias_correction": bias_correction,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            bc = group["bias_correction"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("AdamRef does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # Decay the first and second moment running average
                exp_avg.mul_(b1).add_(grad, alpha=1.0 - b1)
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                if bc:
                    bc1 = 1.0 - (b1**t)
                    bc2 = 1.0 - (b2**t)
                    step_size = lr / bc1
                    denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-lr)

        return loss
