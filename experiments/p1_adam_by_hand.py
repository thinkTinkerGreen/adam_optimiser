"""Part 1: Adam by Hand vs PyTorch FP64 and FP32 (One Weight, Five Gradients)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from adamlab.adam_ref import adam_by_hand_decimal
from adamlab.io import atomic_save_json


def run_part1(results_dir: str | Path = "results") -> dict[str, Any]:
    """Execute Part 1 evaluation and write results/p1_adam_by_hand.json."""
    w0_val = 1.0
    lr_val = 0.01
    b1_val = 0.9
    b2_val = 0.999
    eps_val = 1e-8
    grads = [0.1, -0.2, 0.3, -0.4, 0.5]

    # 1. Oracle run (Decimal 40-digit precision)
    oracle_history = adam_by_hand_decimal(
        grads, w0=w0_val, lr=lr_val, b1=b1_val, b2=b2_val, eps=eps_val
    )

    # 2. PyTorch FP64 run
    w_fp64 = torch.tensor([w0_val], dtype=torch.float64, requires_grad=True)
    opt_fp64 = torch.optim.Adam(
        [w_fp64], lr=lr_val, betas=(b1_val, b2_val), eps=eps_val, foreach=False
    )

    torch_fp64_history = []
    for t, g in enumerate(grads, start=1):
        w_before = w_fp64.item()
        w_fp64.grad = torch.tensor([g], dtype=torch.float64)
        opt_fp64.step()

        w_after = w_fp64.item()
        state = opt_fp64.state[w_fp64]
        m = state["exp_avg"].item()
        v = state["exp_avg_sq"].item()
        step_idx = state["step"]
        if isinstance(step_idx, torch.Tensor):
            step_idx = int(step_idx.item())

        bc1 = 1.0 - (b1_val**t)
        bc2 = 1.0 - (b2_val**t)
        m_hat = m / bc1
        v_hat = v / bc2
        step_val = w_before - w_after

        torch_fp64_history.append(
            {
                "t": t,
                "g": g,
                "m": m,
                "v": v,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "step": step_val,
                "w": w_after,
            }
        )

    # 3. PyTorch FP32 run
    w_fp32 = torch.tensor([w0_val], dtype=torch.float32, requires_grad=True)
    opt_fp32 = torch.optim.Adam(
        [w_fp32], lr=lr_val, betas=(b1_val, b2_val), eps=eps_val, foreach=False
    )

    torch_fp32_history = []
    for t, g in enumerate(grads, start=1):
        w_before = w_fp32.item()
        w_fp32.grad = torch.tensor([g], dtype=torch.float32)
        opt_fp32.step()

        w_after = w_fp32.item()
        state = opt_fp32.state[w_fp32]
        m = state["exp_avg"].item()
        v = state["exp_avg_sq"].item()
        step_idx = state["step"]
        if isinstance(step_idx, torch.Tensor):
            step_idx = int(step_idx.item())

        bc1 = 1.0 - (b1_val**t)
        bc2 = 1.0 - (b2_val**t)
        m_hat = m / bc1
        v_hat = v / bc2
        step_val = w_before - w_after

        torch_fp32_history.append(
            {
                "t": t,
                "g": g,
                "m": m,
                "v": v,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "step": step_val,
                "w": w_after,
            }
        )

    # 4. Comparison table and max absolute errors
    quantities = ["m", "v", "m_hat", "v_hat", "step", "w"]
    max_err_fp64: dict[str, float] = dict.fromkeys(quantities, 0.0)
    max_err_fp32: dict[str, float] = dict.fromkeys(quantities, 0.0)

    comparison_table = []
    for i in range(len(grads)):
        o = oracle_history[i]
        p64 = torch_fp64_history[i]
        p32 = torch_fp32_history[i]

        step_entry: dict[str, Any] = {"t": o["t"], "g": float(o["g"])}

        for q in quantities:
            o_val = float(o[q])
            val64 = float(p64[q])
            val32 = float(p32[q])
            err64 = abs(val64 - o_val)
            err32 = abs(val32 - o_val)

            if err64 > max_err_fp64[q]:
                max_err_fp64[q] = err64
            if err32 > max_err_fp32[q]:
                max_err_fp32[q] = err32

            step_entry[q] = {
                "oracle": f"{o_val:.12f}",
                "torch_fp64": f"{val64:.12f}",
                "torch_fp32": f"{val32:.12f}",
                "err_fp64": f"{err64:.2e}",
                "err_fp32": f"{err32:.2e}",
            }
        comparison_table.append(step_entry)

    overall_max_err_fp64 = max(max_err_fp64.values())
    overall_max_err_fp32 = max(max_err_fp32.values())

    # Format human-readable output
    out_payload = {
        "overall_max_err_fp64": overall_max_err_fp64,
        "overall_max_err_fp32": overall_max_err_fp32,
        "max_err_fp64": max_err_fp64,
        "max_err_fp32": max_err_fp32,
        "comparison_table": comparison_table,
        "oracle_raw": [{k: str(v) for k, v in step.items()} for step in oracle_history],
    }

    out_path = Path(results_dir) / "p1_adam_by_hand.json"
    atomic_save_json(out_path, out_payload)
    print(
        f"[Part 1] Done. Overall max error FP64: {overall_max_err_fp64:.2e}, FP32: {overall_max_err_fp32:.2e}"
    )
    return out_payload


if __name__ == "__main__":
    run_part1()
