"""Part 2: Bias Correction Off vs On: First 20 Steps and Horizon Simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from adamlab.adam_ref import AdamRef
from adamlab.io import atomic_save_json


def closed_form_ratio(
    t: int | np.ndarray, b1: float = 0.9, b2: float = 0.999
) -> float | np.ndarray:
    """Compute theoretical ratio r_t = (1 - b1^t) / sqrt(1 - b2^t)."""
    t_arr = np.asarray(t, dtype=np.float64)
    bc1 = 1.0 - (b1**t_arr)
    bc2 = 1.0 - (b2**t_arr)
    return bc1 / np.sqrt(bc2)


def compute_t_star(
    ratios: list[float] | np.ndarray,
    theta: float = 0.05,
    horizon: int = 20000,
) -> int:
    """Smallest t* such that for all t >= t* up to horizon, |r_t - 1| < theta."""
    r = np.asarray(ratios, dtype=np.float64)
    limit = min(len(r), horizon)

    # Condition: |r_t - 1| < theta
    satisfies = np.abs(r[:limit] - 1.0) < theta

    # Search backwards from horizon
    for t_idx in range(limit - 1, -1, -1):
        if not satisfies[t_idx]:
            # The next step was the start of the suffix
            if t_idx + 1 < limit:
                return t_idx + 2  # 1-based step index
            return limit + 1

    return 1


def run_part2(
    results_dir: str | Path = "results", figures_dir: str | Path = "figures"
) -> dict[str, Any]:
    """Execute Part 2 simulations, compute t* table, and generate 3-panel figure."""
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    b1 = 0.9
    b2_list = [0.999, 0.99, 0.98, 0.95]
    horizon = 20000
    n_weights = 10000
    lr = 0.01
    eps = 1e-8

    # 1. Closed-form t* table for all beta2
    t_star_table = {}
    steps_arr = np.arange(1, horizon + 1)

    for b2 in b2_list:
        cf_ratios = closed_form_ratio(steps_arr, b1=b1, b2=b2)
        t_star_5 = compute_t_star(cf_ratios, theta=0.05, horizon=horizon)
        t_star_1 = compute_t_star(cf_ratios, theta=0.01, horizon=horizon)
        t_star_table[str(b2)] = {
            "t_star_5pct": int(t_star_5),
            "t_star_1pct": int(t_star_1),
        }

    # 2. Simulation for first 20 steps (5 seeds) for both S-const and S-noisy
    seeds = [0, 1, 2, 3, 4]
    first_20_steps = 20

    # Containers for noisy simulation across seeds
    noisy_ratios_by_seed = []
    noisy_delta_bc_by_seed = []
    noisy_delta_nobc_by_seed = []
    noisy_disp_bc_by_seed = []
    noisy_disp_nobc_by_seed = []

    for seed in seeds:
        gen = torch.Generator(device="cpu").manual_seed(seed)

        # Parameters for BC and noBC
        w_bc = torch.zeros(n_weights, dtype=torch.float32, requires_grad=True)
        w_nobc = torch.zeros(n_weights, dtype=torch.float32, requires_grad=True)

        opt_bc = AdamRef([w_bc], lr=lr, betas=(b1, 0.999), eps=eps, bias_correction=True)
        opt_nobc = AdamRef([w_nobc], lr=lr, betas=(b1, 0.999), eps=eps, bias_correction=False)

        seed_delta_bc = []
        seed_delta_nobc = []
        seed_disp_bc = []
        seed_disp_nobc = []
        seed_ratio = []

        w_bc_0 = w_bc.detach().clone()
        w_nobc_0 = w_nobc.detach().clone()

        for _s in range(1, first_20_steps + 1):
            # Noisy gradient: N(0.5, 1.0)
            g = torch.randn(n_weights, generator=gen, dtype=torch.float32).mul_(1.0).add_(0.5)

            prev_bc = w_bc.detach().clone()
            prev_nobc = w_nobc.detach().clone()

            w_bc.grad = g.clone()
            w_nobc.grad = g.clone()

            opt_bc.step()
            opt_nobc.step()

            delta_bc = torch.abs(w_bc.detach() - prev_bc).mean().item()
            delta_nobc = torch.abs(w_nobc.detach() - prev_nobc).mean().item()

            disp_bc = torch.abs(w_bc.detach() - w_bc_0).mean().item()
            disp_nobc = torch.abs(w_nobc.detach() - w_nobc_0).mean().item()

            seed_delta_bc.append(delta_bc)
            seed_delta_nobc.append(delta_nobc)
            seed_disp_bc.append(disp_bc)
            seed_disp_nobc.append(disp_nobc)
            seed_ratio.append(delta_nobc / delta_bc if delta_bc > 0 else 1.0)

        noisy_delta_bc_by_seed.append(seed_delta_bc)
        noisy_delta_nobc_by_seed.append(seed_delta_nobc)
        noisy_disp_bc_by_seed.append(seed_disp_bc)
        noisy_disp_nobc_by_seed.append(seed_disp_nobc)
        noisy_ratios_by_seed.append(seed_ratio)

    noisy_ratios_arr = np.array(noisy_ratios_by_seed)  # (5, 20)
    noisy_ratios_mean = np.mean(noisy_ratios_arr, axis=0)
    noisy_ratios_std = np.std(noisy_ratios_arr, axis=0)

    noisy_delta_bc_mean = np.mean(np.array(noisy_delta_bc_by_seed), axis=0)
    noisy_delta_nobc_mean = np.mean(np.array(noisy_delta_nobc_by_seed), axis=0)

    noisy_disp_bc_mean = np.mean(np.array(noisy_disp_bc_by_seed), axis=0)
    noisy_disp_nobc_mean = np.mean(np.array(noisy_disp_nobc_by_seed), axis=0)

    # Closed-form values for steps 1..20 (b2=0.999)
    t_20 = np.arange(1, 21)
    cf_ratios_20 = closed_form_ratio(t_20, b1=b1, b2=0.999)

    # 3. Generate 3-panel figure
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    # Panel (a): mean |Δw| per step
    axes[0].plot(t_20, noisy_delta_bc_mean, "o-", label="With BC", color="#1f77b4", linewidth=2)
    axes[0].plot(
        t_20, noisy_delta_nobc_mean, "s--", label="Without BC", color="#d62728", linewidth=2
    )
    axes[0].set_xlabel("Step t")
    axes[0].set_ylabel(r"Mean $|\Delta w|$ per step")
    axes[0].set_title(r"(a) Mean Step Magnitude $|\Delta w|$")
    axes[0].set_xticks(range(1, 21, 2))
    axes[0].legend()

    # Panel (b): ratio |Δw_noBC| / |Δw_BC|
    axes[1].plot(t_20, cf_ratios_20, "k-", label="Closed Form (S-const)", linewidth=2.5)
    axes[1].plot(
        t_20, noisy_ratios_mean, "o--", label="Simulation (S-noisy)", color="#2ca02c", linewidth=2
    )
    axes[1].fill_between(
        t_20,
        noisy_ratios_mean - noisy_ratios_std,
        noisy_ratios_mean + noisy_ratios_std,
        color="#2ca02c",
        alpha=0.2,
    )
    axes[1].axhline(1.0, color="gray", linestyle=":")
    axes[1].set_xlabel("Step t")
    axes[1].set_ylabel(r"Ratio $|\Delta w_{\mathrm{noBC}}| / |\Delta w_{\mathrm{BC}}|$")
    axes[1].set_title(r"(b) Ratio: No-BC to BC (Peak at $t \approx 12$)")
    axes[1].set_xticks(range(1, 21, 2))
    axes[1].legend()

    # Panel (c): cumulative displacement |w_t - w_0|
    axes[2].plot(t_20, noisy_disp_bc_mean, "o-", label="With BC", color="#1f77b4", linewidth=2)
    axes[2].plot(
        t_20, noisy_disp_nobc_mean, "s--", label="Without BC", color="#d62728", linewidth=2
    )
    axes[2].set_xlabel("Step t")
    axes[2].set_ylabel(r"Cumulative $|w_t - w_0|$")
    axes[2].set_title(r"(c) Cumulative Displacement")
    axes[2].set_xticks(range(1, 21, 2))
    axes[2].legend()

    plt.tight_layout()
    png_path = figures_path / "p2_first20.png"
    svg_path = figures_path / "p2_first20.svg"
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()

    # 4. Long-horizon simulation cross-check (S-const and S-noisy) for b2=0.999
    # Simulate S-const 1 weight to H=20,000
    w_const_bc = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    w_const_nobc = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt_c_bc = AdamRef([w_const_bc], lr=lr, betas=(b1, 0.999), eps=eps, bias_correction=True)
    opt_c_nobc = AdamRef([w_const_nobc], lr=lr, betas=(b1, 0.999), eps=eps, bias_correction=False)

    sim_const_ratios = []
    g_one = torch.tensor([1.0], dtype=torch.float64)
    for _ in range(horizon):
        w_const_bc.grad = g_one.clone()
        w_const_nobc.grad = g_one.clone()
        p_bc = w_const_bc.item()
        p_nobc = w_const_nobc.item()
        opt_c_bc.step()
        opt_c_nobc.step()
        d_bc = abs(w_const_bc.item() - p_bc)
        d_nobc = abs(w_const_nobc.item() - p_nobc)
        sim_const_ratios.append(d_nobc / d_bc if d_bc > 0 else 1.0)

    sim_t_star_5 = compute_t_star(sim_const_ratios, theta=0.05, horizon=horizon)
    sim_t_star_1 = compute_t_star(sim_const_ratios, theta=0.01, horizon=horizon)

    out_payload = {
        "t_star_table": t_star_table,
        "sim_cross_check_0_999": {
            "closed_form_5pct": t_star_table["0.999"]["t_star_5pct"],
            "simulation_5pct": int(sim_t_star_5),
            "closed_form_1pct": t_star_table["0.999"]["t_star_1pct"],
            "simulation_1pct": int(sim_t_star_1),
        },
        "peak_ratio": {
            "t": 12,
            "closed_form": float(cf_ratios_20[11]),
            "simulation_noisy_mean": float(noisy_ratios_mean[11]),
        },
        "r_at_1": float(cf_ratios_20[0]),
        "r_at_20": float(cf_ratios_20[19]),
        "headline_conclusion": "Within the first 20 steps, the difference between bias correction on vs off never stops mattering for beta2=0.999. The ratio peaks at step 12 at 6.57x and remains at 6.24x at step 20.",
    }

    atomic_save_json(results_path / "p2_bias_correction.json", out_payload)
    print(
        f"[Part 2] Done. t*(5%, b2=0.999) = {t_star_table['0.999']['t_star_5pct']}, t*(1%, b2=0.999) = {t_star_table['0.999']['t_star_1pct']}"
    )
    return out_payload


if __name__ == "__main__":
    run_part2()
