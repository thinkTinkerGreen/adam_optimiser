"""Part 5: Learning Rate Sweeps across Widths (256, 512, 1024) and Width-4096 Extrapolation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from adamlab.io import RunResult, RunSpec, atomic_save_json
from adamlab.stats import (
    bootstrap_ci,
    compute_basin_half_width,
    fit_quadratic_vertex,
    fit_scaling_law,
)
from adamlab.sweep import check_and_extend_grid, get_base_lr_grid
from adamlab.train import run_training


def run_part5(
    device: str = "cpu",
    use_synthetic: bool = False,
    results_dir: str | Path | None = None,
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Execute Part 5 LR sweeps, quadratic vertex fitting, bootstrap CIs, and 4096 extrapolation."""
    if results_dir is None:
        results_dir = os.environ.get("ADAMLAB_RESULTS_DIR", "results")
    res_dir = Path(results_dir)
    fig_dir = Path(figures_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    widths = [256, 512, 1024]
    tune_seeds = [0, 1, 2]
    report_seeds = [100, 101, 102]
    total_steps = 300
    warmup_steps = 30

    grid = get_base_lr_grid()
    grid_extensions: list[dict[str, Any]] = []

    # Store results per width: width -> lr -> seed -> RunResult
    results_by_width: dict[int, dict[float, dict[int, RunResult]]] = {w: {} for w in widths}

    for w in widths:
        current_grid = list(grid)
        while True:
            # Run all grid points for current width that aren't run yet
            for lr in current_grid:
                if lr not in results_by_width[w]:
                    results_by_width[w][lr] = {}
                for s in tune_seeds:
                    if s not in results_by_width[w][lr]:
                        spec = RunSpec(
                            experiment="p5_sweep",
                            width=w,
                            lr=lr,
                            schedule="cosine",
                            total_steps=total_steps,
                            warmup_steps=warmup_steps,
                            seed=s,
                            device=device,
                            use_synthetic=use_synthetic,
                            eval_steps=[200, 300],
                        )
                        res = run_training(spec, save_to_disk=True, results_dir=res_dir)
                        results_by_width[w][lr][s] = res

            # Calculate 3-seed mean tune_val loss at step 300
            mean_losses = [
                float(
                    np.mean(
                        [
                            results_by_width[w][lr][s].eval_losses.get("step_300_tune_val", 5.0)
                            for s in tune_seeds
                        ]
                    )
                )
                for lr in current_grid
            ]

            extended, new_grid, direction = check_and_extend_grid(current_grid, mean_losses)
            if extended:
                grid_extensions.append(
                    {
                        "width": w,
                        "direction": direction,
                        "previous_grid_size": len(current_grid),
                        "new_grid_size": len(new_grid),
                    }
                )
                current_grid = new_grid
            else:
                break

    # Save extensions log
    atomic_save_json(res_dir / "sweep_extensions.json", grid_extensions)

    # Quadratic vertex fitting and basin analysis per width
    minima_summary = {}
    opt_log2_lrs = []
    basin_widths = {}

    for w in widths:
        w_grid = sorted(results_by_width[w].keys())
        loss_matrix = np.array(
            [
                [
                    results_by_width[w][lr][s].eval_losses.get("step_300_tune_val", 5.0)
                    for s in tune_seeds
                ]
                for lr in w_grid
            ]
        )  # (n_lrs, 3)

        mean_losses = np.mean(loss_matrix, axis=1)
        grid_argmin_idx = int(np.argmin(mean_losses))
        grid_argmin_lr = w_grid[grid_argmin_idx]

        # 5 points centered on argmin
        start_idx = max(0, min(grid_argmin_idx - 2, len(w_grid) - 5))
        window_lrs = w_grid[start_idx : start_idx + 5]
        window_losses = mean_losses[start_idx : start_idx + 5]

        log2_window = np.log2(window_lrs)
        x_star, is_fallback, (a, b, c) = fit_quadratic_vertex(log2_window, window_losses)
        lr_star = float(2.0**x_star)
        opt_log2_lrs.append(x_star)

        # Basin half-width
        h_w = compute_basin_half_width(np.log2(w_grid), mean_losses, threshold_nats=0.02)
        basin_widths[w] = h_w

        # Bootstrap 95% CI on x*
        def fit_vertex_on_seed_sample(
            sample_matrix: np.ndarray,
            s_idx: int = start_idx,
            l_win: np.ndarray = log2_window,
        ) -> float:
            m_loss = np.mean(sample_matrix, axis=1)
            w_losses = m_loss[s_idx : s_idx + 5]
            vx, _, _ = fit_quadratic_vertex(l_win, w_losses)
            return vx

        _, ci_low, ci_high = bootstrap_ci(
            loss_matrix,
            stat_fn=lambda sm: fit_vertex_on_seed_sample(sm),
            n_resamples=1000,
            seed=12345,
        )

        # Held-out confirmation on report seeds at snapped grid LR
        snapped_lr = w_grid[int(np.argmin(np.abs(np.array(w_grid) - lr_star)))]
        held_out_losses = []
        for rs in report_seeds:
            spec_rep = RunSpec(
                experiment="p5_report",
                width=w,
                lr=snapped_lr,
                schedule="cosine",
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                seed=rs,
                device=device,
                use_synthetic=use_synthetic,
                eval_steps=[300],
            )
            res_rep = run_training(spec_rep, save_to_disk=True, results_dir=res_dir)
            held_out_losses.append(res_rep.eval_losses.get("step_300_test", 5.0))

        minima_summary[str(w)] = {
            "width": w,
            "grid_argmin_lr": grid_argmin_lr,
            "fitted_x_star": x_star,
            "lr_star": lr_star,
            "ci_95": [float(2.0**ci_low), float(2.0**ci_high)],
            "ci_95_log2": [ci_low, ci_high],
            "basin_half_width_log2": h_w,
            "is_fallback": is_fallback,
            "snapped_lr": snapped_lr,
            "held_out_test_mean": float(np.mean(held_out_losses)),
            "held_out_test_se": float(
                np.std(held_out_losses, ddof=1) / math.sqrt(len(held_out_losses))
            ),
        }

    # 4. Width-4096 Extrapolation (§8.5)
    scaling_res = fit_scaling_law(widths, opt_log2_lrs)
    slope_b = scaling_res["slope_b"]
    pred_lr_4096 = scaling_res["pred_lr_4096"]
    constrained_lr_4096 = scaling_res["constrained_lr_4096"]

    # Bootstrap 95% CI on slope b and pred_log2_4096
    rng = np.random.default_rng(12345)
    n_boot = 1000
    boot_slopes = []
    boot_preds = []

    for _ in range(n_boot):
        sample_x_stars = []
        for _idx, w in enumerate(widths):
            w_grid = sorted(results_by_width[w].keys())
            loss_matrix = np.array(
                [
                    [
                        results_by_width[w][lr][s].eval_losses.get("step_300_tune_val", 5.0)
                        for s in tune_seeds
                    ]
                    for lr in w_grid
                ]
            )
            # Resample the 3 seeds
            sampled_seeds = rng.choice(3, size=3, replace=True)
            sampled_matrix = loss_matrix[:, sampled_seeds]
            m_loss = np.mean(sampled_matrix, axis=1)
            argmin_idx = int(np.argmin(m_loss))
            start_i = max(0, min(argmin_idx - 2, len(w_grid) - 5))
            w_lrs = w_grid[start_i : start_i + 5]
            vx, _, _ = fit_quadratic_vertex(np.log2(w_lrs), m_loss[start_i : start_i + 5])
            sample_x_stars.append(vx)

        boot_fit = fit_scaling_law(widths, sample_x_stars)
        boot_slopes.append(boot_fit["slope_b"])
        boot_preds.append(boot_fit["pred_log2_4096"])

    slope_ci_low = float(np.percentile(boot_slopes, 2.5))
    slope_ci_high = float(np.percentile(boot_slopes, 97.5))
    pred_ci_low = float(np.percentile(boot_preds, 2.5))
    pred_ci_high = float(np.percentile(boot_preds, 97.5))
    w95_log2 = pred_ci_high - pred_ci_low

    # Confidence label based on §8.5 rule
    h_1024 = basin_widths[1024]
    if w95_log2 <= h_1024:
        raw_confidence = "moderate"
    elif w95_log2 <= 2.0 * h_1024:
        raw_confidence = "low-moderate"
    else:
        raw_confidence = "low"
    confidence_label = "moderate" if raw_confidence == "moderate" else raw_confidence

    contains_neg_one = bool(slope_ci_low <= -1.0 <= slope_ci_high)

    # Hypotheses evaluation
    h3_supported = bool(opt_log2_lrs[0] > opt_log2_lrs[1] > opt_log2_lrs[2])
    h4_supported = bool(-1.0 <= slope_b <= 0.0)
    h5_supported = bool(
        minima_summary["256"]["held_out_test_mean"]
        > minima_summary["512"]["held_out_test_mean"]
        > minima_summary["1024"]["held_out_test_mean"]
    )

    # 5. Generate Figures (figures/p5_lr_sweep.png)
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {256: "#1f77b4", 512: "#2ca02c", 1024: "#d62728"}
    for w in widths:
        w_grid = sorted(results_by_width[w].keys())
        loss_matrix = np.array(
            [
                [
                    results_by_width[w][lr][s].eval_losses.get("step_300_tune_val", 5.0)
                    for s in tune_seeds
                ]
                for lr in w_grid
            ]
        )
        m_loss = np.mean(loss_matrix, axis=1)

        ax1.plot(w_grid, m_loss, "o-", color=colors[w], label=f"Width {w}", linewidth=1.8)
        # Faint seed points
        for seed_idx in range(len(tune_seeds)):
            ax1.scatter(w_grid, loss_matrix[:, seed_idx], color=colors[w], alpha=0.2, s=15)

        # Mark minimum
        m_info = minima_summary[str(w)]
        ax1.axvline(m_info["lr_star"], color=colors[w], linestyle=":", alpha=0.8)
        ax1.scatter([m_info["grid_argmin_lr"]], [np.min(m_loss)], color=colors[w], s=80, marker="*")
        ax1.annotate(
            f"{w}: {m_info['lr_star']:.2e}",
            (m_info["lr_star"], np.min(m_loss)),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"Learning Rate (log$_2$ scale)")
    ax1.set_ylabel("Tune Val Loss at Step 300 (nats)")
    ax1.set_title("Part 5: Loss vs Learning Rate across Widths")
    ax1.legend()

    # Panel 2: x* vs log2(width) and scaling law extrapolation
    log2_w_arr = np.array([np.log2(w) for w in widths])
    extrap_x = np.linspace(7.5, 12.5, 100)
    extrap_y = scaling_res["intercept_a"] + slope_b * extrap_x

    ax2.scatter(log2_w_arr, opt_log2_lrs, color="black", s=60, zorder=5, label="Measured Minima")
    ax2.plot(extrap_x, extrap_y, "r--", label=f"OLS Fit (slope b={slope_b:.2f})")
    ax2.fill_between(
        extrap_x,
        extrap_x * slope_ci_low + scaling_res["intercept_a"],
        extrap_x * slope_ci_high + scaling_res["intercept_a"],
        color="red",
        alpha=0.15,
        label="95% Bootstrap CI",
    )

    # Mark width 4096 prediction
    ax2.scatter(
        [12.0],
        [scaling_res["pred_log2_4096"]],
        color="blue",
        marker="D",
        s=80,
        zorder=6,
        label=f"Width 4096: {pred_lr_4096:.2e}",
    )
    ax2.set_xlabel(r"$\log_2(\mathrm{Width})$")
    ax2.set_ylabel(r"Optimal $\log_2(\mathrm{LR})$")
    ax2.set_title("Scaling Law Extrapolation to Width 4096")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "p5_lr_sweep.png", dpi=300)
    plt.close()

    out_payload = {
        "minima_table": minima_summary,
        "scaling_law": {
            "slope_b": slope_b,
            "slope_b_ci_95": [slope_ci_low, slope_ci_high],
            "intercept_a": scaling_res["intercept_a"],
            "pred_log2_4096": scaling_res["pred_log2_4096"],
            "pred_lr_4096": pred_lr_4096,
            "constrained_1_over_w_lr_4096": constrained_lr_4096,
            "w95_prediction_interval_log2": w95_log2,
            "basin_half_width_1024_log2": h_1024,
            "confidence_label": confidence_label,
            "ci_contains_minus_one": contains_neg_one,
            "practical_trial_range": [pred_lr_4096 * 0.5, pred_lr_4096 * 2.0],
        },
        "hypotheses_status": {
            "H3_optimal_lr_decreases_with_width": h3_supported,
            "H4_slope_between_minus_one_and_zero": h4_supported,
            "H5_loss_improves_with_width": h5_supported,
        },
        "fitted_vertices": {str(w): {"lr_star": minima_summary[str(w)]["lr_star"]} for w in widths},
    }

    atomic_save_json(res_dir / "p5_lr_sweep.json", out_payload)
    print(
        f"[Part 5] Done. Width-4096 predicted LR = {pred_lr_4096:.2e} ({confidence_label} confidence, slope b={slope_b:.2f})"
    )
    return out_payload


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser(description="Part 5: Learning Rate Sweeps across Widths")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device"
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--results-dir", default=None, help="Directory to store results")
    parser.add_argument("--figures-dir", default="figures", help="Directory to store figures")
    args = parser.parse_args()

    run_part5(
        device=args.device,
        use_synthetic=args.synthetic,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
