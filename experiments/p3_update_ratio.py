"""Part 3: Layer-wise Update-to-Weight Ratio and Warmup Termination (t_w) Detection."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from adamlab.io import RunSpec, atomic_save_json, load_json
from adamlab.metrics import categorize_parameter
from adamlab.train import run_training


def centered_moving_average(arr: np.ndarray, window_size: int = 11) -> np.ndarray:
    """Centered moving average with edge truncation."""
    n = len(arr)
    smoothed = np.empty(n, dtype=np.float64)
    half = window_size // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed[i] = float(np.mean(arr[start:end]))
    return smoothed


def compute_t_w(
    r_w30: np.ndarray,
    r_w0: np.ndarray,
    thetas: list[float] | None = None,
) -> dict[str, int]:
    """Compute step index t_w after which discrepancy d(s) < theta for all s' >= s."""
    if thetas is None:
        thetas = [0.02, 0.05, 0.10]

    n_steps = len(r_w30)
    # Relative discrepancy: |r_W30 - r_W0| / r_W0
    eps = 1e-12
    disc = np.abs(r_w30 - r_w0) / np.maximum(r_w0, eps)

    results: dict[str, int] = {}
    for theta in thetas:
        t_found = n_steps
        for s in range(n_steps):
            if np.all(disc[s:] < theta):
                t_found = s + 1  # 1-based step index
                break
        results[f"theta_{int(theta * 100)}pct"] = t_found
    return results


def run_part3(
    peak_lr: float | None = None,
    device: str = "cpu",
    use_synthetic: bool = False,
    results_dir: str | Path | None = None,
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Execute Part 3 experiments and compute layer-wise t_w metrics."""
    if results_dir is None:
        results_dir = os.environ.get("ADAMLAB_RESULTS_DIR", "results")
    res_dir = Path(results_dir)
    fig_dir = Path(figures_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Determine peak LR
    if peak_lr is None:
        p5_file = res_dir / "p5_lr_sweep.json"
        if p5_file.exists():
            p5_data = load_json(p5_file)
            peak_lr = float(p5_data["fitted_vertices"]["512"]["lr_star"])
        else:
            peak_lr = 2.0**-10.0  # 9.765625e-4 default

    seeds = [0, 1, 2]
    total_steps = 300
    print(
        f"[Part 3] Starting update-to-weight ratio tracking (peak_lr={peak_lr:.2e}, seeds={seeds}, device={device})..."
    )

    # 1. Run W30 (Warmup 30) and W0 (No Warmup)
    runs_w30 = []
    runs_w0 = []

    for s in seeds:
        print(f"  [Part 3] Running seed {s} (W30 and W0 arms)...")
        spec_w30 = RunSpec(
            experiment="p3_w30",
            width=512,
            lr=peak_lr,
            schedule="constant",
            total_steps=total_steps,
            warmup_steps=30,
            seed=s,
            device=device,
            use_synthetic=use_synthetic,
            log_ratios=True,
            eval_steps=[300],
        )
        spec_w0 = RunSpec(
            experiment="p3_w0",
            width=512,
            lr=peak_lr,
            schedule="constant",
            total_steps=total_steps,
            warmup_steps=0,
            seed=s,
            device=device,
            use_synthetic=use_synthetic,
            log_ratios=True,
            eval_steps=[300],
        )

        res_w30 = run_training(spec_w30, save_to_disk=True, results_dir=res_dir)
        res_w0 = run_training(spec_w0, save_to_disk=True, results_dir=res_dir)

        # Fallback check per Section 5.2 if W0 diverged
        if res_w0.diverged:
            print("[Part 3] W0 diverged at peak LR. Retrying both arms at 0.5x peak LR...")
            peak_lr = 0.5 * peak_lr
            spec_w30.lr = peak_lr
            spec_w0.lr = peak_lr
            res_w30 = run_training(spec_w30, save_to_disk=True, results_dir=res_dir)
            res_w0 = run_training(spec_w0, save_to_disk=True, results_dir=res_dir)

        runs_w30.append(res_w30)
        runs_w0.append(res_w0)

    # 2. Extract per-tensor ratios and average over seeds
    sample_ratios = runs_w30[0].summary_metrics["ratios"][0]
    tensor_names = [k for k in sample_ratios.keys() if k not in ("step", "lr")]

    # Array shape: (n_tensors, total_steps)
    r_w30_avg = {name: np.zeros(total_steps, dtype=np.float64) for name in tensor_names}
    r_w0_avg = {name: np.zeros(total_steps, dtype=np.float64) for name in tensor_names}

    for res in runs_w30:
        for step_idx, step_dict in enumerate(res.summary_metrics["ratios"]):
            for name in tensor_names:
                r_w30_avg[name][step_idx] += step_dict[name] / len(runs_w30)

    for res in runs_w0:
        for step_idx, step_dict in enumerate(res.summary_metrics["ratios"]):
            for name in tensor_names:
                r_w0_avg[name][step_idx] += step_dict[name] / len(runs_w0)

    # Smooth curves and compute t_w per tensor
    tw_per_tensor = {}
    smoothed_w30 = {}
    smoothed_w0 = {}

    for name in tensor_names:
        s_w30 = centered_moving_average(r_w30_avg[name], window_size=11)
        s_w0 = centered_moving_average(r_w0_avg[name], window_size=11)
        smoothed_w30[name] = s_w30
        smoothed_w0[name] = s_w0
        tw_per_tensor[name] = compute_t_w(s_w30, s_w0, thetas=[0.02, 0.05, 0.10])

    # Compute global t_w (max over tensors at theta=5%) and median
    tw_5_vals = [m["theta_5pct"] for m in tw_per_tensor.values()]
    global_tw_5 = int(max(tw_5_vals))
    median_tw_5 = int(np.median(tw_5_vals))

    # Per-family averages
    families = ["embed", "ln", "qkv", "proj", "fc1", "fc2", "head"]
    family_tw: dict[str, float] = {}
    family_curves_w30: dict[str, np.ndarray] = {f: np.zeros(total_steps) for f in families}
    family_curves_w0: dict[str, np.ndarray] = {f: np.zeros(total_steps) for f in families}
    family_counts: dict[str, int] = dict.fromkeys(families, 0)

    for name in tensor_names:
        fam = categorize_parameter(name)
        if fam in families:
            family_curves_w30[fam] += smoothed_w30[name]
            family_curves_w0[fam] += smoothed_w0[name]
            family_counts[fam] += 1

    for fam in families:
        if family_counts[fam] > 0:
            family_curves_w30[fam] /= family_counts[fam]
            family_curves_w0[fam] /= family_counts[fam]
            fam_tw = compute_t_w(family_curves_w30[fam], family_curves_w0[fam], thetas=[0.05])
            family_tw[fam] = float(fam_tw["theta_5pct"])

    # 3. Sanity checks
    # Check 1: In first step of W0, r_i(1) ~ lr * sqrt(numel) / ||w||
    # Compare with actual
    # Check 2: W30 correlation with s/30 for s <= 30
    w30_warmup_corrs = []
    ideal_warmup = np.arange(1, 31) / 30.0

    for name in tensor_names:
        corr = float(np.corrcoef(r_w30_avg[name][:30], ideal_warmup)[0, 1])
        if not math.isnan(corr):
            w30_warmup_corrs.append(corr)

    mean_warmup_corr = float(np.mean(w30_warmup_corrs)) if w30_warmup_corrs else 0.0

    # 4. Generate Figures
    # Figure 1: Ratio by family
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    steps_x = np.arange(1, total_steps + 1)
    for fam in families:
        ax.plot(steps_x, family_curves_w30[fam], label=f"{fam} (W30)", linewidth=1.8)
        ax.plot(steps_x, family_curves_w0[fam], linestyle=":", alpha=0.6)

    ax.axvline(30, color="red", linestyle="--", label="Warmup End (Step 30)")
    ax.axvline(
        global_tw_5, color="black", linestyle="-.", label=f"Global $t_w$ (Step {global_tw_5})"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Update-to-Weight Ratio $r(s)$ (log scale)")
    ax.set_title(r"Layer-wise Update-to-Weight Ratio by Parameter Family (W30 vs W0)")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "p3_ratio_by_family.png", dpi=300)
    plt.close()

    # Figure 2: Heatmap (Tensors x Steps)
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_matrix = np.array([np.log10(np.maximum(smoothed_w30[t], 1e-8)) for t in tensor_names])
    cax = ax.imshow(heatmap_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    fig.colorbar(cax, label=r"$\log_{10}(r_i(s))$")
    ax.set_yticks(range(len(tensor_names)))
    ax.set_yticklabels(tensor_names, fontsize=7)
    ax.set_xlabel("Step")
    ax.set_title("Layer Update Ratios Over Training Steps (W30)")
    ax.axvline(30, color="red", linestyle="--", label="Step 30")
    ax.axvline(global_tw_5, color="white", linestyle="-.", label=f"Global $t_w={global_tw_5}$")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "p3_ratio_heatmap.png", dpi=300)
    plt.close()

    out_payload = {
        "peak_lr_used": peak_lr,
        "global_tw_5pct": global_tw_5,
        "median_tw_5pct": median_tw_5,
        "family_tw_5pct": family_tw,
        "mean_warmup_correlation": mean_warmup_corr,
        "tensor_tw_sample": {k: tw_per_tensor[k] for k in list(tw_per_tensor.keys())[:5]},
    }

    atomic_save_json(res_dir / "p3_update_ratio.json", out_payload)
    print(f"[Part 3] Done. Global t_w(5%) = {global_tw_5}, Median = {median_tw_5}")
    return out_payload


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser(description="Part 3: Layer-wise Update-to-Weight Ratio")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device"
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--results-dir", default=None, help="Directory to store results")
    parser.add_argument("--figures-dir", default="figures", help="Directory to store figures")
    args = parser.parse_args()

    run_part3(
        device=args.device,
        use_synthetic=args.synthetic,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
