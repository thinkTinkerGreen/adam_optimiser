"""Part 4: Cosine vs WSD: Literal (Step 200 of 300) and Matched-Budget (T=200) Parity Comparisons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from adamlab.io import RunSpec, atomic_save_json
from adamlab.ledger import ParityLedgerEntry, write_ledger
from adamlab.stats import compute_paired_difference
from adamlab.sweep import get_base_lr_grid
from adamlab.train import run_training


def tune_peak_lr(
    experiment_name: str,
    schedule: str,
    total_steps: int,
    warmup_steps: int = 30,
    decay_steps: int | None = None,
    eval_step: int = 200,
    width: int = 512,
    device: str = "cpu",
    use_synthetic: bool = False,
    results_dir: str | Path = "results",
) -> tuple[float, list[float], dict[float, list[float]]]:
    """Tune peak LR over half-octave base grid on tune_val split at eval_step."""
    res_dir = Path(results_dir)
    grid = get_base_lr_grid()
    tune_seeds = [0, 1, 2]

    losses_by_lr: dict[float, list[float]] = {}
    mean_losses: list[float] = []

    for lr in grid:
        losses_by_lr[lr] = []
        for s in tune_seeds:
            spec = RunSpec(
                experiment=experiment_name,
                width=width,
                lr=lr,
                schedule=schedule,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                seed=s,
                device=device,
                use_synthetic=use_synthetic,
                eval_steps=[eval_step],
            )
            res = run_training(spec, save_to_disk=True, results_dir=res_dir)
            val_loss = res.eval_losses.get(f"step_{eval_step}_tune_val", 5.0)
            losses_by_lr[lr].append(val_loss)
        mean_losses.append(float(np.mean(losses_by_lr[lr])))

    best_idx = int(np.argmin(mean_losses))
    best_lr = grid[best_idx]
    return best_lr, mean_losses, losses_by_lr


def evaluate_on_test(
    experiment_name: str,
    schedule: str,
    total_steps: int,
    lr: float,
    warmup_steps: int = 30,
    decay_steps: int | None = None,
    eval_step: int = 200,
    width: int = 512,
    device: str = "cpu",
    use_synthetic: bool = False,
    results_dir: str | Path = "results",
) -> list[float]:
    """Evaluate tuned model on held-out test split using report seeds {100, 101, 102}."""
    res_dir = Path(results_dir)
    report_seeds = [100, 101, 102]
    test_losses = []

    for s in report_seeds:
        spec = RunSpec(
            experiment=experiment_name,
            width=width,
            lr=lr,
            schedule=schedule,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            seed=s,
            device=device,
            use_synthetic=use_synthetic,
            eval_steps=[eval_step],
        )
        res = run_training(spec, save_to_disk=True, results_dir=res_dir)
        test_losses.append(res.eval_losses.get(f"step_{eval_step}_test", 5.0))

    return test_losses


def run_part4(
    device: str = "cpu",
    use_synthetic: bool = False,
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Execute Part 4 literal and matched-budget comparisons under tuning parity."""
    res_dir = Path(results_dir)
    fig_dir = Path(figures_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    grid_len = len(get_base_lr_grid())  # 17
    n_tune_seeds = 3
    n_trials = grid_len * n_tune_seeds  # 51

    # 1. Tuning for Literal Comparison (T=300 stopped at 200)
    best_lr_cos300, _, _ = tune_peak_lr(
        "p5_sweep",
        "cosine",
        total_steps=300,
        eval_step=200,
        device=device,
        use_synthetic=use_synthetic,
        results_dir=res_dir,
    )
    best_lr_wsd300, _, _ = tune_peak_lr(
        "p4_wsd300_tune",
        "wsd",
        total_steps=300,
        decay_steps=60,
        eval_step=200,
        device=device,
        use_synthetic=use_synthetic,
        results_dir=res_dir,
    )

    # 2. Tuning for Matched Budget Comparison (T=200)
    best_lr_cos200, _, _ = tune_peak_lr(
        "p4_cos200_tune",
        "cosine",
        total_steps=200,
        eval_step=200,
        device=device,
        use_synthetic=use_synthetic,
        results_dir=res_dir,
    )
    best_lr_wsd200, _, _ = tune_peak_lr(
        "p4_wsd200_tune",
        "wsd",
        total_steps=200,
        decay_steps=40,
        eval_step=200,
        device=device,
        use_synthetic=use_synthetic,
        results_dir=res_dir,
    )

    # 3. Held-out Test Evaluations on Report Seeds {100, 101, 102}
    test_cos300 = evaluate_on_test(
        "p4_cos300_rep",
        "cosine",
        300,
        best_lr_cos300,
        eval_step=200,
        device=device,
        use_synthetic=use_synthetic,
        results_dir=res_dir,
    )
    test_wsd300 = evaluate_on_test(
        "p4_wsd300_rep",
        "wsd",
        300,
        best_lr_wsd300,
        decay_steps=60,
        eval_step=200,
        device=device,
        use_synthetic=use_synthetic,
        results_dir=res_dir,
    )

    test_cos200 = evaluate_on_test(
        "p4_cos200_rep",
        "cosine",
        200,
        best_lr_cos200,
        eval_step=200,
        device=device,
        use_synthetic=use_synthetic,
        results_dir=res_dir,
    )
    test_wsd200 = evaluate_on_test(
        "p4_wsd200_rep",
        "wsd",
        200,
        best_lr_wsd200,
        decay_steps=40,
        eval_step=200,
        device=device,
        use_synthetic=use_synthetic,
        results_dir=res_dir,
    )

    # 4. Paired Statistics
    paired_d1 = compute_paired_difference(test_cos300, test_wsd300)
    paired_d2 = compute_paired_difference(test_cos200, test_wsd200)

    # Pre-registered Decision Rules (§7.5)
    def evaluate_decision(paired: dict[str, float]) -> str:
        delta = paired["mean_diff"]  # loss_cosine - loss_wsd
        se = paired["se"]
        is_decisive = abs(delta) > (2.0 * se) and abs(delta) > 0.01
        if not is_decisive:
            return "TIE"
        return "COSINE" if delta < 0.0 else "WSD"

    d1_verdict = evaluate_decision(paired_d1)
    d2_verdict = evaluate_decision(paired_d2)

    # KEEP Decision:
    # If D2 favours cosine -> keep cosine. If D2 is tie or favours WSD -> keep WSD
    if d2_verdict == "COSINE":
        keep_decision = "Cosine"
        keep_rationale = "Cosine showed statistically decisive performance gain at matched budget."
    elif d2_verdict == "WSD":
        keep_decision = "WSD"
        keep_rationale = "WSD demonstrated decisive loss advantage at matched budget."
    else:
        keep_decision = "WSD"
        keep_rationale = (
            "Matched budget comparison between Cosine and WSD is a statistical tie. "
            "WSD is kept because it enables zero-restart cooldowns and checkpoint reuse across arbitrary training durations."
        )

    # Hypotheses evaluation
    h1_supported = bool(np.mean(test_cos300) < np.mean(test_wsd300))
    h2_supported = bool(d2_verdict == "TIE")

    # 5. Populate and Update Parity Ledger
    ledger_entries = [
        ParityLedgerEntry(
            comparison_id="P4_LITERAL_200",
            arm="COS300",
            tuned_knobs="peak_lr",
            grid_id="grid_half_octave_17",
            n_grid_points=grid_len,
            n_seeds_tune=n_tune_seeds,
            n_trials=n_trials,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=best_lr_cos300,
        ),
        ParityLedgerEntry(
            comparison_id="P4_LITERAL_200",
            arm="WSD300",
            tuned_knobs="peak_lr",
            grid_id="grid_half_octave_17",
            n_grid_points=grid_len,
            n_seeds_tune=n_tune_seeds,
            n_trials=n_trials,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=best_lr_wsd300,
        ),
        ParityLedgerEntry(
            comparison_id="P4_MATCHED_200",
            arm="COS200",
            tuned_knobs="peak_lr",
            grid_id="grid_half_octave_17",
            n_grid_points=grid_len,
            n_seeds_tune=n_tune_seeds,
            n_trials=n_trials,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=best_lr_cos200,
        ),
        ParityLedgerEntry(
            comparison_id="P4_MATCHED_200",
            arm="WSD200",
            tuned_knobs="peak_lr",
            grid_id="grid_half_octave_17",
            n_grid_points=grid_len,
            n_seeds_tune=n_tune_seeds,
            n_trials=n_trials,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=best_lr_wsd200,
        ),
    ]
    write_ledger(ledger_entries, filepath=res_dir / "parity_ledger.csv")

    # 6. Generate Figures
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )

    # Figure 1: p4_curves.png (Schedule vs Loss)
    from adamlab.schedules import get_schedule_fn

    cos300_fn = get_schedule_fn("cosine", 300, 30)
    wsd300_fn = get_schedule_fn("wsd", 300, 30, decay_steps=60)
    steps_300 = np.arange(1, 301)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_top.plot(
        steps_300,
        [best_lr_cos300 * cos300_fn(s) for s in steps_300],
        label=f"COS300 (peak {best_lr_cos300:.2e})",
        color="#1f77b4",
    )
    ax_top.plot(
        steps_300,
        [best_lr_wsd300 * wsd300_fn(s) for s in steps_300],
        label=f"WSD300 (peak {best_lr_wsd300:.2e})",
        color="#ff7f0e",
        linestyle="--",
    )
    ax_top.axvline(200, color="black", linestyle=":", label="Stop Evaluation Step 200")
    ax_top.set_ylabel("Learning Rate")
    ax_top.set_title("Part 4: Learning Rate Trajectories (COS300 vs WSD300)")
    ax_top.legend()

    # Bottom: bar plot of test loss at step 200
    mean_cos300 = float(np.mean(test_cos300))
    mean_wsd300 = float(np.mean(test_wsd300))
    se_cos300 = float(np.std(test_cos300, ddof=1) / math.sqrt(len(test_cos300)))
    se_wsd300 = float(np.std(test_wsd300, ddof=1) / math.sqrt(len(test_wsd300)))

    ax_bot.bar(
        [1, 2],
        [mean_cos300, mean_wsd300],
        yerr=[se_cos300, se_wsd300],
        capsize=5,
        color=["#1f77b4", "#ff7f0e"],
        alpha=0.8,
        width=0.4,
    )
    ax_bot.set_xticks([1, 2])
    ax_bot.set_xticklabels(["COS300 at Step 200", "WSD300 at Step 200"])
    ax_bot.set_ylabel("Test Loss at Step 200 (nats)")
    ax_bot.set_title(
        f"Literal Comparison: Δ = {paired_d1['mean_diff']:.4f} ± {paired_d1['se']:.4f} nats ({d1_verdict})"
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "p4_curves.png", dpi=300)
    plt.close()

    # Figure 2: p4_matched.png (COS200 vs WSD200)
    fig, ax = plt.subplots(figsize=(7, 5))
    mean_cos200 = float(np.mean(test_cos200))
    mean_wsd200 = float(np.mean(test_wsd200))
    se_cos200 = float(np.std(test_cos200, ddof=1) / math.sqrt(len(test_cos200)))
    se_wsd200 = float(np.std(test_wsd200, ddof=1) / math.sqrt(len(test_wsd200)))

    ax.bar(
        [1, 2],
        [mean_cos200, mean_wsd200],
        yerr=[se_cos200, se_wsd200],
        capsize=5,
        color=["#2ca02c", "#d62728"],
        alpha=0.8,
        width=0.4,
    )
    for _i, (l_cos, l_wsd) in enumerate(zip(test_cos200, test_wsd200, strict=False)):
        ax.plot([1, 2], [l_cos, l_wsd], "k-o", alpha=0.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["COS200 (Matched)", "WSD200 (Matched)"])
    ax.set_ylabel("Held-Out Test Loss at Step 200 (nats)")
    ax.set_title(
        f"Matched Budget: Δ = {paired_d2['mean_diff']:.4f} ± {paired_d2['se']:.4f} nats ({d2_verdict})"
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "p4_matched.png", dpi=300)
    plt.close()

    out_payload = {
        "literal_comparison_d1": {
            "cos300_tuned_lr": best_lr_cos300,
            "wsd300_tuned_lr": best_lr_wsd300,
            "cos300_test_losses": test_cos300,
            "wsd300_test_losses": test_wsd300,
            "cos300_mean": mean_cos300,
            "cos300_se": se_cos300,
            "wsd300_mean": mean_wsd300,
            "wsd300_se": se_wsd300,
            "paired_stats": paired_d1,
            "verdict": d1_verdict,
        },
        "matched_budget_comparison_d2": {
            "cos200_tuned_lr": best_lr_cos200,
            "wsd200_tuned_lr": best_lr_wsd200,
            "cos200_test_losses": test_cos200,
            "wsd200_test_losses": test_wsd200,
            "cos200_mean": mean_cos200,
            "cos200_se": se_cos200,
            "wsd200_mean": mean_wsd200,
            "wsd200_se": se_wsd200,
            "paired_stats": paired_d2,
            "verdict": d2_verdict,
        },
        "keep_recommendation": {
            "chosen_schedule": keep_decision,
            "rationale": keep_rationale,
        },
        "hypotheses_status": {
            "H1_cosine_lower_in_literal_stop": h1_supported,
            "H2_matched_budget_statistical_tie": h2_supported,
        },
    }

    atomic_save_json(res_dir / "p4_cosine_vs_wsd.json", out_payload)
    print(
        f"[Part 4] Done. D1 (Literal): {d1_verdict}, D2 (Matched): {d2_verdict} -> KEEP: {keep_decision}"
    )
    return out_payload


if __name__ == "__main__":
    run_part4()
