"""End-to-end smoke test exercising the full pipeline in < 60s on CPU."""

from __future__ import annotations

from adamlab.io import RunSpec, build_manifest
from adamlab.ledger import ParityLedgerEntry, validate_parity_ledger
from adamlab.train import run_training
from experiments.p1_adam_by_hand import run_part1
from experiments.p2_bias_correction import run_part2


def run_smoke_test() -> None:
    print("=== [Smoke Test] 1. Running Part 1 (Adam by Hand) ===")
    p1_res = run_part1()
    assert p1_res["overall_max_err_fp64"] <= 1e-12

    print("=== [Smoke Test] 2. Running Part 2 (Bias Correction) ===")
    p2_res = run_part2()
    assert p2_res["sim_cross_check_0_999"]["simulation_5pct"] == 2375

    print("=== [Smoke Test] 3. Running Neural Net Training Run (Width 64, 20 steps) ===")
    spec = RunSpec(
        experiment="smoke_test",
        width=64,
        depth=2,
        lr=0.001,
        schedule="cosine",
        total_steps=20,
        warmup_steps=5,
        seed=42,
        batch_size=8,
        context_length=32,
        device="cpu",
        use_synthetic=True,
        eval_steps=[20],
    )
    run_res = run_training(spec, save_to_disk=True)
    assert not run_res.diverged
    assert "step_20_tune_val" in run_res.eval_losses

    print("=== [Smoke Test] 4. Validating Parity Ledger ===")
    sample_ledger = [
        ParityLedgerEntry(
            comparison_id="SMOKE_COMP",
            arm="ARM_1",
            tuned_knobs="lr",
            grid_id="smoke_grid",
            n_grid_points=2,
            n_seeds_tune=1,
            n_trials=2,
            select_split="tune_val",
            report_split="test",
            report_seeds="100",
            eval_step=20,
            tuned_value=0.001,
        ),
        ParityLedgerEntry(
            comparison_id="SMOKE_COMP",
            arm="ARM_2",
            tuned_knobs="lr",
            grid_id="smoke_grid",
            n_grid_points=2,
            n_seeds_tune=1,
            n_trials=2,
            select_split="tune_val",
            report_split="test",
            report_seeds="100",
            eval_step=20,
            tuned_value=0.001,
        ),
    ]
    violations = validate_parity_ledger(sample_ledger)
    assert not violations

    print("=== [Smoke Test] 5. Building Manifest ===")
    manifest = build_manifest("results")
    assert len(manifest) > 0

    print("=== [Smoke Test] PASSED successfully in all stages ===")


if __name__ == "__main__":
    run_smoke_test()
