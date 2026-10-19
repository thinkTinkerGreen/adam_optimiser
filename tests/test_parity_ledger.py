"""Tests for the parity ledger validator and schema consistency."""

from __future__ import annotations

from adamlab.ledger import ParityLedgerEntry, validate_parity_ledger


def test_parity_ledger_valid_case() -> None:
    """Valid matched-budget comparison passes validation without errors."""
    entries = [
        ParityLedgerEntry(
            comparison_id="P4_MATCHED_200",
            arm="COS200",
            tuned_knobs="peak_lr",
            grid_id="grid_half_octave_17",
            n_grid_points=17,
            n_seeds_tune=3,
            n_trials=51,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=0.002,
        ),
        ParityLedgerEntry(
            comparison_id="P4_MATCHED_200",
            arm="WSD200",
            tuned_knobs="peak_lr",
            grid_id="grid_half_octave_17",
            n_grid_points=17,
            n_seeds_tune=3,
            n_trials=51,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=0.002,
        ),
    ]
    violations = validate_parity_ledger(entries)
    assert not violations


def test_parity_ledger_catches_unequal_trials() -> None:
    """Validator catches unequal trial count between arms."""
    entries = [
        ParityLedgerEntry(
            comparison_id="COMP_FAIL",
            arm="ARM_A",
            tuned_knobs="peak_lr",
            grid_id="grid_A",
            n_grid_points=10,
            n_seeds_tune=3,
            n_trials=30,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=0.001,
        ),
        ParityLedgerEntry(
            comparison_id="COMP_FAIL",
            arm="ARM_B",
            tuned_knobs="peak_lr",
            grid_id="grid_A",
            n_grid_points=5,
            n_seeds_tune=3,
            n_trials=15,  # Unequal!
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=0.001,
        ),
    ]
    violations = validate_parity_ledger(entries)
    assert any("unequal trial budgets" in v for v in violations)


def test_parity_ledger_catches_data_leakage() -> None:
    """Validator catches select_split == report_split."""
    entries = [
        ParityLedgerEntry(
            comparison_id="COMP_LEAK",
            arm="ARM_A",
            tuned_knobs="peak_lr",
            grid_id="grid_A",
            n_grid_points=10,
            n_seeds_tune=3,
            n_trials=30,
            select_split="test",
            report_split="test",  # Data leakage!
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=0.001,
        ),
        ParityLedgerEntry(
            comparison_id="COMP_LEAK",
            arm="ARM_B",
            tuned_knobs="peak_lr",
            grid_id="grid_A",
            n_grid_points=10,
            n_seeds_tune=3,
            n_trials=30,
            select_split="test",
            report_split="test",
            report_seeds="100,101,102",
            eval_step=200,
            tuned_value=0.001,
        ),
    ]
    violations = validate_parity_ledger(entries)
    assert any("uses same split for tuning and reporting" in v for v in violations)
