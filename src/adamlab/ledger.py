"""Parity ledger management, CSV serialization, and invariant validation."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel


class ParityLedgerEntry(BaseModel):
    """Single comparison arm entry in the parity ledger."""

    comparison_id: str
    arm: str
    tuned_knobs: str
    grid_id: str
    n_grid_points: int
    n_seeds_tune: int
    n_trials: int
    select_split: str
    report_split: str
    report_seeds: str
    eval_step: int
    tuned_value: float


LEDGER_FIELDS = [
    "comparison_id",
    "arm",
    "tuned_knobs",
    "grid_id",
    "n_grid_points",
    "n_seeds_tune",
    "n_trials",
    "select_split",
    "report_split",
    "report_seeds",
    "eval_step",
    "tuned_value",
]


def write_ledger(
    entries: list[ParityLedgerEntry], filepath: str | Path = "results/parity_ledger.csv"
) -> None:
    """Save ledger entries to CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for e in entries:
            writer.writerow(e.model_dump())


def load_ledger(filepath: str | Path = "results/parity_ledger.csv") -> list[ParityLedgerEntry]:
    """Load and parse parity ledger CSV."""
    path = Path(filepath)
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                ParityLedgerEntry(
                    comparison_id=row["comparison_id"],
                    arm=row["arm"],
                    tuned_knobs=row["tuned_knobs"],
                    grid_id=row["grid_id"],
                    n_grid_points=int(row["n_grid_points"]),
                    n_seeds_tune=int(row["n_seeds_tune"]),
                    n_trials=int(row["n_trials"]),
                    select_split=row["select_split"],
                    report_split=row["report_split"],
                    report_seeds=row["report_seeds"],
                    eval_step=int(row["eval_step"]),
                    tuned_value=float(row["tuned_value"]),
                )
            )
    return entries


def validate_parity_ledger(entries: list[ParityLedgerEntry]) -> list[str]:
    """Validate strict parity invariants across all comparisons.

    Returns:
        List of violation messages (empty if all checks pass).
    """
    violations: list[str] = []
    by_comp: dict[str, list[ParityLedgerEntry]] = {}
    for e in entries:
        by_comp.setdefault(e.comparison_id, []).append(e)

    for comp_id, arms in by_comp.items():
        if len(arms) < 2:
            violations.append(f"Comparison {comp_id} has fewer than 2 arms ({len(arms)}).")
            continue

        base_arm = arms[0]
        for arm in arms:
            # Check 1: select_split != report_split
            if arm.select_split == arm.report_split:
                violations.append(
                    f"Arm '{arm.arm}' in {comp_id} uses same split for tuning and reporting: {arm.select_split}"
                )

            # Check 2: Same grid_id
            if arm.grid_id != base_arm.grid_id:
                violations.append(
                    f"Arms in {comp_id} have differing grid_ids: {arm.arm}={arm.grid_id} vs {base_arm.arm}={base_arm.grid_id}"
                )

            # Check 3: Equal n_trials
            if arm.n_trials != base_arm.n_trials:
                violations.append(
                    f"Arms in {comp_id} have unequal trial budgets: {arm.arm}={arm.n_trials} vs {base_arm.arm}={base_arm.n_trials}"
                )

            # Check 4: Equal evaluation step
            if arm.eval_step != base_arm.eval_step:
                violations.append(
                    f"Arms in {comp_id} evaluated at differing steps: {arm.arm}={arm.eval_step} vs {base_arm.arm}={base_arm.eval_step}"
                )

    return violations
