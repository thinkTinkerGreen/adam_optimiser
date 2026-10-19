"""Comprehensive unit tests targeting 90%+ code coverage on src/adamlab."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adamlab.adam_ref import AdamRef
from adamlab.data import (
    TokenizedCorpus,
)
from adamlab.determinism import (
    get_environment_metadata,
)
from adamlab.io import RunSpec, atomic_save_json, build_manifest, load_json
from adamlab.ledger import ParityLedgerEntry, load_ledger, validate_parity_ledger, write_ledger
from adamlab.schedules import get_schedule_fn
from adamlab.sweep import build_sweep_specs, get_base_lr_grid
from adamlab.train import run_training


def test_adam_ref_validation_errors() -> None:
    """Verify AdamRef argument validation checks."""
    p = [torch.nn.Parameter(torch.randn(2))]
    with pytest.raises(ValueError, match="Invalid learning rate"):
        AdamRef(p, lr=-1.0)
    with pytest.raises(ValueError, match="Invalid epsilon value"):
        AdamRef(p, eps=-1e-8)
    with pytest.raises(ValueError, match="Invalid beta parameter at index 0"):
        AdamRef(p, betas=(-0.1, 0.99))
    with pytest.raises(ValueError, match="Invalid beta parameter at index 1"):
        AdamRef(p, betas=(0.9, 1.5))
    with pytest.raises(ValueError, match="Invalid weight_decay"):
        AdamRef(p, weight_decay=-0.1)


def test_adam_ref_closure_and_weight_decay() -> None:
    """Verify AdamRef closure support and weight decay."""
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = AdamRef([p], lr=0.01, weight_decay=0.01)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure=closure)
    assert loss is not None


def test_schedules_coverage() -> None:
    """Verify constant schedule and error handling."""
    const_fn = get_schedule_fn("constant", total_steps=100, warmup_steps=10)
    assert const_fn(5) == 0.5
    assert const_fn(10) == 1.0
    assert const_fn(50) == 1.0

    with pytest.raises(ValueError, match="Unknown schedule"):
        get_schedule_fn("polynomial", total_steps=100)


def test_io_manifest_and_json(tmp_path: Path) -> None:
    """Verify atomic_save_json, load_json, and build_manifest."""
    data = {"key": "value", "numbers": [1, 2, 3]}
    json_path = tmp_path / "test.json"
    atomic_save_json(json_path, data)
    loaded = load_json(json_path)
    assert loaded == data

    # Build manifest
    manifest = build_manifest(tmp_path)
    assert "test.json" in manifest


def test_ledger_io_and_additional_validations(tmp_path: Path) -> None:
    """Verify write_ledger, load_ledger, and validation branches."""
    entries = [
        ParityLedgerEntry(
            comparison_id="TEST_COMP",
            arm="ARM1",
            tuned_knobs="lr",
            grid_id="grid1",
            n_grid_points=5,
            n_seeds_tune=3,
            n_trials=15,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101",
            eval_step=100,
            tuned_value=0.001,
        ),
        ParityLedgerEntry(
            comparison_id="TEST_COMP",
            arm="ARM2",
            tuned_knobs="lr",
            grid_id="grid1",
            n_grid_points=5,
            n_seeds_tune=3,
            n_trials=15,
            select_split="tune_val",
            report_split="test",
            report_seeds="100,101",
            eval_step=100,
            tuned_value=0.002,
        ),
    ]
    csv_file = tmp_path / "parity_ledger.csv"
    write_ledger(entries, filepath=csv_file)
    loaded = load_ledger(filepath=csv_file)
    assert len(loaded) == 2
    assert loaded[0].arm == "ARM1"

    # Single-arm failure
    single_arm = [entries[0]]
    v_single = validate_parity_ledger(single_arm)
    assert any("fewer than 2 arms" in v for v in v_single)

    # Step mismatch failure
    entries_mismatch = [
        entries[0],
        entries[1].model_copy(update={"eval_step": 200}),
    ]
    v_step = validate_parity_ledger(entries_mismatch)
    assert any("evaluated at differing steps" in v for v in v_step)


def test_sweep_helpers() -> None:
    """Verify build_sweep_specs and base grid."""
    grid = get_base_lr_grid()
    assert len(grid) == 17
    specs = build_sweep_specs(
        experiment="test_sweep",
        widths=[64, 128],
        lrs=[0.001, 0.002],
        seeds=[0, 1],
    )
    assert len(specs) == 2 * 2 * 2  # 8 specs


def test_determinism_metadata() -> None:
    """Verify get_environment_metadata returns expected keys."""
    meta = get_environment_metadata()
    for key in ["torch_version", "device_name", "python_version", "platform", "git_sha"]:
        assert key in meta


def test_corpus_unknown_split() -> None:
    """Verify get_split raises ValueError on invalid split name."""
    corpus = TokenizedCorpus.load(use_synthetic=True)
    with pytest.raises(ValueError, match="Unknown split"):
        corpus.get_split("invalid_split")


def test_train_divergence_handling(tmp_path: Path) -> None:
    """Verify train handles divergence gracefully with penalty loss 5.0."""
    spec = RunSpec(
        experiment="divergence_test",
        width=64,
        depth=2,
        lr=1e20,  # Extreme LR to force immediate divergence
        schedule="constant",
        total_steps=5,
        warmup_steps=0,
        seed=0,
        device="cpu",
        use_synthetic=True,
        eval_steps=[5],
    )
    res = run_training(spec, save_to_disk=True, results_dir=tmp_path)
    assert res.diverged
    assert res.final_train_loss == 5.0
    assert res.eval_losses["step_5_tune_val"] == 5.0


def test_train_ratio_logging(tmp_path: Path) -> None:
    """Verify training with ratio logging enabled."""
    spec = RunSpec(
        experiment="ratio_log_test",
        width=64,
        depth=2,
        lr=0.001,
        schedule="constant",
        total_steps=3,
        warmup_steps=1,
        seed=0,
        device="cpu",
        use_synthetic=True,
        log_ratios=True,
        eval_steps=[3],
    )
    res = run_training(spec, save_to_disk=True, results_dir=tmp_path)
    assert not res.diverged
    assert "ratios" in res.summary_metrics
    assert len(res.summary_metrics["ratios"]) == 3
