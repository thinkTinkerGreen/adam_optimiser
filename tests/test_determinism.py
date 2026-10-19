"""Tests for determinism: duplicate runs produce bitwise identical losses."""

from __future__ import annotations

from adamlab.io import RunSpec
from adamlab.train import run_training


def test_determinism_same_runspec_twice() -> None:
    """Verify that two identical RunSpecs yield bitwise identical training and eval losses."""
    spec = RunSpec(
        experiment="test_determinism",
        width=64,
        depth=2,
        lr=0.001,
        schedule="cosine",
        total_steps=10,
        warmup_steps=2,
        seed=123,
        batch_size=8,
        context_length=32,
        device="cpu",
        use_synthetic=True,
        eval_steps=[10],
    )

    res1 = run_training(spec, save_to_disk=False)
    res2 = run_training(spec, save_to_disk=False)

    assert res1.final_train_loss == res2.final_train_loss
    assert res1.eval_losses["step_10_tune_val"] == res2.eval_losses["step_10_tune_val"]
    assert res1.eval_losses["step_10_test"] == res2.eval_losses["step_10_test"]
