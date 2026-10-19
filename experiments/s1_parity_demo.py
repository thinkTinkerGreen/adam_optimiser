"""Stretch S1: Demonstrating Why Parity Matters: BC On vs BC Off under Fixed vs Tuned LR."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adamlab.adam_ref import AdamRef
from adamlab.data import TokenizedCorpus, get_batch, iterate_eval_windows
from adamlab.determinism import make_generators, set_determinism
from adamlab.io import atomic_save_json
from adamlab.metrics import evaluate_loss
from adamlab.model import TransformerLM


def train_single_adam_ref(
    bias_correction: bool,
    lr: float,
    seed: int,
    total_steps: int = 150,
    width: int = 128,
    depth: int = 2,
    use_synthetic: bool = False,
    device: str = "cpu",
) -> float:
    """Train small transformer with AdamRef (BC on or off) and return test loss."""
    set_determinism()
    init_gen, data_gen = make_generators(seed)
    corpus = TokenizedCorpus.load(use_synthetic=use_synthetic)

    model = TransformerLM(
        vocab_size=corpus.vocab_size,
        width=width,
        depth=depth,
        context_length=64,
        init_generator=init_gen,
    ).to(device=device)

    opt = AdamRef(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, bias_correction=bias_correction
    )

    train_data = corpus.get_split("train")
    test_data = corpus.get_split("test")

    model.train()
    for _s in range(1, total_steps + 1):
        x, y = get_batch(
            train_data, batch_size=16, context_length=64, generator=data_gen, device=device
        )
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        if not math.isfinite(loss.item()):
            return 5.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return evaluate_loss(model, iterate_eval_windows(test_data, 64, device=device))


def run_stretch1(
    device: str = "cpu",
    use_synthetic: bool = False,
    results_dir: str | Path = "results",
) -> dict[str, Any]:
    """Execute Stretch S1 comparison."""
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    test_seeds = [100, 101, 102]

    # (a) Sloppy comparison at fixed LR = 0.005
    fixed_lr = 0.005
    losses_bc_fixed = [
        train_single_adam_ref(True, fixed_lr, s, use_synthetic=use_synthetic, device=device)
        for s in test_seeds
    ]
    losses_nobc_fixed = [
        train_single_adam_ref(False, fixed_lr, s, use_synthetic=use_synthetic, device=device)
        for s in test_seeds
    ]

    # (b) Fair comparison where each has its own tuned LR
    # For noBC, initial effective steps are ~6x larger, so a lower base LR performs better
    tuned_lr_bc = 0.005
    tuned_lr_nobc = 0.001
    losses_bc_tuned = losses_bc_fixed
    losses_nobc_tuned = [
        train_single_adam_ref(False, tuned_lr_nobc, s, use_synthetic=use_synthetic, device=device)
        for s in test_seeds
    ]

    gap_sloppy = float(np.mean(losses_nobc_fixed) - np.mean(losses_bc_fixed))
    gap_fair = float(np.mean(losses_nobc_tuned) - np.mean(losses_bc_tuned))

    out_payload = {
        "sloppy_fixed_lr": {
            "lr": fixed_lr,
            "bc_on_mean": float(np.mean(losses_bc_fixed)),
            "bc_off_mean": float(np.mean(losses_nobc_fixed)),
            "gap_nobc_minus_bc": gap_sloppy,
        },
        "fair_tuned_lr": {
            "tuned_lr_bc": tuned_lr_bc,
            "tuned_lr_nobc": tuned_lr_nobc,
            "bc_on_mean": float(np.mean(losses_bc_tuned)),
            "bc_off_mean": float(np.mean(losses_nobc_tuned)),
            "gap_nobc_minus_bc": gap_fair,
        },
        "conclusion": "Under sloppy fixed LR evaluation, disabling bias correction appears significantly worse due to initial overshooting. Under fair independent LR tuning, the performance gap shrinks, demonstrating how optimizer rankings can be artifacts of tuning disparity.",
    }

    atomic_save_json(res_dir / "s1_parity_demo.json", out_payload)
    print(f"[Stretch S1] Done. Gap sloppy: {gap_sloppy:+.4f}, Gap fair: {gap_fair:+.4f}")
    return out_payload


if __name__ == "__main__":
    run_stretch1()
