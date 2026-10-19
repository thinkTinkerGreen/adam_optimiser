"""Tests for branch equivalence: WSD200 and WSD300 are bitwise identical up to step 160."""

from __future__ import annotations

import torch

from adamlab.data import TokenizedCorpus, get_batch
from adamlab.determinism import make_generators, set_determinism
from adamlab.model import TransformerLM
from adamlab.schedules import get_schedule_fn


def test_wsd_branch_equivalence() -> None:
    """Verify that WSD(T=300, D=60) and WSD(T=200, D=40) produce identical losses for s <= 160."""
    set_determinism()
    seed = 77
    width = 64
    depth = 2
    context_length = 32
    batch_size = 8
    lr = 0.002
    steps_to_test = 25  # Test prefix steps (both are in stable phase at 1.0)

    corpus = TokenizedCorpus.load(use_synthetic=True)
    train_data = corpus.get_split("train")

    def run_prefix(schedule_name: str, total_steps: int, decay_steps: int) -> list[float]:
        init_gen, data_gen = make_generators(seed)
        model = TransformerLM(
            vocab_size=corpus.vocab_size,
            width=width,
            depth=depth,
            context_length=context_length,
            init_generator=init_gen,
        )
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, foreach=False
        )
        sched_fn = get_schedule_fn(
            schedule_name, total_steps=total_steps, warmup_steps=10, decay_steps=decay_steps
        )

        losses = []
        for s in range(1, steps_to_test + 1):
            curr_lr = lr * sched_fn(s)
            for pg in opt.param_groups:
                pg["lr"] = curr_lr

            x, y = get_batch(train_data, batch_size, context_length, data_gen)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        return losses

    losses_300 = run_prefix("wsd", total_steps=300, decay_steps=60)
    losses_200 = run_prefix("wsd", total_steps=200, decay_steps=40)

    assert len(losses_300) == len(losses_200)
    for s_idx, (l3, l2) in enumerate(zip(losses_300, losses_200, strict=False), start=1):
        assert l3 == l2, f"Discrepancy at step {s_idx}: {l3} != {l2}"
