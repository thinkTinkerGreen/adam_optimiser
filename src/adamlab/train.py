"""Single-run training engine executing a RunSpec to produce a RunResult."""

from __future__ import annotations

import csv
import gzip
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from adamlab.data import TokenizedCorpus, get_batch, iterate_eval_windows
from adamlab.determinism import get_environment_metadata, make_generators, set_determinism
from adamlab.io import RunResult, RunSpec, atomic_save_json
from adamlab.metrics import compute_update_ratios, evaluate_loss
from adamlab.model import TransformerLM
from adamlab.schedules import get_schedule_fn


def run_training(
    spec: RunSpec, save_to_disk: bool = True, results_dir: str | Path | None = None
) -> RunResult:
    """Execute a single deterministic training run specified by spec."""
    set_determinism()

    if results_dir is None:
        results_dir = os.environ.get("ADAMLAB_RESULTS_DIR", "results")
    out_dir = Path(results_dir) / "runs" / spec.run_id

    # If run has already completed and save_to_disk is enabled, return cached
    final_file = out_dir / "final.json"
    if save_to_disk and final_file.exists():
        from adamlab.io import load_json

        return RunResult.model_validate(load_json(final_file))

    # Initialize generators
    init_gen, data_gen = make_generators(spec.seed)

    # Load corpus
    corpus = TokenizedCorpus.load(use_synthetic=spec.use_synthetic)

    device = torch.device(spec.device)

    # Build model
    model = TransformerLM(
        vocab_size=corpus.vocab_size,
        width=spec.width,
        depth=spec.depth,
        context_length=spec.context_length,
        init_generator=init_gen,
    ).to(device=device, dtype=torch.float32)

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=spec.weight_decay,
        foreach=False,
    )

    # Build schedule
    schedule_fn = get_schedule_fn(
        schedule_name=spec.schedule,
        total_steps=spec.total_steps,
        warmup_steps=spec.warmup_steps,
        decay_steps=spec.decay_steps,
        f_min=spec.f_min,
    )

    train_data = corpus.get_split("train")
    val_data = corpus.get_split("tune_val")
    test_data = corpus.get_split("test")

    eval_losses: dict[str, float] = {}
    ratio_logs: list[dict[str, Any]] = []
    step_metrics: list[dict[str, Any]] = []

    diverged = False
    last_train_loss = 0.0

    model.train()

    for s in range(1, spec.total_steps + 1):
        mult = schedule_fn(s)
        curr_lr = spec.lr * mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        x, y = get_batch(train_data, spec.batch_size, spec.context_length, data_gen, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            diverged = True
            break

        loss.backward()  # type: ignore[no-untyped-call]

        # Compute grad norm and check finiteness
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip).item()
        if not math.isfinite(grad_norm):
            diverged = True
            break

        last_train_loss = loss_val

        # Snapshot parameters before step if logging ratios
        if spec.log_ratios:
            w_before = {name: p.detach().clone().cpu() for name, p in model.named_parameters()}
            optimizer.step()
            w_after = {name: p.detach().clone().cpu() for name, p in model.named_parameters()}
            ratios = compute_update_ratios(w_before, w_after)
            ratios["step"] = s
            ratios["lr"] = curr_lr
            ratio_logs.append(ratios)
        else:
            optimizer.step()

        step_metrics.append(
            {
                "step": s,
                "train_loss": loss_val,
                "lr": curr_lr,
                "grad_norm": grad_norm,
            }
        )

        # Scheduled evaluations
        if s in spec.eval_steps:
            val_loss = evaluate_loss(
                model, iterate_eval_windows(val_data, spec.context_length, device=device)
            )
            eval_losses[f"step_{s}_tune_val"] = val_loss
            test_loss = evaluate_loss(
                model, iterate_eval_windows(test_data, spec.context_length, device=device)
            )
            eval_losses[f"step_{s}_test"] = test_loss
            model.train()

    if diverged:
        last_train_loss = 5.0
        for step_target in spec.eval_steps:
            eval_losses[f"step_{step_target}_tune_val"] = 5.0
            eval_losses[f"step_{step_target}_test"] = 5.0

    summary_metrics: dict[str, Any] = {}
    if spec.log_ratios:
        summary_metrics["ratios"] = ratio_logs

    result = RunResult(
        run_id=spec.run_id,
        spec=spec,
        diverged=diverged,
        final_train_loss=last_train_loss,
        eval_losses=eval_losses,
        env_metadata=get_environment_metadata(),
        summary_metrics=summary_metrics,
    )

    if save_to_disk:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_json(out_dir / "final.json", result)

        # Write compressed metrics CSV
        if step_metrics:
            gz_path = out_dir / "metrics.csv.gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as f_gz:
                writer = csv.DictWriter(f_gz, fieldnames=["step", "train_loss", "lr", "grad_norm"])
                writer.writeheader()
                for row in step_metrics:
                    writer.writerow(row)

    return result
