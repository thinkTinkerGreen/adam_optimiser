"""Metrics computation: update-to-weight ratios per layer, eval loss, and tensor categorization."""

from __future__ import annotations

from collections.abc import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F


def categorize_parameter(name: str) -> str:
    """Classify a parameter name into one of the canonical parameter families."""
    if "tok_emb" in name or "pos_emb" in name:
        return "embed"
    elif "ln" in name:
        return "ln"
    elif "attn.qkv" in name:
        return "qkv"
    elif "attn.proj" in name:
        return "proj"
    elif "mlp.fc1" in name:
        return "fc1"
    elif "mlp.fc2" in name:
        return "fc2"
    elif "head" in name:
        return "head"
    return "other"


def compute_update_ratios(
    weights_before: dict[str, torch.Tensor],
    weights_after: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute empirical update-to-weight ratio ||w(s) - w(s-1)||_2 / ||w(s-1)||_2 per tensor.

    Args:
        weights_before: snapshot of parameter tensors before optimizer step.
        weights_after: snapshot of parameter tensors after optimizer step.

    Returns:
        dict mapping parameter name to update ratio float.
    """
    ratios: dict[str, float] = {}
    for name, w_prev in weights_before.items():
        w_curr = weights_after[name]
        norm_prev = torch.linalg.vector_norm(w_prev, ord=2).item()
        delta = w_curr - w_prev
        norm_delta = torch.linalg.vector_norm(delta, ord=2).item()

        if norm_prev == 0.0:
            ratios[name] = float("nan")
        else:
            ratios[name] = float(norm_delta / norm_prev)
    return ratios


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    eval_batches: Generator[tuple[torch.Tensor, torch.Tensor], None, None]
    | list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Compute exact mean cross-entropy loss in nats over all non-overlapping windows."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in eval_batches:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()

    return float(total_loss / total_tokens) if total_tokens > 0 else float("nan")
