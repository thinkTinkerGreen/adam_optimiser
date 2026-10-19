"""Tests for update-to-weight ratio metric and tensor categorization."""

from __future__ import annotations

import math

import torch

from adamlab.metrics import categorize_parameter, compute_update_ratios


def test_update_ratio_toy_tensor() -> None:
    """Verify update ratio calculation on known vector norms."""
    w_before = {"toy": torch.tensor([3.0, 4.0], dtype=torch.float32)}  # norm = 5.0
    w_after = {
        "toy": torch.tensor([3.03, 4.04], dtype=torch.float32)
    }  # delta = [0.03, 0.04], norm = 0.05

    ratios = compute_update_ratios(w_before, w_after)
    assert "toy" in ratios
    assert math.isclose(ratios["toy"], 0.01, rel_tol=1e-5)


def test_update_ratio_zero_norm_guard() -> None:
    """Verify that zero norm handles gracefully with NaN."""
    w_before = {"zero": torch.tensor([0.0, 0.0], dtype=torch.float32)}
    w_after = {"zero": torch.tensor([0.1, 0.2], dtype=torch.float32)}

    ratios = compute_update_ratios(w_before, w_after)
    assert math.isnan(ratios["zero"])


def test_categorize_parameter() -> None:
    """Verify layer-family categorization."""
    assert categorize_parameter("tok_emb.weight") == "embed"
    assert categorize_parameter("pos_emb.weight") == "embed"
    assert categorize_parameter("blocks.0.ln1.weight") == "ln"
    assert categorize_parameter("blocks.0.attn.qkv.weight") == "qkv"
    assert categorize_parameter("blocks.1.attn.proj.bias") == "proj"
    assert categorize_parameter("blocks.2.mlp.fc1.weight") == "fc1"
    assert categorize_parameter("blocks.3.mlp.fc2.bias") == "fc2"
    assert categorize_parameter("head.weight") == "head"
