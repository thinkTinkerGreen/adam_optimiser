"""Deterministic environment configuration, seeding, and environment fingerprinting."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

# CUBLAS workspace config must be set before any CUDA context / torch initialization
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


def set_determinism() -> None:
    """Enforce deterministic algorithms and FP32 precision flags."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def get_git_sha() -> str:
    """Return the current Git commit SHA or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2.0
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_environment_metadata() -> dict[str, Any]:
    """Capture runtime environment metadata for reproducibility logs."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    cuda_version = torch.version.cuda if cuda_available else None
    return {
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": get_git_sha(),
    }


def make_generators(seed: int) -> tuple[torch.Generator, torch.Generator]:
    """Create independent CPU generators for model initialization and data sampling.

    Args:
        seed: Base seed for the run.

    Returns:
        tuple (init_gen, data_gen) where init_gen has seed `seed` and
        data_gen has seed `seed + 10_000`.
    """
    init_gen = torch.Generator(device="cpu").manual_seed(seed)
    data_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    return init_gen, data_gen
