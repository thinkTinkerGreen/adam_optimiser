"""Serialization, Pydantic schemas, deterministic run ID hashing, and atomic file I/O."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunSpec(BaseModel):
    """Specification defining an isolated, deterministic training run."""

    experiment: str
    width: int
    depth: int = 4
    lr: float
    schedule: str  # 'cosine', 'wsd', 'constant'
    total_steps: int = 300
    warmup_steps: int = 30
    decay_steps: int | None = None
    f_min: float = 0.1
    seed: int = 0
    batch_size: int = 32
    context_length: int = 128
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "cpu"
    use_synthetic: bool = False
    log_ratios: bool = False
    eval_steps: list[int] = Field(default_factory=lambda: [200, 300])

    def canonical_json(self) -> str:
        """Deterministic canonical JSON serialization for hashing."""
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))

    @property
    def run_id(self) -> str:
        """16-character SHA-256 hash of the canonical specification."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]


class RunResult(BaseModel):
    """Encapsulated outputs and metadata of a completed training run."""

    run_id: str
    spec: RunSpec
    diverged: bool = False
    final_train_loss: float
    eval_losses: dict[str, float]  # e.g., {'step_200_tune_val': ..., 'step_300_tune_val': ...}
    env_metadata: dict[str, Any] = Field(default_factory=dict)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)


def atomic_save_json(filepath: str | Path, data: dict[str, Any] | BaseModel) -> None:
    """Atomically write JSON data using a temporary file in the target directory."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = data.model_dump() if isinstance(data, BaseModel) else data

    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        temp_name = f.name

    os.replace(temp_name, path)


def load_json(filepath: str | Path) -> dict[str, Any]:
    """Read JSON file from disk."""
    with open(filepath, encoding="utf-8") as f:
        res: dict[str, Any] = json.load(f)
        return res


def build_manifest(results_dir: str | Path = "results") -> dict[str, str]:
    """Compute sha256 checksums of all committed result files in results directory."""
    dir_path = Path(results_dir)
    manifest: dict[str, str] = {}

    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json" and not path.name.endswith(".csv.gz"):
            rel_path = str(path.relative_to(dir_path))
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            manifest[rel_path] = h.hexdigest()

    manifest_file = dir_path / "MANIFEST.json"
    atomic_save_json(manifest_file, manifest)
    return manifest
