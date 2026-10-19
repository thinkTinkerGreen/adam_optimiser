"""Tests for data checksum verification and synthetic corpus fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path

from adamlab.data import TokenizedCorpus, compute_sha256, synthetic_corpus


def test_synthetic_corpus_length_and_vocab() -> None:
    """Synthetic corpus creates exact length and 65 unique ASCII tokens."""
    text = synthetic_corpus(n=50000, seed=42)
    assert len(text) == 50000
    vocab = sorted(set(text))
    assert len(vocab) == 65


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    """compute_sha256 matches standard hashlib."""
    dummy_file = tmp_path / "test.txt"
    content = b"Deterministic optimizer benchmarking\n"
    dummy_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    actual = compute_sha256(dummy_file)
    assert actual == expected


def test_tokenized_corpus_splits() -> None:
    """Tokenized corpus correctly creates 80% train, 10% val, 10% test splits."""
    corpus = TokenizedCorpus.load(use_synthetic=True)
    total_len = len(corpus.train_data) + len(corpus.tune_val_data) + len(corpus.test_data)
    assert math_close(len(corpus.train_data) / total_len, 0.80)
    assert math_close(len(corpus.tune_val_data) / total_len, 0.10)
    assert math_close(len(corpus.test_data) / total_len, 0.10)


def math_close(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) < tol
