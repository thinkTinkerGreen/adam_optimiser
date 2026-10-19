"""Data loading, SHA256 integrity verification, tokenization, splits, and batch generation."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Generator
from pathlib import Path

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
EXPECTED_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def compute_sha256(filepath: str | Path) -> str:
    """Compute the SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(data_dir: str | Path = "data") -> Path:
    """Download Tiny Shakespeare if needed and strictly verify its SHA256 checksum."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path / "input.txt"

    if not file_path.exists():
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, file_path)

    actual_hash = compute_sha256(file_path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"Dataset SHA256 mismatch! Expected {EXPECTED_SHA256}, got {actual_hash}")

    return file_path


def synthetic_corpus(n: int = 200_000, seed: int = 0) -> str:
    """Generate a seeded deterministic synthetic corpus for offline CI runs."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    # 65 unique ASCII characters to match vocabulary size exactly
    chars = [chr(i) for i in range(32, 97)]  # 65 chars
    indices = torch.randint(0, len(chars), (n,), generator=rng).tolist()
    return "".join(chars[i] for i in indices)


class TokenizedCorpus:
    """Tokenized character-level corpus with contiguous 80/10/10 train/tune_val/test splits."""

    def __init__(self, text: str) -> None:
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = dict(enumerate(self.vocab))

        # Encode full text as uint8 tensor
        indices = [self.char_to_idx[c] for c in text]
        data = torch.tensor(indices, dtype=torch.uint8)

        n = len(data)
        n_train = int(0.80 * n)
        n_val = int(0.10 * n)

        self.train_data = data[:n_train]
        self.tune_val_data = data[n_train : n_train + n_val]
        self.test_data = data[n_train + n_val :]

    @classmethod
    def load(cls, data_dir: str | Path = "data", use_synthetic: bool = False) -> TokenizedCorpus:
        if use_synthetic:
            return cls(synthetic_corpus())
        file_path = ensure_dataset(data_dir)
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return cls(text)

    def get_split(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.train_data
        elif split == "tune_val":
            return self.tune_val_data
        elif split == "test":
            return self.test_data
        else:
            raise ValueError(f"Unknown split: {split}. Choose from 'train', 'tune_val', 'test'.")


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random training batch using the provided generator."""
    max_idx = len(data) - context_length - 1
    # Generate 32 start offsets
    ix = torch.randint(0, max_idx + 1, (batch_size,), generator=generator)
    x_list = [data[i : i + context_length] for i in ix]
    y_list = [data[i + 1 : i + context_length + 1] for i in ix]

    x = torch.stack(x_list).to(device=device, dtype=torch.long)
    y = torch.stack(y_list).to(device=device, dtype=torch.long)
    return x, y


def iterate_eval_windows(
    data: torch.Tensor,
    context_length: int = 128,
    batch_size: int = 128,
    device: torch.device | str = "cpu",
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Iterate deterministically through all non-overlapping windows of length T in the split."""
    total_tokens = len(data)
    total_windows = (total_tokens - 1) // context_length

    for i in range(0, total_windows, batch_size):
        end_window = min(i + batch_size, total_windows)
        x_list = []
        y_list = []
        for w in range(i, end_window):
            start = w * context_length
            x_list.append(data[start : start + context_length])
            y_list.append(data[start + 1 : start + context_length + 1])
        x = torch.stack(x_list).to(device=device, dtype=torch.long)
        y = torch.stack(y_list).to(device=device, dtype=torch.long)
        yield x, y
