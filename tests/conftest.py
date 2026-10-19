"""Pytest fixtures and configuration."""

from __future__ import annotations

import pytest

from adamlab.determinism import set_determinism


@pytest.fixture(autouse=True)
def enforce_determinism() -> None:
    """Enforce deterministic algorithm execution across all tests."""
    set_determinism()
