"""Tests for README rendering and consistency auditing."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_readme import check_rendered_readme
from scripts.render_readme import render_readme


def test_readme_render_and_check() -> None:
    """Verify that rendering README produces no leftover tags and passes validation check."""
    readme_path = Path("README.md")
    template_path = Path("README.template.md")

    if not template_path.exists():
        pytest.skip("README.template.md not yet created.")

    render_readme(template_path=template_path, output_path=readme_path, results_dir="results")
    errors = check_rendered_readme(readme_path=readme_path, results_dir="results")
    assert not errors, f"README check reported errors: {errors}"
