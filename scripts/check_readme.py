"""Validate integrity, formatting, and reproducibility of the rendered README.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from scripts.render_readme import load_results_context


def check_rendered_readme(
    readme_path: str | Path = "README.md", results_dir: str | Path | None = None
) -> list[str]:
    """Audit the rendered README.md for unrendered template tags and consistency."""
    if results_dir is None:
        import os

        results_dir = os.environ.get("ADAMLAB_RESULTS_DIR", "results")
    path = Path(readme_path)
    if not path.exists():
        return [f"README file does not exist: {path}"]

    content = path.read_text(encoding="utf-8")
    errors = []

    # 1. Check for unrendered Jinja tags
    unrendered_tags = re.findall(r"\{\{.*?\}\}", content)
    if unrendered_tags:
        errors.append(f"Found unrendered Jinja tags: {unrendered_tags[:5]}")

    unrendered_blocks = re.findall(r"\{%.*?%\}", content)
    if unrendered_blocks:
        errors.append(f"Found unrendered Jinja blocks: {unrendered_blocks[:5]}")

    # 2. Check for required section headers (Section 13)
    required_sections = [
        "Answers at a Glance",
        "Setup and Reproduction",
        "Part 1: Adam by Hand (1 Weight, 5 Gradients)",
        "Part 2: Bias Correction Off vs. On",
        "Part 3: Layer-wise Update-to-Weight Ratio and Warmup",
        "Part 4: Cosine vs. WSD (Literal & Matched-Budget)",
        "Part 5: Learning Rate Sweep across Widths and Width-4096 Recommendation",
        "Tuning Parity Protocol & Parity Ledger",
        "Deviations from Pre-registration",
        "Threats to Validity & Limitations",
        "References",
    ]

    for sec in required_sections:
        if sec not in content:
            errors.append(f"Missing required section header: '{sec}'")

    # 3. Check consistency of numbers from results
    ctx = load_results_context(results_dir)
    p1 = ctx.get("p1")
    if p1:
        fp64_err_str = f"{p1['overall_max_err_fp64']:.2e}"
        if fp64_err_str not in content:
            errors.append(f"Expected FP64 max error {fp64_err_str} not found in README content.")

    p2 = ctx.get("p2")
    if p2:
        t_star_5 = str(p2["t_star_table"]["0.999"]["t_star_5pct"])
        if t_star_5 not in content:
            errors.append(f"Expected t*(5%)={t_star_5} not found in README content.")

    return errors


if __name__ == "__main__":
    errs = check_rendered_readme()
    if errs:
        print("[Check README] FAILED with errors:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("[Check README] PASSED: README is completely rendered and consistent with results.")
