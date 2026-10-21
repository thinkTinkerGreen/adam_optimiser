"""Render README.md from README.template.md using measured results."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import jinja2

from adamlab.io import load_json


def load_results_context(results_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and structure all experiment outputs into template context."""
    if results_dir is None:
        results_dir = os.environ.get("ADAMLAB_RESULTS_DIR", "results")
    res_path = Path(results_dir)

    def read_json_or_none(filename: str) -> dict[str, Any] | None:
        # 1. Primary path (e.g. Drive if set)
        p = res_path / filename
        if p.exists():
            return load_json(p)
        # 2. Local fallback in repository
        fallback = Path("results") / filename
        if fallback.exists():
            return load_json(fallback)
        # 3. Dynamic run for fast mathematical verifications if missing
        if filename == "p1_adam_by_hand.json":
            try:
                from experiments.p1_adam_by_hand import run_part1

                return run_part1(results_dir=res_path)
            except Exception:
                pass
        elif filename == "p2_bias_correction.json":
            try:
                from experiments.p2_bias_correction import run_part2

                return run_part2(results_dir=res_path)
            except Exception:
                pass
        return None

    p1 = read_json_or_none("p1_adam_by_hand.json")
    p2 = read_json_or_none("p2_bias_correction.json")
    p3 = read_json_or_none("p3_update_ratio.json")
    p4 = read_json_or_none("p4_cosine_vs_wsd.json")
    p5 = read_json_or_none("p5_lr_sweep.json")
    s1 = read_json_or_none("s1_parity_demo.json")

    # Load parity ledger CSV
    ledger_entries = []
    ledger_file = res_path / "parity_ledger.csv"
    if not ledger_file.exists():
        ledger_file = Path("results") / "parity_ledger.csv"
    if ledger_file.exists():
        with open(ledger_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ledger_entries = list(reader)

    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "s1": s1,
        "ledger": ledger_entries,
    }


def render_readme(
    template_path: str | Path = "README.template.md",
    output_path: str | Path = "README.md",
    results_dir: str | Path | None = None,
) -> str:
    """Render the Jinja2 template and save the final README.md."""
    tmpl_path = Path(template_path)
    if not tmpl_path.exists():
        raise FileNotFoundError(f"Template not found at {tmpl_path}")

    with open(tmpl_path, encoding="utf-8") as f:
        template_text = f.read()

    context = load_results_context(results_dir)

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Custom filters
    env.filters["scinot"] = lambda val: f"{val:.2e}" if isinstance(val, (int, float)) else str(val)
    env.filters["dp4"] = lambda val: f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
    env.filters["dp6"] = lambda val: f"{val:.6f}" if isinstance(val, (int, float)) else str(val)

    template = env.from_string(template_text)
    rendered = template.render(**context)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"[Render README] Rendered {len(rendered)} bytes to {output_path}")
    return rendered


if __name__ == "__main__":
    render_readme()
