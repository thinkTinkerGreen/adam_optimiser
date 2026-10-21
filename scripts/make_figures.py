"""Generate or refresh all publication-grade figures from results."""

from __future__ import annotations

import os
from pathlib import Path

from experiments.p2_bias_correction import run_part2


def make_all_figures(
    results_dir: str | Path | None = None, figures_dir: str | Path = "figures"
) -> None:
    if results_dir is None:
        results_dir = os.environ.get("ADAMLAB_RESULTS_DIR", "results")
    res_path = Path(results_dir)
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    # 1. Part 2 figures
    p2_png = fig_path / "p2_first20.png"
    if not p2_png.exists():
        print("[Figures] Generating Part 2 figures...")
        run_part2(results_dir=res_path, figures_dir=fig_path)

    # Check existence of figures
    figures = [
        "p2_first20.png",
        "p2_first20.svg",
        "p3_ratio_by_family.png",
        "p3_ratio_heatmap.png",
        "p4_curves.png",
        "p4_matched.png",
        "p5_lr_sweep.png",
    ]
    for fig in figures:
        p = fig_path / fig
        if p.exists():
            print(f"  [OK] Figure: {p} ({p.stat().st_size} bytes)")
        else:
            print(f"  [PENDING] Figure: {p} (will be produced upon experiment completion)")


if __name__ == "__main__":
    make_all_figures()
