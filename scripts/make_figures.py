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

    # 2. Part 3 figures
    if not (fig_path / "p3_ratio_by_family.png").exists():
        try:
            from experiments.p3_update_ratio import run_part3

            print("[Figures] Generating Part 3 figures...")
            run_part3(results_dir=res_path, figures_dir=fig_path)
        except Exception as e:
            print(f"[Figures] Part 3 figures skipped: {e}")

    # 3. Part 4 figures
    if not (fig_path / "p4_curves.png").exists():
        try:
            from experiments.p4_cosine_vs_wsd import run_part4

            print("[Figures] Generating Part 4 figures...")
            run_part4(results_dir=res_path, figures_dir=fig_path)
        except Exception as e:
            print(f"[Figures] Part 4 figures skipped: {e}")

    # 4. Part 5 figures
    if not (fig_path / "p5_lr_sweep.png").exists():
        try:
            from experiments.p5_lr_sweep import run_part5

            print("[Figures] Generating Part 5 figures...")
            run_part5(results_dir=res_path, figures_dir=fig_path)
        except Exception as e:
            print(f"[Figures] Part 5 figures skipped: {e}")

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
