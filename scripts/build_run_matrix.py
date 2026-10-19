"""Generate deterministic ordered matrix of RunSpecs for all experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adamlab.io import RunSpec
from adamlab.sweep import get_base_lr_grid


def generate_run_matrix(
    widths: list[int] | None = None,
    tune_seeds: list[int] | None = None,
    report_seeds: list[int] | None = None,
    device: str = "cpu",
    use_synthetic: bool = False,
) -> list[RunSpec]:
    """Build complete deterministic ordered list of RunSpecs for Parts 3, 4, and 5."""
    if widths is None:
        widths = [256, 512, 1024]
    if tune_seeds is None:
        tune_seeds = [0, 1, 2]
    if report_seeds is None:
        report_seeds = [100, 101, 102]

    base_grid = get_base_lr_grid()
    specs: list[RunSpec] = []

    # 1. Part 5 Sweep (Widths 256, 512, 1024, Cosine 300 steps, tune seeds 0..2)
    # Also feeds Part 4 COS300 at width 512!
    for w in sorted(widths):
        for lr in sorted(base_grid):
            for s in sorted(tune_seeds):
                specs.append(
                    RunSpec(
                        experiment="p5_sweep",
                        width=w,
                        lr=lr,
                        schedule="cosine",
                        total_steps=300,
                        warmup_steps=30,
                        seed=s,
                        device=device,
                        use_synthetic=use_synthetic,
                        eval_steps=[200, 300],
                    )
                )

    # 2. Part 4 WSD300 Sweep (Width 512, WSD 300 steps, D=60, tune seeds 0..2)
    for lr in sorted(base_grid):
        for s in sorted(tune_seeds):
            specs.append(
                RunSpec(
                    experiment="p4_wsd300_tune",
                    width=512,
                    lr=lr,
                    schedule="wsd",
                    total_steps=300,
                    warmup_steps=30,
                    decay_steps=60,
                    seed=s,
                    device=device,
                    use_synthetic=use_synthetic,
                    eval_steps=[200, 300],
                )
            )

    # 3. Part 4 Matched Budget 200 Sweeps (Width 512, T=200, tune seeds 0..2)
    for lr in sorted(base_grid):
        for s in sorted(tune_seeds):
            # COS200
            specs.append(
                RunSpec(
                    experiment="p4_cos200_tune",
                    width=512,
                    lr=lr,
                    schedule="cosine",
                    total_steps=200,
                    warmup_steps=30,
                    seed=s,
                    device=device,
                    use_synthetic=use_synthetic,
                    eval_steps=[200],
                )
            )
            # WSD200
            specs.append(
                RunSpec(
                    experiment="p4_wsd200_tune",
                    width=512,
                    lr=lr,
                    schedule="wsd",
                    total_steps=200,
                    warmup_steps=30,
                    decay_steps=40,
                    seed=s,
                    device=device,
                    use_synthetic=use_synthetic,
                    eval_steps=[200],
                )
            )

    # Sort key: (experiment, width, schedule, lr, seed)
    specs.sort(key=lambda sp: (sp.experiment, sp.width, sp.schedule, sp.lr, sp.seed))
    return specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emit run matrix JSON.")
    parser.add_argument("--device", default="cpu", help="Compute device (e.g. cpu, cuda).")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic corpus.")
    parser.add_argument("--out", default="results/run_matrix.json", help="Output path.")
    args = parser.parse_args()

    matrix = generate_run_matrix(device=args.device, use_synthetic=args.synthetic)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in matrix], f, indent=2)
    print(f"Generated {len(matrix)} run specifications in {out_path}")
