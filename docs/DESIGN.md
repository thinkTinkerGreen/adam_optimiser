# System Design and Architecture: adam-fair-tuning

## 1. Architectural Philosophy
The `adamlab` architecture implements an end-to-end reproducible research laboratory for optimizer dynamics.
Key tenets:
* **Separation of Specification from Execution:** Every run is described completely by an immutable `RunSpec` (Pydantic model) that is hashed deterministically (`sha256(canonical_json(RunSpec))[:16]`).
* **Parity Enforcement:** Any comparison between optimizer schedules or variants must register in `results/parity_ledger.csv` and satisfy strict equal-budget constraints enforced by unit tests.
* **Deterministic Oracle:** The reference Adam implementation (`adam_ref.py`) operates independently in arbitrary-precision arithmetic (`decimal.Decimal` with 40-digit precision) to serve as a ground-truth baseline without PyTorch dependencies.

## 2. Package Structure (`src/adamlab`)

```
src/adamlab/
├── __init__.py        # Package exports and version metadata
├── determinism.py     # Global seeding, deterministic backend switches, environment fingerprinting
├── data.py            # Corpus management, SHA-256 verification, tokenization, batch generators
├── model.py           # Decoder-only pre-LN Transformer (depth L=4, causal self-attention)
├── schedules.py       # Learning rate schedules: Linear warmup, Cosine decay, WSD (Warmup-Stable-Decay)
├── adam_ref.py        # Arbitrary precision decimal Adam oracle & PyTorch AdamRef with bias-correction toggle
├── train.py           # Single-run execution engine taking RunSpec -> RunResult
├── metrics.py         # Dynamic metrics: update-to-weight ratio per layer, loss, gradient norms
├── stats.py           # Quadratic vertex regression, bootstrap confidence intervals, paired SE
├── sweep.py           # Hyperparameter grid builder, dynamic edge-extension heuristics
├── ledger.py          # Parity ledger validator and recorder
└── io.py              # Atomic disk serialization, Pydantic schemas, run ID generation
```

## 3. Data Flow
1. **Experiment Definition:** Scripts (`experiments/p*.py`) define the experiment matrix using `sweep.py`.
2. **Execution:** `train.py` executes each run with dedicated CPU generators for initialization and data sampling.
3. **Storage:** Results write atomically via temporary files into `results/runs/<run_id>/final.json`.
4. **Aggregation:** Aggregators compute cross-seed statistics, bootstrap intervals, and parity ledgers.
5. **Visualization & Documentation:** `scripts/make_figures.py` generates vector and raster figures; `scripts/render_readme.py` renders `README.md` from `README.template.md` without hard-coded numbers.
