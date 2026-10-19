# Contributing Guidelines

We welcome contributions to `adam-fair-tuning`! This repository enforces strict empirical reproducibility, pre-registered decision rules, and CI/CD quality gates.

## Branching and Pull Request Protocol
1. **Branch Protection:** The `main` branch is protected. All modifications must be submitted via Pull Requests.
2. **Quality Gates:** Every PR must pass all CI checks:
   * Ruff linting & formatting (`make lint`)
   * Strict type checking (`make type`)
   * Unit and property tests with $\ge 85\%$ test coverage (`make test`)
   * End-to-end smoke test (`make smoke`)
   * Notebook hygiene check (`nbstripout`)
   * README synchronization check (`scripts/check_readme.py`)
3. **Commit Messages:** Follow Conventional Commits format:
   * `feat: ...`
   * `fix: ...`
   * `docs: ...`
   * `test: ...`
   * `refactor: ...`

## Parity Protocol
If introducing a new optimizer arm or comparison:
1. Register the comparison in `results/parity_ledger.csv`.
2. Ensure both arms tune the identical hyperparameter search grid with identical seeds and trial budgets.
3. Validate that `pytest tests/test_parity_ledger.py` passes.
