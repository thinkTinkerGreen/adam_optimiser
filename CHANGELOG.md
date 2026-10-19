# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-19

### Added
- **Part 1:** Arbitrary precision 40-digit Decimal Adam oracle and PyTorch FP64/FP32 comparison table.
- **Part 2:** Bias-correction comparison across first 20 steps and long-horizon simulations ($H=20,000$) with empirical $t^*$ computation.
- **Part 3:** Layer-wise update-to-weight ratio tracking across depth $L=4$ Transformer layers and empirical warmup termination step ($t_w$) detection.
- **Part 4:** Fair comparison between Cosine decay and Warmup-Stable-Decay (WSD) schedules at step 200 and matched budget ($T=200$) under tuning parity.
- **Part 5:** Learning rate sweep across widths 256, 512, 1024, quadratic vertex estimation, bootstrap confidence intervals, and width-4096 extrapolation.
- **Parity Protocol:** Parity ledger (`results/parity_ledger.csv`) and automated test enforcement.
- **CI/CD:** Multi-stage GitHub Actions workflows for linting, typing, unit testing, coverage, smoke testing, CodeQL, and automated release.
