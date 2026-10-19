# Verification Sources and Primary References

All items marked `[VERIFY]` in the research plan and codebase specifications are verified against primary sources and recorded here per rule R7.

| Item | Verified Value | Primary Source | Access Date | Notes |
|------|----------------|----------------|-------------|-------|
| Python version | 3.12.14 (CI: 3.11, 3.12) | Python Software Foundation releases / `.python-version` | 2026-10-19 | Supported active stable releases |
| PyTorch release | 2.14.1 (CPU wheel in uv.lock) | PyTorch official PyPI & download.pytorch.org index | 2026-10-19 | PyTorch 2.14.1 is pinned in lockfile |
| TinyShakespeare URL | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` | Andrej Karpathy's `char-rnn` GitHub repository | 2026-10-19 | HTTP 200 OK, fetched directly |
| TinyShakespeare SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` | Verified via `sha256sum data/input.txt` | 2026-10-19 | Length: 1,115,394 chars, 65 unique chars |
| TinyShakespeare Vocab Size | Exactly 65 unique characters | `len(set(open('data/input.txt').read()))` | 2026-10-19 | Sorted ASCII vocabulary from `\n` to `}` |
| PyTorch `Adam` single-tensor update implementation | `torch/optim/adam.py` lines 347–551 (`_single_tensor_adam`) | `torch/optim/adam.py` in PyTorch 2.14.1 source | 2026-10-19 | Epsilon placement: `denom = (exp_avg_sq.sqrt() / sqrt(1 - beta2^t)).add_(eps)` and `param.addcdiv_(exp_avg, denom, value=-step_size)` where `step_size = lr / (1 - beta1^t)` |
| `astral-sh/setup-uv` commit SHA | `cec208311dfd045dd5311c1add060b2062131d57` (v8.0.0) | `github.com/astral-sh/setup-uv` release | 2026-10-19 | Immutable tag release |
| `actions/checkout` commit SHA | `11bd71901bbe5b1630ceea73d27597364c9af683` (v4.2.2) | `git ls-remote https://github.com/actions/checkout` | 2026-10-19 | Official Actions repository |
| `actions/upload-artifact` commit SHA | `b4b15b8c7c6ac21ea08fcf65892d2ee8f75cf882` (v4.4.3) | `git ls-remote https://github.com/actions/upload-artifact` | 2026-10-19 | Official Actions repository |
| `actions/dependency-review-action` commit SHA | `5a2ce3f5b92ee19cbb1541a4984c76d921601d7c` (v4.3.4) | `git ls-remote https://github.com/actions/dependency-review-action` | 2026-10-19 | Official Actions repository |
| `github/codeql-action` commit SHA | `c1f0c97bcb634b400fc376ab260a5f187961f8e0` (v3.26.10) | `git ls-remote https://github.com/github/codeql-action` | 2026-10-19 | Official CodeQL Actions repository |
| Paper: Wen et al. (WSD River Valley) | Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, Tengyu Ma (arXiv 2410.05192, 2024) | arXiv API / HTML header citation metadata | 2026-10-19 | Title: "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective" |
| Paper: Wen et al. (Fantastic Optimizers) | Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang (arXiv 2509.02046, ICLR 2026) | arXiv API / HTML header citation metadata | 2026-10-19 | Title: "Fantastic Pretraining Optimizers and Where to Find Them" |
| Paper: Hägele et al. (WSD Scaling Laws) | Alexander Hägele et al., NeurIPS 2024 (arXiv 2405.18392) | NeurIPS 2024 Proceedings / arXiv | 2026-10-19 | Title: "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" |
| Paper: Yang et al. (Tensor Programs V / µP) | Greg Yang et al., 2022 (arXiv 2203.03466) | arXiv 2203.03466 | 2026-10-19 | Title: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" |
