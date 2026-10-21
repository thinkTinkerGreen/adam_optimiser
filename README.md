# Adam by Hand → Fair Tuning

[![CI](https://github.com/thinkTinkerGreen/adam_optimiser/actions/workflows/ci.yml/badge.svg)](https://github.com/thinkTinkerGreen/adam_optimiser/actions/workflows/ci.yml)
[![CodeQL](https://github.com/thinkTinkerGreen/adam_optimiser/actions/workflows/codeql.yml/badge.svg)](https://github.com/thinkTinkerGreen/adam_optimiser/actions/workflows/codeql.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical research laboratory investigating Adam optimizer dynamics from first principles: arbitrary-precision hand reproduction, bias-correction horizon dynamics, layer-wise update-to-weight ratios, cosine versus warmup-stable-decay (WSD) schedules under strict tuning parity, learning rate scaling laws across model widths (256, 512, 1024), and principled width-4096 extrapolation. Every single numerical value reported below is deterministically rendered directly from validated test runs and serialized metrics logs.

---

## Answers at a Glance

| Question | Answer | Primary Reference / Evidence |
| :--- | :--- | :--- |
| **1. Adam by hand vs. PyTorch agreement** | **FP64 max error:** `7.77e-16`<br>**FP32 max error:** `3.83e-08` | [Part 1: Adam by Hand](#part-1-adam-by-hand-1-weight-5-gradients) |
| **2. Bias correction difference threshold ($t^*$, $\beta_2=0.999$)** | **5% threshold:** `2375` steps<br>**1% threshold:** `3925` steps<br>*(Does **not** vanish within 20 steps; peaks at step 12 at 6.57×)* | [Part 2: Bias Correction Off vs. On](#part-2-bias-correction-off-vs-on) |
| **3. Step where warmup stops changing update ratio ($t_w$)** | **Global $t_w$ (5% threshold):** Step `300`<br>**Median $t_w$:** Step `300` | [Part 3: Layer-wise Update Ratio](#part-3-layer-wise-update-to-weight-ratio-and-warmup) |
| **4. Cosine vs. WSD at step 200 & KEEP decision** | **Literal Stop (Step 200 of 300):** Cosine = `2.0819` vs. WSD = `2.1038` nats<br>**Matched Budget ($T=200$):** `WSD`<br>**Recommendation:** Keep **`WSD`** | [Part 4: Cosine vs. WSD](#part-4-cosine-vs-wsd-literal--matched-budget) |
| **5. The three LR minima (Widths 256 / 512 / 1024)** | **Width 256:** `3.15e-03`<br>**Width 512:** `1.29e-03`<br>**Width 1024:** `4.92e-04` | [Part 5: LR Sweeps](#part-5-learning-rate-sweep-across-widths-and-width-4096-recommendation) |
| **6. Learning rate at width 4096 + confidence** | **Predicted LR:** `7.77e-05` (1/width constrained: `1.23e-04`)<br>**Confidence:** `moderate` (slope $b = -1.34$) | [Part 5: Extrapolation](#part-5-learning-rate-sweep-across-widths-and-width-4096-recommendation) |
| **7. Tuning parity statement** | Both arms tuned across identical 17-point half-octave grids with equal trial budgets (51 trials/arm); pre-registered on `tune_val` and evaluated on `test`. | [Parity Ledger](#tuning-parity-protocol--parity-ledger) |

---

## Setup and Reproduction

### Environment & Toolchain
- **Language:** Python 3.12 (CI test matrix: 3.11, 3.12).
- **Package Manager:** `uv` with fully resolved and frozen lockfile `uv.lock`.
- **PyTorch:** Pinned release 2.14.1 (CPU wheel in lockfile; Colab runtime utilizes environment build).
- **Precision:** Strict Single-Precision Floating Point (`float32`). TensorFloat32 (`allow_tf32=False`) and CuDNN heuristics (`benchmark=False`) are disabled.
- **Attention Backend:** Enforced deterministic `torch.nn.attention.SDPBackend.MATH`.

### Quickstart Commands
```bash
# Clone the repository
git clone https://github.com/thinkTinkerGreen/adam_optimiser.git
cd adam_optimiser

# Initialize environment and install dependencies
make setup

# Run linting, strict type checking, and unit test suite
make check

# Run end-to-end smoke test (<60s on CPU)
make smoke
```

---

## Part 1: Adam by Hand (1 Weight, 5 Gradients)

### 1.1 Specification and Initial Values
- Weight: $w_0 = 1.0$
- Learning rate: $\eta = 0.01$
- Hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, `weight_decay = 0`, `amsgrad = False`
- Gradient sequence: $g_1 = +0.1, g_2 = -0.2, g_3 = +0.3, g_4 = -0.4, g_5 = +0.5$

### 1.2 Mathematical Derivations & Analytical Updates
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\text{step}_t = \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}, \quad w_t = w_{t-1} - \text{step}_t
$$

### 1.3 Step-by-Step Numerical Comparison Table
The reference oracle was computed in 40-digit arbitrary precision decimal arithmetic using Python's `decimal.Decimal` module:

| $t$ | $g_t$ | Metric | Oracle (Decimal 40-digit) | PyTorch FP64 | PyTorch FP32 | Abs Error (FP64) | Abs Error (FP32) |
|---|---|---|---|---|---|---|---|
| **1** | `0.1` | $m$ | `0.010000000000` | `0.010000000000` | `0.010000000708` | `1.73e-18` | `7.08e-10` |
| | | $v$ | `0.000010000000` | `0.000010000000` | `0.000010000001` | `1.02e-20` | `6.57e-13` |
| | | $\hat{m}$ | `0.100000000000` | `0.100000000000` | `0.100000007078` | `0.00e+00` | `7.08e-09` |
| | | $\hat{v}$ | `0.010000000000` | `0.010000000000` | `0.010000000657` | `1.73e-18` | `6.57e-10` |
| | | $\text{step}$ | `0.009999999000` | `0.009999999000` | `0.009999990463` | `4.86e-17` | `8.54e-09` |
| | | $w$ | `0.990000001000` | `0.990000001000` | `0.990000009537` | `0.00e+00` | `8.54e-09` |
| **2** | `-0.2` | $m$ | `-0.011000000000` | `-0.011000000000` | `-0.011000000872` | `0.00e+00` | `8.72e-10` |
| | | $v$ | `0.000049990000` | `0.000049990000` | `0.000049990002` | `5.42e-20` | `1.57e-12` |
| | | $\hat{m}$ | `-0.057894736842` | `-0.057894736842` | `-0.057894741430` | `1.39e-17` | `4.59e-09` |
| | | $\hat{v}$ | `0.025007503752` | `0.025007503752` | `0.025007504538` | `3.68e-16` | `7.86e-10` |
| | | $\text{step}$ | `-0.003661035039` | `-0.003661035039` | `-0.003661036491` | `5.16e-17` | `1.45e-09` |
| | | $w$ | `0.993661036039` | `0.993661036039` | `0.993661046028` | `1.11e-16` | `9.99e-09` |
| **3** | `0.3` | $m$ | `0.020100000000` | `0.020100000000` | `0.020100001246` | `6.94e-18` | `1.25e-09` |
| | | $v$ | `0.000139940010` | `0.000139940010` | `0.000139940021` | `1.08e-19` | `1.07e-11` |
| | | $\hat{m}$ | `0.074169741697` | `0.074169741697` | `0.074169746294` | `0.00e+00` | `4.60e-09` |
| | | $\hat{v}$ | `0.046693347783` | `0.046693347783` | `0.046693351368` | `4.86e-17` | `3.59e-09` |
| | | $\text{step}$ | `0.003432410644` | `0.003432410644` | `0.003432393074` | `2.34e-17` | `1.76e-08` |
| | | $w$ | `0.990228625395` | `0.990228625395` | `0.990228652954` | `1.11e-16` | `2.76e-08` |
| **4** | `-0.4` | $m$ | `-0.021910000000` | `-0.021910000000` | `-0.021910000592` | `3.47e-18` | `5.92e-10` |
| | | $v$ | `0.000299800070` | `0.000299800070` | `0.000299800100` | `2.71e-19` | `3.00e-11` |
| | | $\hat{m}$ | `-0.063710380925` | `-0.063710380925` | `-0.063710382647` | `1.39e-17` | `1.72e-09` |
| | | $\hat{v}$ | `0.075062536258` | `0.075062536258` | `0.075062543781` | `7.77e-16` | `7.52e-09` |
| | | $\text{step}$ | `-0.002325404822` | `-0.002325404822` | `-0.002325415611` | `2.30e-17` | `1.08e-08` |
| | | $w$ | `0.992554030216` | `0.992554030216` | `0.992554068565` | `1.11e-16` | `3.83e-08` |
| **5** | `0.5` | $m$ | `0.030281000000` | `0.030281000000` | `0.030281001702` | `1.39e-17` | `1.70e-09` |
| | | $v$ | `0.000549500270` | `0.000549500270` | `0.000549500342` | `4.34e-19` | `7.25e-11` |
| | | $\hat{m}$ | `0.073944470221` | `0.073944470221` | `0.073944474377` | `1.39e-17` | `4.16e-09` |
| | | $\hat{v}$ | `0.110120074002` | `0.110120074002` | `0.110120088539` | `3.61e-16` | `1.45e-08` |
| | | $\text{step}$ | `0.002228293746` | `0.002228293746` | `0.002228319645` | `4.34e-17` | `2.59e-08` |
| | | $w$ | `0.990325736470` | `0.990325736470` | `0.990325748920` | `1.11e-16` | `1.25e-08` |

> [!NOTE]
> **Verified PyTorch Implementation:** Inspected in `torch/optim/adam.py` (lines 525–545). PyTorch does not store $\hat{m}$ and $\hat{v}$ explicitly in memory buffers; rather, it analytically folds the bias corrections into scalar step sizes:
> ```python
> step_size = lr / bias_correction1
> bias_correction2_sqrt = bias_correction2**0.5
> denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
> param.addcdiv_(exp_avg, denom, value=-step_size)
> ```
> This form is mathematically identical to the canonical textbook formulation. Maximum absolute error between the Decimal oracle and PyTorch FP64 is **`7.77e-16`** (passing the $\le 10^{-12}$ gate), and FP32 is **`3.83e-08`** (passing the $\le 2 \times 10^{-6}$ gate).

---

## Part 2: Bias Correction Off vs. On

### 2.1 First 20 Steps Simulation
When bias correction is disabled, the update rule becomes:

$$
\text{step}_t = \frac{\eta \cdot m_t}{\sqrt{v_t} + \varepsilon}
$$

Under constant gradient $g$, the uncorrected to corrected step magnitude ratio follows:

$$
r_t = \frac{|\Delta w_{\text{noBC}}|}{|\Delta w_{\text{BC}}|} = \frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}}
$$

![Part 2 First 20 Steps](figures/p2_first20.png)

### 2.2 Analytical & Empirical $t^*$ Horizon Table
Threshold $t^*(\theta)$ defines the smallest step such that for **all** $t \ge t^*$ through horizon $H = 20{,}000$:

$$
\left| \frac{|\Delta w_{\text{noBC}}|}{|\Delta w_{\text{BC}}|} - 1 \right| < \theta
$$

| $\beta_2$ Value | Closed Form $t^*(5\%)$ | Simulation $t^*(5\%)$ | Closed Form $t^*(1\%)$ | Simulation $t^*(1\%)$ | Peak Ratio ($t=12$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.999** (Default) | `2375` | `2375` | `3925` | `3925` | 6.57× |
| **0.99** | `237` | `237` | `391` | `391` | 2.11× |
| **0.98** | `118` | `118` | `195` | `195` | 1.50× |
| **0.95** (LLM Default) | `43` | `43` | `76` | `76` | 1.13× |

> [!IMPORTANT]
> **Plain Language Finding:** For the default Adam setting ($\beta_2 = 0.999$), **the difference between enabling and disabling bias correction never stops mattering inside the first 20 steps**. The step ratio begins at 3.16×, climbs to a peak of **6.57× at step 12**, and remains at **6.24× at step 20**. It requires **2,375 steps** to fall within 5% of parity. In contrast, under $\beta_2 = 0.95$ (widely used in modern LLM pre-training), $t^*(5\%)$ drops to only **43 steps**, aligning closely with nominal warmup durations.

---

## Part 3: Layer-wise Update-to-Weight Ratio and Warmup

### 3.1 Empirical Metric Definition
For every parameter tensor $i$ at step $s$:

$$
r_i(s) = \frac{\|w_i(s) - w_i(s-1)\|_2}{\|w_i(s-1)\|_2}
$$

Warmup termination step $t_{w,i}(5\%)$ is the first step where moving-average discrepancy satisfies:

$$
d_i(s) = \frac{|r_i^{\text{W30}}(s) - r_i^{\text{W0}}(s)|}{r_i^{\text{W0}}(s)} < 5\%
$$

for all remaining steps.

![Part 3 Family Ratios](figures/p3_ratio_by_family.png)
![Part 3 Heatmap](figures/p3_ratio_heatmap.png)

### 3.2 Warmup Termination Step ($t_w$) Summary
- **Global $t_w$ (all layers $\le 5\%$ discrepancy):** Step **300** (Nominal warmup: 30 steps).
- **Median $t_w$ across parameter tensors:** Step **300**.
- **First-step scaling check:** In W0, element update scales as $\eta$, matching predicted ratio.
- **Linear warmup correlation:** Mean Pearson correlation between $r_i(s)$ and $s/30$ for $s \le 30$ is **0.9242**.

---

## Part 4: Cosine vs. WSD (Literal & Matched-Budget)

### 4.1 Tuning and Test Results
- **Comparison Split:** Tuned on `tune_val` (seeds {0, 1, 2}), reported on held-out `test` (seeds {100, 101, 102}).
- **Trial Parity:** Exactly 51 trials per arm (17 grid points × 3 tuning seeds).

| Comparison | Arm | Tuned Peak LR | Test Loss Mean (nats) | Test Loss SE | Paired $\Delta$ (nats) | Paired SE | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Literal (Step 200 of 300)** | COS300 | `1.95e-03` | `2.0819` | `0.0131` | `-0.0219` | `0.0093` | **COSINE** |
| | WSD300 | `1.95e-03` | `2.1038` | `0.0128` | | | |
| **Matched Budget ($T=200$)** | COS200 | `1.95e-03` | `2.0862` | `0.0062` | `0.0318` | `0.0091` | **WSD** |
| | WSD200 | `1.95e-03` | `2.0543` | `0.0107` | | | |

![Part 4 Trajectories](figures/p4_curves.png)
![Part 4 Matched Budget](figures/p4_matched.png)

### 4.2 KEEP Decision and Rationale
- **Final Decision:** KEEP **`WSD`**.
- **Scientific Rationale:** WSD demonstrated decisive loss advantage at matched budget.

---

## Part 5: Learning Rate Sweep across Widths and Width-4096 Recommendation

### 5.1 Sweeps across Widths 256, 512, 1024
Evaluated over base half-octave grid $lr = 2^k$ for $k \in \{-14, -13.5, \ldots, -6\}$:

| Width ($w$) | Grid Argmin LR | Fitted Vertex $x^*$ ($\log_2$) | Fitted Optimal LR ($lr^*$) | 95% Bootstrap CI | Basin Half-Width ($h_w$) | Held-out Test Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** | `3.91e-03` | `-8.31` | `3.15e-03` | [`1.95e-03`, `7.81e-03`] | `0.50` oct | `1.9926` ± `0.0028` |
| **512** | `1.38e-03` | `-9.60` | `1.29e-03` | [`6.91e-04`, `2.76e-03`] | `0.25` oct | `1.9582` ± `0.0071` |
| **1024** | `4.88e-04` | `-10.99` | `4.92e-04` | [`2.44e-04`, `9.77e-04`] | `0.50` oct | `2.0003` ± `0.0034` |

![Part 5 Sweep and Extrapolation](figures/p5_lr_sweep.png)

### 5.2 Recommendation for Width 4096
- **Data-Driven Free Fit LR:** **`7.77e-05`** ($\log_2(lr^*) = -13.65$).
- **Constrained $1/\text{width}$ LR ($lr_{1024} / 4$):** **`1.23e-04`**.
- **Empirical Scaling Slope ($b$):** **`-1.34`** (95% CI: [`-1.46`, `-1.30`]).
- **Confidence Rating:** **`moderate`** (95% prediction interval width $W_{95} = 0.40$ octaves compared to width-1024 basin half-width $h_{1024} = 0.50$ octaves).
- **Practical Trial Range for Width 4096:** [`3.89e-05`, `1.55e-04`].

---

## Tuning Parity Protocol & Parity Ledger

Every comparison adheres strictly to the four parity rules:
1. Identical tuned hyperparameters (peak learning rate).
2. Identical search grids and trial budgets.
3. Separation of tuning split (`tune_val`) from reporting split (`test`).
4. Identical evaluation steps and token budgets.

### Parity Ledger Entries
| Comparison ID | Arm | Tuned Knobs | Grid ID | Trials | Tuning Split | Report Split | Eval Step | Tuned Value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P4_LITERAL_200` | **COS300** | `peak_lr` | `grid_half_octave_17` | 51 | `tune_val` | `test` | 200 | `0.001953125` |
| `P4_LITERAL_200` | **WSD300** | `peak_lr` | `grid_half_octave_17` | 51 | `tune_val` | `test` | 200 | `0.001953125` |
| `P4_MATCHED_200` | **COS200** | `peak_lr` | `grid_half_octave_17` | 51 | `tune_val` | `test` | 200 | `0.001953125` |
| `P4_MATCHED_200` | **WSD200** | `peak_lr` | `grid_half_octave_17` | 51 | `tune_val` | `test` | 200 | `0.001953125` |


---

## Deviations from Pre-registration
None. All hyperparameter grids, decision criteria ($D_1, D_2$), statistical thresholds ($\theta = 5\%$), and confidence rules were committed to `docs/PREREGISTRATION.md` (tagged `prereg`) prior to empirical training sweeps.

---

## Threats to Validity & Limitations
1. **Model Depth and Scale:** Experiments fix depth $L = 4$ across widths $256, 512, 1024$. Optimal scaling exponents may shift in deeper architectures ($L \ge 24$).
2. **Corpus Scope:** The dataset is character-level Tiny Shakespeare (~1.1M characters). While optimal for controlled isolation, subword-tokenized corpora at web scale may exhibit different pre-conditioning dynamics.
3. **Parametrization:** The Transformer utilizes standard parametrization (SP), not maximal update parametrization ($\mu\text{P}$). In SP, Adam learning rate is expected to decay with width, whereas $\mu\text{P}$ stabilizes optimal LR across width.
4. **Short Horizons:** 300 steps is an accelerated regime. Although warmup and stable phases are isolated cleanly, multi-billion token training runs may introduce secondary stationary phases.
5. **Hardware Determinism:** Determinism guarantees are verified on identical hardware and PyTorch builds. Cross-architecture bitwise reproducibility (e.g. CPU vs NVIDIA T4) is subject to driver and floating-point implementation differences.

---

## References
1. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
2. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
3. Hu, S., et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*.
4. Hägele, A., et al. (2024). *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*. NeurIPS 2024. [arXiv:2405.18392](https://arxiv.org/abs/2405.18392).
5. Wen, K., Li, Z., Wang, J., Hall, D., Liang, P., & Ma, T. (2024). *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective*. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192).
6. Yang, G., et al. (2022). *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466).
7. Wen, K., Hall, D., Ma, T., & Liang, P. (2026). *Fantastic Pretraining Optimizers and Where to Find Them*. ICLR 2026. [arXiv:2509.02046](https://arxiv.org/abs/2509.02046).
8. PyTorch Documentation. (2026). *torch.optim.Adam*. Verified against PyTorch 2.14.1 single-tensor implementation lines 347–551.

---

## Repository Map

```
adam_optimiser/
├── README.template.md              # Jinja2 template for deterministic documentation
├── README.md                       # Automatically rendered from results/
├── LICENSE                         # MIT License
├── CITATION.cff                    # Citation metadata
├── CONTRIBUTING.md                 # Contribution protocol and quality gates
├── SECURITY.md                     # Security policy and vulnerability reporting
├── CHANGELOG.md                    # Keep-a-Changelog version history
├── pyproject.toml                  # Hatchling build specification and tool configs
├── uv.lock                         # Pinned dependency lockfile
├── .python-version                 # Pinned Python version (3.12)
├── Makefile                        # Reproducibility targets
├── .pre-commit-config.yaml         # Pre-commit formatting and hygiene hooks
├── src/adamlab/                    # Core library
│   ├── determinism.py              # Strict determinism, CPU generators, env capture
│   ├── data.py                     # Tiny Shakespeare download, SHA256 integrity, batching
│   ├── model.py                    # Pre-LN decoder-only Transformer LM
│   ├── schedules.py                # Linear warmup, Cosine decay, WSD scheduler
│   ├── adam_ref.py                 # Decimal 40-digit oracle and AdamRef optimizer
│   ├── train.py                    # Deterministic training engine
│   ├── metrics.py                  # Update-to-weight ratios, eval loss
│   ├── stats.py                    # Quadratic vertex fit, bootstrap, scaling laws
│   ├── sweep.py                    # Hyperparameter grid builder and edge heuristics
│   ├── ledger.py                   # Parity ledger validator
│   └── io.py                       # Atomic serialization, run ID hashing, manifest
├── experiments/                    # Experiment entrypoints (Parts 1–5, Stretch S1)
├── scripts/                        # Automation: build matrix, make figures, render/check README
├── notebooks/adam_by_hand_tuning.ipynb # Google Colab T4 GPU execution notebook
├── tests/                          # 40+ unit and property tests (≥90% coverage)
├── results/                        # Serialized JSON outputs and parity ledger
├── figures/                        # Generated vector (SVG) and raster (PNG) figures
└── docs/                           # Pre-registration, ADRs, design, sources
```

---

## Citation

```bibtex
@software{thinktinkergreen2026adam,
  author = {thinktinkergreen},
  title = {adam-fair-tuning: Empirical Reproducibility and Fair Optimizer Benchmarking from Adam by Hand to Scaling Laws},
  year = {2026},
  url = {https://github.com/thinkTinkerGreen/adam_optimiser}
}
```