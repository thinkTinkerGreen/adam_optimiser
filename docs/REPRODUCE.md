# Reproduction Guide

This guide details how to reproduce all findings, figures, and tables reported in the project.

## Prerequisites
* Python 3.12 (or 3.11)
* `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
* Either a local workstation (CPU is sufficient for Parts 1 & 2, and smoke tests) or a GPU environment (Google Colab with NVIDIA T4) for the full Transformer sweeps (Parts 3–5).

---

## 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/thinkTinkerGreen/adam_optimiser.git
cd adam_optimiser

# Initialize virtual environment and install dependencies
make setup
```

---

## 2. Running Local CPU Experiments (Parts 1 and 2)

### Part 1: Adam by Hand (1 Weight, 5 Gradients)
Runs the arbitrary-precision 40-digit Decimal oracle against PyTorch FP64 and FP32 implementations:
```bash
make p1
```
Expected output:
* Verification of golden table values.
* Max absolute error $\le 10^{-12}$ for FP64 and $\le 2 \times 10^{-6}$ for FP32.
* Results serialized to `results/p1_adam_by_hand.json`.

### Part 2: Bias Correction Off vs. On
Simulates optimizer dynamics with and without bias correction over the first 20 steps and out to horizon $H = 20,000$ across $\beta_2 \in \{0.999, 0.99, 0.98, 0.95\}$:
```bash
make p2
```
Expected output:
* Computation of $t^*(\theta)$ for $\theta \in \{5\%, 1\%\}$.
* Results serialized to `results/p2_bias_correction.json`.
* Generation of `figures/p2_first20.png` and `figures/p2_first20.svg`.

---

## 3. Running Neural Network Training Experiments (Parts 3, 4, 5)

Training experiments can run locally on CPU for smoke testing or on Google Colab for GPU training:

```bash
# Smoke test (runs on CPU in < 60s)
make smoke

# Full experiments:
make p3      # Part 3: Layer-wise update-to-weight ratio and warmup analysis
make p4      # Part 4: Cosine vs. WSD comparison at step 200 and matched budget
make p5      # Part 5: Learning rate sweeps at widths 256, 512, 1024 and width-4096 extrapolation
```

### Running on Google Colab
1. Open `notebooks/colab_runner.ipynb` in Google Colab.
2. Select a GPU runtime (e.g. Free Tier T4).
3. Mount Google Drive to persist intermediate checkpoint logs and results.
4. Execute cells sequentially. Each completed run writes atomically to Drive, making interrupted sessions immediately resumable.

---

## 4. Generating Figures and Rendering the README

```bash
# Generate all figures
make figures

# Render README from template and results files
make readme

# Run complete verification check
make check
```
