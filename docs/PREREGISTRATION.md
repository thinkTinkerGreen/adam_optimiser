# Pre-Registration: Adam by Hand → Fair Tuning

**Pre-registration Date:** 2026-10-19  
**Status:** Pre-registered before GPU/empirical training runs.  
**Git Tag:** `prereg`

---

## 1. Scope and Objective
This document fixes all experimental protocols, evaluation splits, hyperparameter grids, statistical estimators, stopping rules, and decision criteria prior to running training experiments for Parts 3, 4, and 5 of the study.

---

## 2. Dataset Splits and Evaluation
* **Corpus:** Tiny Shakespeare character-level corpus.
* **Integrity:** SHA-256 verified against `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
* **Splits:** Contiguous non-overlapping slice partition:
  * **Train:** First 80% of tokens (positions `0` to `0.80 * N - 1`).
  * **Tune Validation (`tune_val`):** Next 10% of tokens (positions `0.80 * N` to `0.90 * N - 1`).
  * **Held-out Test (`test`):** Final 10% of tokens (positions `0.90 * N` to `N - 1`).
* **Evaluation Protocol:** Deterministic full evaluation over all non-overlapping windows of length $T = 128$ in the respective split, batch size 128, mean cross-entropy across all tokens in nats. No stochastic sampling in evaluation.

---

## 3. Seeds and RNG Management
* **Tuning Seeds:** $\{0, 1, 2\}$ — used exclusively for hyperparameter selection on `tune_val`.
* **Reporting Seeds:** $\{100, 101, 102\}$ — used exclusively for final confirmation and evaluation on `test`.
* **RNG Generators:**
  * `init_gen = torch.Generator(device="cpu").manual_seed(seed)`: parameter initialization.
  * `data_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)`: batch window selection.
* **Paired Design:** For a given seed, the sequence of batch offsets is identical across all optimizer schedules and learning rates.

---

## 4. Hyperparameter Search Grid and Edge Extension Rule
* **Base LR Grid:** Log-spaced grid $lr = 2^k$ for $k \in \{-14, -13.5, -13, \ldots, -6\}$ (17 points spanning $6.10 \times 10^{-5}$ to $1.56 \times 10^{-2}$).
* **Selection Metric:** Mean `tune_val` loss across tuning seeds $\{0, 1, 2\}$ at the designated evaluation step.
* **Edge Extension Rule:** If the grid argmin for any model width lies at the lowest or highest point of the grid:
  1. Extend the grid by 2 octaves (4 additional points at step size $0.5$) in the direction of the edge.
  2. Evaluate all tuning seeds on the new points.
  3. Repeat recursively until the minimum is strictly interior.
  4. Log all extensions to `results/sweep_extensions.json`.

---

## 5. Part 2: Bias Correction Threshold ($t^*$)
* **Definition:** $t^*(\theta)$ is the smallest step index $t^*$ such that for all $t \ge t^*$ up to horizon $H = 20,000$:
  $$ \left| \frac{|\Delta w_{\text{noBC}}|}{|\Delta w_{\text{BC}}|} - 1 \right| < \theta $$
* **Thresholds:** $\theta \in \{5\%, 1\%\}$.
* **Evaluated $\beta_2$ Values:** $\{0.999, 0.99, 0.98, 0.95\}$ with $\beta_1 = 0.9$.
* **Primary Reference:** Closed-form ratio $r_t = \frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}}$ verified against simulation over 10,000 weights under constant and noisy gradient streams.

---

## 6. Part 3: Update-to-Weight Ratio and Warmup Stop ($t_w$)
* **Layer-wise Metric:**
  $$ r_i(s) = \frac{\|w_i(s) - w_i(s-1)\|_2}{\|w_i(s-1)\|_2} $$
  measured empirically after optimizer step $s$ across every named parameter tensor.
* **Averaging & Smoothing:** 3-seed mean smoothed with centered moving average of window 11 (edge-truncated).
* **Relative Discrepancy:**
  $$ d_i(s) = \frac{|r_i^{\text{W30}}(s) - r_i^{\text{W0}}(s)|}{r_i^{\text{W0}}(s)} $$
* **Tensor Warmup Termination:** Smallest $s$ such that $d_i(s') < \theta$ for all $s' \ge s$ through step 300.
* **Primary Threshold:** $\theta = 5\%$. Secondary sensitivity checks at $\theta \in \{2\%, 10\%\}$.
* **Global $t_w$:** $\max_i t_{w,i}(5\%)$, representing the step after which warmup has ceased to affect the update ratio of every layer.

---

## 7. Part 4: Cosine vs. WSD Decision Rules
* **Arms:**
  * **COS300:** Cosine schedule, $T=300, W=30$, evaluated at step 200.
  * **WSD300:** WSD schedule, $T=300, W=30, D=60$, evaluated at step 200.
  * **COS200:** Cosine schedule, $T=200, W=30$, evaluated at step 200.
  * **WSD200:** WSD schedule, $T=200, W=30, D=40$, evaluated at step 200.
* **Tuning Parity:** Peak LR tuned independently for each arm on `tune_val` (seeds $\{0, 1, 2\}$) over the same half-octave grid.
* **Paired Test Metric:** On held-out `test` split across report seeds $\{100, 101, 102\}$:
  $$ \Delta = \text{mean}(L_{\text{cosine}} - L_{\text{WSD}}), \quad \text{SE} = \frac{\text{std}(L_{\text{cosine}} - L_{\text{WSD}})}{\sqrt{N}} $$
* **Decisiveness Rule:** A comparison is decisive if and only if:
  $$ |\Delta| > 2 \cdot \text{SE} \quad \text{and} \quad |\Delta| > 0.01 \text{ nats} $$
  Otherwise, it is declared a **tie**.
* **Decisions:**
  * **D1 (Literal):** Decisive winner between COS300 and WSD300 at step 200, else tie.
  * **D2 (Matched Budget):** Decisive winner between COS200 and WSD200 at step 200, else tie.
  * **KEEP Recommendation:**
    * If D2 favours cosine $\to$ KEEP Cosine.
    * If D2 is a tie or favours WSD $\to$ KEEP WSD (equal quality with branchability and reuse benefits).

---

## 8. Part 5: Minimum Fitting and Width-4096 Extrapolation
* **Quadratic Vertex Fit:** Fit $L(x) = ax^2 + bx + c$ via least squares on $x = \log_2(lr)$ over the 5 grid points centered on the grid argmin.
  $$ x^* = -\frac{b}{2a} $$
  Accepted if $a > 0$ and $x^*$ falls within the 5-point span; otherwise revert to discrete argmin and flag.
* **Bootstrap Confidence Intervals:** 1,000 bootstrap resamples of tuning seeds to generate 95% CI for $x^*$.
* **Basin Half-Width ($h_w$):** Span of $x$ (in $\log_2$ units) where mean loss is within $0.02$ nats of the minimum.
* **Scaling Law Fit:** Fit $x^*_w = a + b \cdot \log_2(w)$ via OLS across widths $\{256, 512, 1024\}$.
* **Width-4096 Prediction:** $lr_{4096} = 2^{a + 12b}$.
* **Confidence Rating:** Let $W_{95}$ be the 95% prediction interval width in $\log_2$ units:
  * If $W_{95} \le h_{1024}$: "moderate".
  * If $h_{1024} < W_{95} \le 2 \cdot h_{1024}$: "low-moderate".
  * Otherwise: "low".
  * **Cap:** Strictly capped at "moderate" due to 4× extrapolation, fixed depth ($L=4$), short horizon (300 steps), and single corpus.

---

## 9. Divergence Handling
Any run encountering non-finite loss or non-finite gradient norm is marked `diverged=true`, stopped immediately, and assigned a penalty evaluation loss of **5.0 nats** (approx. $\ln(65) + 0.83$).

---

## 10. Hypotheses to Test (Not Pre-Determined Outcomes)
* **H1:** At step 200, COS300 achieves lower evaluation loss than WSD300 because cosine is already in decay ($\approx 37\%$ of peak LR) while WSD is still at peak LR ($100\%$).
* **H2:** At matched compute budget ($T=200$), COS200 and WSD200 achieve comparable evaluation loss within statistical noise.
* **H3:** Optimal learning rate decreases monotonically with model width.
* **H4:** The empirical scaling slope $b = \frac{d \log_2(lr^*)}{d \log_2(w)}$ falls between $-1.0$ and $0.0$.
* **H5:** Evaluation loss at the optimal learning rate strictly improves with increasing model width.
