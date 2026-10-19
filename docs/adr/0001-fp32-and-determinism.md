# ADR 0001: Strict FP32 Arithmetic and Deterministic Training Setup

## Status
Accepted

## Context
Optimizer benchmarks and empirical reproducibility studies frequently suffer from non-deterministic behavior and non-reproducible loss trajectories caused by:
1. Floating point precision differences (bfloat16 vs float32 vs tensorfloat32).
2. Hardware accelerator non-determinism (e.g., non-deterministic atomic operations in cuBLAS, cuDNN auto-tuning heuristics).
3. Non-deterministic attention implementations (e.g., FlashAttention non-deterministic backward passes or hardware-dependent fast kernels).

In our study, we run controlled experiments across CPU and GPU (e.g., Google Colab Nvidia T4 accelerators). Nvidia T4 GPUs lack native bfloat16 tensor core acceleration, and mixed precision introduces stochastic rounding effects that confound fine-grained optimizer analysis (such as measuring update-to-weight ratios or exact step-level comparisons).

## Decision
1. **Precision:** We standardize exclusively on single-precision floating-point (`float32`) arithmetic across all training and evaluation passes.
2. **Deterministic Flags:** We explicitly mandate deterministic library configurations:
   * `CUBLAS_WORKSPACE_CONFIG=:4096:8` set prior to importing PyTorch.
   * `torch.use_deterministic_algorithms(True)`.
   * `torch.backends.cudnn.deterministic = True`.
   * `torch.backends.cudnn.benchmark = False`.
   * `torch.backends.cuda.matmul.allow_tf32 = False`.
   * `torch.backends.cudnn.allow_tf32 = False`.
   * `torch.set_float32_matmul_precision("highest")`.
3. **Attention Backend:** We force the deterministic `MATH` implementation in `scaled_dot_product_attention` using `torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)`.
4. **Independent Generators:** Separate `torch.Generator(device="cpu")` instances are used for parameter initialization (`init_gen`) and batch index sampling (`data_gen`), guaranteeing that weights and data batches are identical across different optimizer schedules.

## Consequences
* Training iterations are bitwise identical across repeat runs on the same hardware and PyTorch build.
* GPU runs execute in pure FP32 without hidden rounding or TF32 truncation.
* Training is marginally slower than unconstrained auto-tuned FP16/TF32 execution, but guarantees scientific rigor and determinism required for optimizer comparisons.
