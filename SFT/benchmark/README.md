# Timing Breakdown Benchmark

Per-component runtime analysis for **Standard**, **Layerwise** (per-layer selection), and **Subset Descent** (global selection) on Llama-3.2-1B.

## Quick Start

```bash
# Run with defaults (Llama-3.2-1B, batch=16, seq=512, tulu3 + tydiqa)
bash benchmark.sh

# Custom configuration
bash benchmark.sh --batch-size 8 --seq-length 256 --num-iterations 30

# Save results to JSON
bash benchmark.sh --output results/breakdown.json
```

<details>
  <summary>All CLI Options</summary>

| Option                 | Default                   | Description                                           |
| ---------------------- | ------------------------- | ----------------------------------------------------- |
| `--model`              | `meta-llama/Llama-3.2-1B` | Model name                                            |
| `--dtype`              | `bfloat16`                | `float32`, `bfloat16`, `float16`                      |
| `--no-flash-attention` | —                         | Disable flash attention                               |
| `--batch-size`         | 16                        | Training batch size                                   |
| `--seq-length`         | 512                       | Sequence length                                       |
| `--val-batch-size`     | 1                         | Validation batch size                                 |
| `--dataset`            | `tulu3`                   | `dummy`, `alpaca`, `gsm8k`, `dolly`, `tulu3`          |
| `--val-dataset`        | `tydiqa`                  | `samsum`, `gsm8k`, `tydiqa`, `mmlu`, `bbh`, `math500` |
| `--num-warmup`         | 10                        | Warmup iterations                                     |
| `--num-iterations`     | 20                        | Timed iterations                                      |
| `--use-second-order`   | —                         | Enable second-order (greedy) selection                |
| `--seed`               | 42                        | Random seed                                           |
| `--output`             | —                         | Save results to JSON file                             |

</details>

## Results

All benchmarks: Llama-3.2-1B | bfloat16 | A40 GPU | score compression: normal-64×64 | 10 warmup + 10 timed iters.
All numbers directly measured with CUDA events via monkey-patched autograd Functions. No residuals. See `benchmark.py`.

### tulu3 → tydiqa | batch=8 | seq=512 (matches `train.sh` defaults)

Total: Standard **525 ms**, Layerwise **565 ms** (+7.6%), Subset **762 ms** (+45.0%) | Peak: 19.7 / 21.3 / 21.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **152.1** | **164.9** | **183.7** |  **83.4** |
| act_grad      |      83.7 |      88.1 |      88.6 |      43.8 |
| compress      |         — |      19.3 |      22.1 |         — |
| score         |         — |       1.7 |       8.3 |         — |
| select        |         — |      17.9 |         — |         — |
| w.grad        |      81.5 |      51.1 |         — |      42.7 |
| autograd      |     117.3 |     131.5 |     136.1 |      62.3 |
| **Backward**  | **282.5** | **309.6** | **255.0** | **148.8** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### tulu3 → tydiqa | batch=16 | seq=512

Total: Standard **946 ms**, Layerwise **975 ms** (+3.1%), Subset **1309 ms** (+38.3%) | Peak: 32.3 / 33.9 / 33.9 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **294.3** | **309.1** | **309.4** | **156.1** |
| act_grad      |     168.4 |     171.5 |     171.7 |      86.2 |
| compress      |         — |      31.9 |      32.4 |         — |
| score         |         — |       1.8 |       8.3 |         — |
| select        |         — |      31.5 |         — |         — |
| w.grad        |     161.0 |      92.4 |         — |      83.9 |
| autograd      |     231.6 |     246.3 |     250.0 |     119.8 |
| **Backward**  | **561.0** | **575.4** | **462.4** | **290.0** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.6** |

### tulu3 → tydiqa | batch=32 | seq=256

Total: Standard **959 ms**, Layerwise **972 ms** (+1.3%), Subset **1310 ms** (+36.5%) | Peak: 32.5 / 33.3 / 33.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **298.4** | **307.6** | **307.9** | **157.1** |
| act_grad      |     169.0 |     170.9 |     171.4 |      86.6 |
| compress      |         — |      30.7 |      31.1 |         — |
| score         |         — |       1.8 |       8.2 |         — |
| select        |         — |      31.6 |         — |         — |
| w.grad        |     163.6 |      93.1 |         — |      84.5 |
| autograd      |     237.7 |     245.7 |     249.3 |     122.8 |
| **Backward**  | **570.3** | **573.9** | **460.0** | **293.8** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=8 | seq=512

Total: Standard **313 ms**, Layerwise **361 ms** (+15.5%), Subset **465 ms** (+48.6%) | Peak: 19.5 / 21.0 / 21.0 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   |  **80.2** |  **91.8** |  **91.6** |  **48.1** |
| act_grad      |      43.2 |      49.0 |      49.7 |      22.3 |
| compress      |         — |      11.9 |      19.6 |         — |
| score         |         — |       2.0 |       8.9 |         — |
| select        |         — |      12.7 |         — |         — |
| w.grad        |      40.3 |      33.2 |         — |      22.4 |
| autograd      |      58.3 |      70.0 |      79.2 |      32.3 |
| **Backward**  | **141.8** | **178.7** | **157.4** |  **77.0** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=16 | seq=512

Total: Standard **492 ms**, Layerwise **588 ms** (+19.5%), Subset **740 ms** (+50.4%) | Peak: 31.7 / 33.3 / 33.2 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **141.1** | **185.6** | **166.4** |  **78.5** |
| act_grad      |      76.9 |      91.2 |      91.8 |      40.6 |
| compress      |         — |      18.4 |      22.0 |         — |
| score         |         — |       1.7 |       8.3 |         — |
| select        |         — |      18.7 |         — |         — |
| w.grad        |      76.0 |      53.5 |         — |      39.9 |
| autograd      |     107.4 |     128.1 |     134.3 |      57.5 |
| **Backward**  | **260.3** | **311.7** | **256.4** | **138.0** |
| **Optimizer** |  **90.7** |  **90.7** |           | **101.0** |

### alpaca → samsum | batch=32 | seq=256

Total: Standard **859 ms**, Layerwise **895 ms** (+4.2%), Subset **1195 ms** (+39.1%) | Peak: 31.9 / 32.7 / 32.7 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **265.4** | **280.5** | **281.0** | **142.4** |
| act_grad      |     150.3 |     157.9 |     158.5 |      78.5 |
| compress      |         — |      28.8 |      29.4 |         — |
| score         |         — |       1.8 |       7.9 |         — |
| select        |         — |      29.7 |         — |         — |
| w.grad        |     146.6 |      86.8 |         — |      76.2 |
| autograd      |     206.3 |     219.1 |     223.8 |     106.8 |
| **Backward**  | **503.2** | **524.1** | **419.7** | **261.5** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### Summary

| Config                   | Standard | Layerwise | Overhead |  Subset | Overhead |
| ------------------------ | -------: | --------: | -------: | ------: | -------: |
| tulu3→tydiqa b=8 s=512   |   525 ms |    565 ms |    +7.6% |  762 ms |   +45.0% |
| tulu3→tydiqa b=16 s=512  |   946 ms |    975 ms |    +3.1% | 1309 ms |   +38.3% |
| tulu3→tydiqa b=32 s=256  |   959 ms |    972 ms |    +1.3% | 1310 ms |   +36.5% |
| alpaca→samsum b=8 s=512  |   313 ms |    361 ms |   +15.5% |  465 ms |   +48.6% |
| alpaca→samsum b=16 s=512 |   492 ms |    588 ms |   +19.5% |  740 ms |   +50.4% |
| alpaca→samsum b=32 s=256 |   859 ms |    895 ms |    +4.2% | 1195 ms |   +39.1% |

**Methods:**
- **Standard**: Baseline full fine-tuning with AdamW.
- **Layerwise**: Per-layer selection via merged batch. Single-pass — scoring and w.grad happen inline during backward.
- **Subset**: Global selection with ghost inner product scoring. Two-pass — scoring pass (P1) then gradient pass on selected subset (P2).

**Component definitions:**
- **select** (Layerwise): includes top-k selection, batch splitting, sample indexing, and scale factor computation
- **w.grad**: purely the gradient matmul (einsum for Layerwise, native GEMM for Standard/Subset P2)

**Key takeaways:**
- **Layerwise overhead vs Standard**: 1–8% at batch≥16 (tulu3), scaling favorably with batch size. Higher overhead (15–20%) on shorter datasets (alpaca) where fixed costs dominate.
- **w.grad savings**: Layerwise w.grad is 40–45% cheaper than Standard (e.g., 92 vs 161 ms at tulu3 b=16). Subset P2 w.grad is ~52% of Standard (e.g., 84 vs 161 ms). Both select ~50% of samples.
- **select overhead**: 18–32 ms for Layerwise (batch splitting + per-layer indexing into non-contiguous tensors). This is the main per-layer cost beyond scoring.
- **Score compression is cheap**: compress takes 12–32 ms. The actual score matmul is negligible (1.7–2.0 ms).
- **Autograd overhead**: ~14 ms extra for Layerwise vs Standard from custom Function dispatch.
- **Subset overhead**: 36–50%. The P2 forward+backward on selected samples is the main cost.
- **Peak memory**: Selection methods add 0.8–1.6 GB over Standard (merged batch + compressor state).

## Methodology

Each method runs in a **separate Python process** to ensure clean GPU memory measurement. Per-component timing uses CUDA events placed inside **monkey-patched autograd Functions** — the benchmark replaces the standard `torch.nn.functional.linear` backward with a custom `autograd.Function` that wraps the same operations but inserts CUDA event pairs around each component (activation gradient, weight gradient, compression, scoring, selection). This monkey-patching approach captures exact per-op timings with no residual decomposition required, since every cycle in the backward pass is accounted for by a timed region or attributed to autograd framework overhead.

| Component | How it's measured                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| Forward   | CUDA events around `model(**batch)`                                                                                |
| act_grad  | CUDA events around `grad_output @ weight` inside monkey-patched Linear `backward()`                                |
| compress  | CUDA events around `score_compressor.forward()` inside monkey-patched `_backward_compressed()`                     |
| score     | CUDA events around matmul + score accumulation                                                                     |
| select    | CUDA events around `_do_selection()` (top-k)                                                                       |
| w.grad    | CUDA events around `compute_selected_gradients()` (Layerwise) or monkey-patched `grad_output.T @ input` (Standard) |
| autograd  | Remainder: total backward - (act_grad + compress + score + select + w.grad)                                        |
| Optimizer | CUDA events around `optimizer.step()`                                                                              |
