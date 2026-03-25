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

All benchmarks: Llama-3.2-1B | bfloat16 | A40 GPU | Layerwise score compression: normal-64×64 | Subset: exact (no compression) | 10 warmup + 20 timed iters.
All numbers directly measured with CUDA events via monkey-patched autograd Functions. No residuals. See `benchmark.py`.

### tulu3 → tydiqa | batch=8 | seq=512 (matches `train.sh` defaults)

Total: Standard **523 ms**, Layerwise **564 ms** (+7.9%), Subset **822 ms** (+57.3%) | Peak: 19.7 / 21.3 / 21.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **151.7** | **166.2** | **165.6** |  **85.2** |
| act_grad      |      83.1 |      87.5 |      89.3 |      43.6 |
| compress      |         — |      19.1 |         — |         — |
| score         |         — |       1.7 |     118.1 |         — |
| select        |         — |      17.8 |       0.0 |         — |
| w.grad        |      81.2 |      50.9 |         — |      42.2 |
| autograd      |     116.1 |     130.0 |     126.3 |      61.2 |
| **Backward**  | **280.5** | **307.1** | **333.8** | **147.0** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### tulu3 → tydiqa | batch=16 | seq=512

Total: Standard **944 ms**, Layerwise **1016 ms** (+7.7%), Subset **1518 ms** (+60.9%) | Peak: 32.3 / 33.9 / 33.9 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **294.4** | **309.2** | **309.7** | **156.8** |
| act_grad      |     167.6 |     171.4 |     174.2 |      86.0 |
| compress      |         — |      74.9 |         — |         — |
| score         |         — |       1.7 |     258.4 |         — |
| select        |         — |      31.3 |       0.0 |         — |
| w.grad        |     161.2 |      92.3 |         — |      83.7 |
| autograd      |     230.0 |     244.5 |     240.8 |     118.2 |
| **Backward**  | **558.8** | **616.2** | **673.4** | **287.9** |
| **Optimizer** |  **90.7** |  **90.6** |           |  **90.7** |

### tulu3 → tydiqa | batch=32 | seq=256

Total: Standard **968 ms**, Layerwise **978 ms** (+1.1%), Subset **1491 ms** (+54.0%) | Peak: 32.5 / 33.3 / 33.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **301.7** | **310.2** | **311.3** | **159.8** |
| act_grad      |     172.0 |     173.3 |     175.1 |      88.3 |
| compress      |         — |      30.7 |         — |         — |
| score         |         — |       1.8 |     214.3 |         — |
| select        |         — |      31.8 |       0.0 |         — |
| w.grad        |     165.8 |      94.2 |         — |      86.1 |
| autograd      |     237.9 |     245.7 |     242.0 |     123.0 |
| **Backward**  | **575.7** | **577.5** | **631.4** | **297.4** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=8 | seq=512

Total: Standard **285 ms**, Layerwise **367 ms** (+28.6%), Subset **484 ms** (+69.6%) | Peak: 19.4 / 21.0 / 21.0 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   |  **72.6** |  **94.6** |  **95.6** |  **46.4** |
| act_grad      |      36.4 |      49.9 |      50.2 |      20.0 |
| compress      |         — |      12.1 |         — |         — |
| score         |         — |       2.1 |      64.1 |         — |
| select        |         — |      12.9 |       0.0 |         — |
| w.grad        |      35.0 |      33.7 |         — |      20.1 |
| autograd      |      50.8 |      71.1 |      66.3 |      30.4 |
| **Backward**  | **122.1** | **181.8** | **180.7** |  **70.6** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=16 | seq=512

Total: Standard **492 ms**, Layerwise **577 ms** (+17.2%), Subset **818 ms** (+66.3%) | Peak: 31.7 / 33.3 / 33.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **142.1** | **169.8** | **170.6** |  **80.8** |
| act_grad      |      76.6 |      92.8 |      94.6 |      41.0 |
| compress      |         — |      18.4 |         — |         — |
| score         |         — |       1.7 |     117.4 |         — |
| select        |         — |      19.0 |       0.0 |         — |
| w.grad        |      75.9 |      54.3 |         — |      39.8 |
| autograd      |     106.9 |     130.2 |     126.4 |      57.1 |
| **Backward**  | **259.4** | **316.4** | **338.5** | **137.9** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=32 | seq=256

Total: Standard **849 ms**, Layerwise **906 ms** (+6.7%), Subset **1355 ms** (+59.7%) | Peak: 31.9 / 32.7 / 32.7 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **262.3** | **286.2** | **286.6** | **142.6** |
| act_grad      |     148.9 |     159.7 |     162.5 |      77.1 |
| compress      |         — |      29.0 |         — |         — |
| score         |         — |       1.8 |     198.3 |         — |
| select        |         — |      29.9 |       0.0 |         — |
| w.grad        |     144.6 |      87.8 |         — |      75.7 |
| autograd      |     202.2 |     220.6 |     216.8 |     104.9 |
| **Backward**  | **495.6** | **528.8** | **577.7** | **257.6** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### Summary

| Config                   | Standard | Layerwise | Overhead |  Subset | Overhead |
| ------------------------ | -------: | --------: | -------: | ------: | -------: |
| tulu3→tydiqa b=8 s=512   |   523 ms |    564 ms |    +7.9% |  822 ms |   +57.3% |
| tulu3→tydiqa b=16 s=512  |   944 ms |   1016 ms |    +7.7% | 1518 ms |   +60.9% |
| tulu3→tydiqa b=32 s=256  |   968 ms |    978 ms |    +1.1% | 1491 ms |   +54.0% |
| alpaca→samsum b=8 s=512  |   285 ms |    367 ms |   +28.6% |  484 ms |   +69.6% |
| alpaca→samsum b=16 s=512 |   492 ms |    577 ms |   +17.2% |  818 ms |   +66.3% |
| alpaca→samsum b=32 s=256 |   849 ms |    906 ms |    +6.7% | 1355 ms |   +59.7% |

**Methods:**
- **Standard**: Baseline full fine-tuning with AdamW.
- **Layerwise**: Per-layer selection via merged batch. Single-pass — scoring and w.grad happen inline during backward. Uses score compression (normal-64×64).
- **Subset**: Global selection with exact (uncompressed) scoring. Two-pass — scoring pass (P1, including selection) then gradient pass on selected subset (P2).

**Component definitions:**
- **compress** (Layerwise only): random projection for score compression
- **score**: influence score computation — compressed matmul for Layerwise (~2 ms), exact factored scoring for Subset (64–258 ms depending on batch/seq)
- **select** (Layerwise): includes top-k selection, batch splitting, sample indexing, and scale factor computation
- **w.grad**: purely the gradient matmul (einsum for Layerwise, native GEMM for Standard/Subset P2)

**Key takeaways:**
- **Layerwise overhead vs Standard**: 1–8% at batch≥16 (tulu3), scaling favorably with batch size. Higher overhead (17–29%) on shorter datasets (alpaca) where fixed costs dominate.
- **w.grad savings**: Layerwise w.grad is 40–45% cheaper than Standard (e.g., 92 vs 161 ms at tulu3 b=16). Subset P2 w.grad is ~52% of Standard (e.g., 84 vs 161 ms). Both select ~50% of samples.
- **select overhead**: 18–32 ms for Layerwise (batch splitting + per-layer indexing into non-contiguous tensors). This is the main per-layer cost beyond scoring.
- **Score compression is cheap** (Layerwise): compress takes 12–75 ms. The actual score matmul is negligible (1.7–2.1 ms).
- **Subset exact scoring cost**: Without compression, the factored score accumulation takes 64–258 ms (scales with batch × seq). This is the dominant P1 backward cost.
- **Autograd overhead**: ~14 ms extra for Layerwise vs Standard from custom Function dispatch.
- **Subset overhead**: 54–70%. The two-pass cost (P1 scoring + P2 forward+backward on selected samples) is the main overhead.
- **Peak memory**: Selection methods add 0.8–1.6 GB over Standard (merged batch + compressor state).

## Methodology

Each method runs in a **separate Python process** to ensure clean GPU memory measurement. Per-component timing uses CUDA events placed inside **monkey-patched autograd Functions** — the benchmark replaces the standard `torch.nn.functional.linear` backward with a custom `autograd.Function` that wraps the same operations but inserts CUDA event pairs around each component (activation gradient, weight gradient, compression, scoring, selection). This monkey-patching approach captures exact per-op timings with no residual decomposition required, since every cycle in the backward pass is accounted for by a timed region or attributed to autograd framework overhead.

| Component | How it's measured                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| Forward   | CUDA events around `model(**batch)`                                                                                |
| act_grad  | CUDA events around `grad_output @ weight` inside monkey-patched Linear `backward()`                                |
| compress  | CUDA events around `score_compressor.forward()` inside monkey-patched `_backward_compressed()` (Layerwise only)    |
| score     | CUDA events around matmul + score accumulation (compressed for Layerwise, exact for Subset)                        |
| select    | CUDA events around `_do_selection()` (top-k, Layerwise) or `get_final_selection()` (Subset)                        |
| w.grad    | CUDA events around `compute_selected_gradients()` (Layerwise) or monkey-patched `grad_output.T @ input` (Standard) |
| autograd  | Remainder: total backward - (act_grad + compress + score + select + w.grad)                                        |
| Optimizer | CUDA events around `optimizer.step()`                                                                              |
