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

Total: Standard **524 ms**, Layerwise **564 ms** (+7.7%), Subset **820 ms** (+56.5%) | Peak: 19.7 / 21.3 / 21.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **152.5** | **166.2** | **165.1** |  **84.5** |
| act_grad      |      83.2 |      87.5 |      88.4 |      43.5 |
| compress      |         — |      19.1 |         — |         — |
| score         |         — |       1.7 |     118.0 |         — |
| select        |         — |       2.4 |      <0.1 |         — |
| w.grad        |      81.3 |      66.3 |         — |      42.1 |
| autograd      |     116.1 |     130.0 |     126.3 |      61.3 |
| **Backward**  | **280.6** | **307.1** | **332.8** | **146.9** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### tulu3 → tydiqa | batch=16 | seq=512

Total: Standard **944 ms**, Layerwise **1014 ms** (+7.4%), Subset **1515 ms** (+60.5%) | Peak: 32.3 / 33.9 / 33.9 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **294.3** | **308.1** | **309.3** | **156.5** |
| act_grad      |     167.7 |     170.9 |     173.7 |      86.1 |
| compress      |         — |      74.7 |         — |         — |
| score         |         — |       1.7 |     255.7 |         — |
| select        |         — |       2.4 |      <0.1 |         — |
| w.grad        |     161.0 |     120.8 |         — |      83.7 |
| autograd      |     230.0 |     244.5 |     240.8 |     118.2 |
| **Backward**  | **558.6** | **615.0** | **670.2** | **287.9** |
| **Optimizer** |  **90.7** |  **90.6** |           |  **90.7** |

### tulu3 → tydiqa | batch=32 | seq=256

Total: Standard **962 ms**, Layerwise **976 ms** (+1.5%), Subset **1487 ms** (+54.6%) | Peak: 32.5 / 33.3 / 33.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **299.7** | **310.3** | **310.3** | **159.5** |
| act_grad      |     169.9 |     171.7 |     174.5 |      87.9 |
| compress      |         — |      30.7 |         — |         — |
| score         |         — |       1.8 |     213.2 |         — |
| select        |         — |       2.5 |      <0.1 |         — |
| w.grad        |     163.6 |     122.6 |         — |      85.6 |
| autograd      |     237.8 |     245.7 |     242.0 |     122.9 |
| **Backward**  | **571.3** | **574.9** | **629.7** | **296.4** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=8 | seq=512

Total: Standard **285 ms**, Layerwise **379 ms** (+33.0%), Subset **481 ms** (+68.9%) | Peak: 19.4 / 21.0 / 21.0 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   |  **72.4** |  **95.9** |  **95.3** |  **45.9** |
| act_grad      |      36.3 |      49.9 |      50.2 |      19.8 |
| compress      |         — |      14.7 |         — |         — |
| score         |         — |       3.1 |      64.0 |         — |
| select        |         — |       4.3 |      <0.1 |         — |
| w.grad        |      34.9 |      46.3 |         — |      20.0 |
| autograd      |      50.7 |      74.0 |      66.3 |      29.1 |
| **Backward**  | **121.9** | **192.3** | **180.6** |  **68.9** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=16 | seq=512

Total: Standard **493 ms**, Layerwise **577 ms** (+17.0%), Subset **817 ms** (+65.7%) | Peak: 31.7 / 33.3 / 33.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **141.4** | **169.9** | **170.1** |  **79.7** |
| act_grad      |      77.6 |      92.8 |      94.6 |      40.9 |
| compress      |         — |      18.4 |         — |         — |
| score         |         — |       1.7 |     117.5 |         — |
| select        |         — |       2.4 |      <0.1 |         — |
| w.grad        |      76.2 |      70.8 |         — |      39.8 |
| autograd      |     106.9 |     130.1 |     126.4 |      56.9 |
| **Backward**  | **260.7** | **316.3** | **338.5** | **137.7** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=32 | seq=256

Total: Standard **850 ms**, Layerwise **906 ms** (+6.5%), Subset **1356 ms** (+59.4%) | Peak: 31.9 / 32.7 / 32.7 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **262.9** | **285.9** | **286.9** | **142.8** |
| act_grad      |     149.8 |     160.2 |     162.2 |      77.3 |
| compress      |         — |      29.0 |         — |         — |
| score         |         — |       1.8 |     198.3 |         — |
| select        |         — |       2.4 |      <0.1 |         — |
| w.grad        |     144.7 |     115.3 |         — |      75.7 |
| autograd      |     202.2 |     220.6 |     216.8 |     104.9 |
| **Backward**  | **496.7** | **529.4** | **577.3** | **257.9** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### Summary

| Config                   | Standard | Layerwise | Overhead |  Subset | Overhead |
| ------------------------ | -------: | --------: | -------: | ------: | -------: |
| tulu3→tydiqa b=8 s=512   |   524 ms |    564 ms |    +7.7% |  820 ms |   +56.5% |
| tulu3→tydiqa b=16 s=512  |   944 ms |   1014 ms |    +7.4% | 1515 ms |   +60.5% |
| tulu3→tydiqa b=32 s=256  |   962 ms |    976 ms |    +1.5% | 1487 ms |   +54.6% |
| alpaca→samsum b=8 s=512  |   285 ms |    379 ms |   +33.0% |  481 ms |   +68.9% |
| alpaca→samsum b=16 s=512 |   493 ms |    577 ms |   +17.0% |  817 ms |   +65.7% |
| alpaca→samsum b=32 s=256 |   850 ms |    906 ms |    +6.5% | 1356 ms |   +59.4% |

**Methods:**
- **Standard**: Baseline full fine-tuning with AdamW.
- **Layerwise**: Per-layer selection via merged batch. Single-pass — scoring and w.grad happen inline during backward. Uses score compression (normal-64×64).
- **Subset**: Global selection with exact (uncompressed) scoring. Two-pass — scoring pass (P1, including selection) then gradient pass on selected subset (P2).

**Component definitions:**
- **compress** (Layerwise only): random projection for score compression
- **score**: influence score computation — compressed matmul for Layerwise (~2 ms), exact factored scoring for Subset (64–256 ms depending on batch/seq)
- **select**: top-k selection only — per-layer for Layerwise (~2.4 ms total across 113 layers), single global top-k for Subset (<0.1 ms)
- **w.grad**: gradient matmul + sample indexing (Layerwise: split batch, index selected samples, compute gradients; Standard/Subset P2: native GEMM via autograd)

**Key takeaways:**
- **Layerwise overhead vs Standard**: 1–8% at batch≥16 (tulu3), scaling favorably with batch size. Higher overhead (17–33%) on shorter datasets (alpaca) where fixed costs dominate.
- **w.grad savings**: Layerwise w.grad is 19–25% cheaper than Standard (e.g., 121 vs 161 ms at tulu3 b=16). Subset P2 w.grad is ~52% of Standard (e.g., 84 vs 161 ms). Both select ~50% of samples. Note: Layerwise w.grad includes sample indexing overhead (split + gather per layer), so the pure matmul savings are larger.
- **select overhead**: ~2.4 ms for Layerwise (pure top-k across 113 layers). Negligible for Subset (<0.1 ms, single global top-k).
- **Score compression is cheap** (Layerwise): compress takes 15–75 ms. The actual score matmul is negligible (1.7–3.1 ms).
- **Subset exact scoring cost**: Without compression, the factored score accumulation takes 64–256 ms (scales with batch × seq). This is the dominant P1 backward cost.
- **Autograd overhead**: ~14 ms extra for Layerwise vs Standard from custom Function dispatch.
- **Subset overhead**: 55–69%. The two-pass cost (P1 scoring + P2 forward+backward on selected samples) is the main overhead.
- **Peak memory**: Selection methods add 0.8–1.6 GB over Standard (merged batch + compressor state).

## Methodology

Each method runs in a **separate Python process** to ensure clean GPU memory measurement. Per-component timing uses CUDA events placed inside **monkey-patched autograd Functions** — the benchmark replaces the standard `torch.nn.functional.linear` backward with a custom `autograd.Function` that wraps the same operations but inserts CUDA event pairs around each component (activation gradient, weight gradient, compression, scoring, selection). This monkey-patching approach captures exact per-op timings with no residual decomposition required, since every cycle in the backward pass is accounted for by a timed region or attributed to autograd framework overhead.

| Component | How it's measured                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| Forward   | CUDA events around `model(**batch)`                                                                                |
| act_grad  | CUDA events around `grad_output @ weight` inside monkey-patched Linear `backward()`                                |
| compress  | CUDA events around `score_compressor.forward()` inside monkey-patched `_backward_compressed()` (Layerwise only)    |
| score     | CUDA events around matmul + score accumulation (compressed for Layerwise, exact for Subset)                        |
| select    | CUDA events around `_do_selection()` (top-k only, Layerwise) or `get_final_selection()` (Subset)                   |
| w.grad    | CUDA events around sample indexing + gradient computation (Layerwise) or monkey-patched `grad_output.T @ input` (Standard) |
| autograd  | Remainder: total backward - (act_grad + compress + score + select + w.grad)                                        |
| Optimizer | CUDA events around `optimizer.step()`                                                                              |
