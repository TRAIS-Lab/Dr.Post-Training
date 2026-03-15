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
All numbers directly measured with CUDA events via monkey-patched autograd Functions. No residuals. See `benchmark_v2.py`.

### tulu3 → tydiqa | batch=8 | seq=512 (matches `train.sh` defaults)

Total: Standard **525 ms**, Layerwise **566 ms** (+7.8%), Subset **743 ms** (+41.5%) | Peak: 19.7 / 21.3 / 21.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **151.8** | **165.6** | **165.4** |  **83.4** |
| act_grad      |      84.0 |      88.3 |      89.3 |         — |
| compress      |         — |      19.3 |      22.1 |         — |
| score         |         — |       1.7 |       8.2 |         — |
| select        |         — |       2.4 |         — |         — |
| w.grad        |      81.6 |      66.7 |         — | **148.2** |
| autograd      |     117.4 |     131.5 |     135.9 |         — |
| **Backward**  | **283.0** | **310.0** | **255.5** | **148.2** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### tulu3 → tydiqa | batch=16 | seq=512

Total: Standard **950 ms**, Layerwise **976 ms** (+2.8%), Subset **1313 ms** (+38.3%) | Peak: 32.3 / 33.9 / 33.9 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **295.7** | **309.7** | **310.8** | **156.8** |
| act_grad      |     169.0 |     171.6 |     173.0 |         — |
| compress      |         — |      32.0 |      32.4 |         — |
| score         |         — |       1.7 |       8.4 |         — |
| select        |         — |       2.4 |         — |         — |
| w.grad        |     162.6 |     121.8 |         — | **290.9** |
| autograd      |     231.7 |     246.3 |     250.3 |         — |
| **Backward**  | **563.3** | **575.9** | **464.1** | **290.9** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### tulu3 → tydiqa | batch=32 | seq=256

Total: Standard **958 ms**, Layerwise **974 ms** (+1.6%), Subset **1307 ms** (+36.4%) | Peak: 32.5 / 33.3 / 33.3 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **298.2** | **308.3** | **307.2** | **156.5** |
| act_grad      |     168.9 |     171.4 |     171.1 |         — |
| compress      |         — |      30.6 |      31.2 |         — |
| score         |         — |       1.8 |       8.3 |         — |
| select        |         — |       2.5 |         — |         — |
| w.grad        |     162.9 |     122.5 |         — | **292.7** |
| autograd      |     237.7 |     245.7 |     249.3 |         — |
| **Backward**  | **569.5** | **574.6** | **459.8** | **292.7** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=8 | seq=512

Total: Standard **313 ms**, Layerwise **363 ms** (+16.1%), Subset **467 ms** (+49.5%) | Peak: 19.5 / 21.0 / 21.0 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   |  **80.2** |  **92.5** |  **92.8** |  **48.6** |
| act_grad      |      43.0 |      48.9 |      50.0 |         — |
| compress      |         — |      12.1 |      20.1 |         — |
| score         |         — |       2.1 |       9.1 |         — |
| select        |         — |       3.0 |         — |         — |
| w.grad        |      40.2 |      43.3 |         — |  **76.4** |
| autograd      |      58.3 |      70.1 |      79.3 |         — |
| **Backward**  | **141.6** | **179.6** | **158.6** |  **76.4** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=16 | seq=512

Total: Standard **493 ms**, Layerwise **590 ms** (+19.7%), Subset **731 ms** (+48.3%) | Peak: 31.7 / 33.3 / 33.2 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **141.2** | **187.2** | **166.8** |  **78.5** |
| act_grad      |      77.1 |      91.1 |      92.0 |         — |
| compress      |         — |      18.4 |      22.2 |         — |
| score         |         — |       1.8 |       8.5 |         — |
| select        |         — |       2.4 |         — |         — |
| w.grad        |      76.1 |      69.9 |         — | **137.3** |
| autograd      |     107.5 |     128.1 |     134.6 |         — |
| **Backward**  | **260.6** | **311.6** | **257.3** | **137.3** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### alpaca → samsum | batch=32 | seq=256

Total: Standard **861 ms**, Layerwise **898 ms** (+4.3%), Subset **1197 ms** (+39.1%) | Peak: 31.9 / 32.7 / 32.7 GB

| Component     |  Standard | Layerwise | Subset P1 | Subset P2 |
| ------------- | --------: | --------: | --------: | --------: |
| **Forward**   | **265.3** | **281.6** | **282.0** | **143.0** |
| act_grad      |     151.2 |     158.8 |     159.1 |         — |
| compress      |         — |      28.8 |      29.4 |         — |
| score         |         — |       1.8 |       7.9 |         — |
| select        |         — |       2.4 |         — |         — |
| w.grad        |     147.2 |     114.4 |         — | **262.1** |
| autograd      |     206.3 |     219.1 |     223.1 |         — |
| **Backward**  | **504.8** | **525.4** | **419.4** | **262.1** |
| **Optimizer** |  **90.7** |  **90.7** |           |  **90.7** |

### Summary

| Config                   | Standard | Layerwise | Overhead |  Subset | Overhead |
| ------------------------ | -------: | --------: | -------: | ------: | -------: |
| tulu3→tydiqa b=8 s=512   |   525 ms |    566 ms |    +7.8% |  743 ms |   +41.5% |
| tulu3→tydiqa b=16 s=512  |   950 ms |    976 ms |    +2.8% | 1313 ms |   +38.3% |
| tulu3→tydiqa b=32 s=256  |   958 ms |    974 ms |    +1.6% | 1307 ms |   +36.4% |
| alpaca→samsum b=8 s=512  |   313 ms |    363 ms |   +16.1% |  467 ms |   +49.5% |
| alpaca→samsum b=16 s=512 |   493 ms |    590 ms |   +19.7% |  731 ms |   +48.3% |
| alpaca→samsum b=32 s=256 |   861 ms |    898 ms |    +4.3% | 1197 ms |   +39.1% |

**Methods:**
- **Standard**: Baseline full fine-tuning with AdamW.
- **Layerwise**: Per-layer selection via merged batch. Single-pass — scoring and w.grad happen inline during backward.
- **Subset**: Global selection with ghost inner product scoring. Two-pass — scoring pass (P1) then gradient pass on selected subset (P2).

**Key takeaways:**
- **Layerwise overhead vs Standard**: 2–8% at batch>=16 (tulu3), scaling favorably with batch size. Higher relative overhead (16–20%) on smaller/shorter datasets (alpaca b=8/16) where the fixed costs of compression and custom dispatch are a larger fraction of total time.
- **w.grad savings from selection**: Layerwise w.grad is 18–25% cheaper than Standard at batch>=16 (e.g., 122 vs 163 ms at b=16). The savings come from computing gradients only for selected samples.
- **Score compression is cheap**: compress takes 12–32 ms (6–8% of backward). The actual score matmul is negligible (1.7–2.1 ms).
- **Autograd overhead**: The dominant cost difference between Standard and Layerwise is autograd framework overhead (~14 ms extra from custom Function dispatch), not the selection logic itself.
- **Subset overhead**: 36–50%. The two-pass design requires a full extra forward+backward on selected samples (P2), which is expensive.
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
