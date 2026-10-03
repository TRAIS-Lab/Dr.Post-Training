# Timing Benchmark

Per-component runtime analysis for **Standard**, **LayerWiseSubset**,
**GlobalSubset (two-pass)**, and **GlobalSubset (one-pass)** with four
scoring mechanisms (`compress`, `gip`, `pip`, `direct`; `pip` = per-token inner product, `gip` = ghost inner product).

Two suites are maintained for different hardware:

- **A40 suite** — 3 small/mid models from different families, packed
  one-model-per-GPU on a single A40x4 node.
- **H200 suite** — one 10B+ model (Qwen3-14B-Base) at longer sequences on a single
  H200, one array task per `(n, T)`.

## Suites

### A40 — `run_benchmarks.sh` / `slurm/launch_all.sh`

3 architecture families to factor out family-specific kernel quirks.

| Model            | hidden | intermediate | L  | heads (q / kv) | head dim | GQA |
|------------------|--------|--------------|----|----------------|----------|-----|
| SmolLM2-360M     |  960   |  2560        | 32 | 15 / 5         | 64       | 3:1 |
| TinyLlama-1.1B   | 2048   |  5632        | 22 | 32 / 4         | 64       | 8:1 |
| Llama-3.2-3B     | 3072   |  8192        | 28 | 24 / 8         | 128      | 3:1 |

Configs: `(n=8, T=512, m=1)` and `(n=2, T=1024, m=1)`. One model per
GPU, three GPUs in parallel.

### H200 — `slurm/launch_h200.sh`

One model at the 8-14B scale on a single H200 (141 GB), submitted as one Slurm job array (one
GPU per task, one `(n, T)` per task, `MAX_CONCURRENT` throttle).

| Model           | hidden | intermediate | L  | heads (q / kv) | head dim | GQA |
|-----------------|--------|--------------|----|----------------|----------|-----|
| Qwen3-8B-Base   | 4096   | 12288        | 36 | 32 / 8         | 128      | 4:1 |
| Qwen3-14B-Base  | 5120   | 17408        | 40 | 40 / 8         | 128      | 5:1 |

Configs: `(n, T) = (8, 512), (16, 512), (32, 256), (8, 1024), (4, 1024), (16, 256), (4, 2048),
(2, 2048), (2, 4096)`, `m = 1`, `k = n/2`, plus one standalone scoring job (T-sweep to 32768).
Protocol: bf16 weights, activations (autocast) and Adam state, **fused AdamW** (`--fused-adamw`;
the default foreach implementation materialises full-size temporaries, `_foreach_sqrt` of the
second moments alone is ~30 GB at 14B), the CuTe backend's **Hopper kernels**
(`drpt/kernels/hopper_ops.py`, TMA + `wgmma`; see the Hopper note below), and **no activation
checkpointing** by default: Qwen3-8B fits up to ~9k tokens per step on one H200 (static state
66 GB, ~6.6 MB per token), the primary setting. `CKPT=1` turns checkpointing on, which Qwen3-14B
needs beyond ~2.5k tokens per step (static state ~105 GB, ~10 MB per token); those results land
in `breakdown_checkpointing/` and carry the two measurement effects described in the Hopper note.

Results: `results/h200/{breakdown,breakdown_checkpointing,scoring}/`; the paper tables come from
`SFT/tables/system_efficiency_h200.py` (`--model-tag`, `--subdir`, `--totals plain|phases`), which
reads the snapshot copied to `results/paper/h200/` like the A40 generator. The 1.7B/4B shapes
remain available in `benchmark_scoring.py` (`--model-tag`) and through `MODELS=`.

## Benchmarks

### Benchmark 1: Per-Component Breakdown

End-to-end training-step timing decomposed into forward, activation
backward, w.grad, score, and optimizer-step. Same `(n, T, m)` across all
models in the suite for fair comparison.

```bash
bash SFT/benchmark/run_benchmarks.sh breakdown      # A40, no checkpointing
bash SFT/benchmark/run_benchmarks.sh checkpointing  # A40, with gradient checkpointing
bash SFT/benchmark/slurm/launch_h200.sh             # H200 suite
```

Results: `results/breakdown/`, `results/breakdown_checkpointing/`,
`results/h200/`.

### Benchmark 2: Scoring Comparison

Standalone scoring-function timing with synthetic tensors. No model
loading — tests all scoring regimes at scale.

- **m-sweep** (T=512, m=1..16): gip vs pip crossover
- **T-sweep** (m=1, T=256..8192): pip vs direct crossover

```bash
bash SFT/benchmark/run_benchmarks.sh scoring
```

Results: `results/scoring/`. The GIP / PIP / direct rows time the library's dispatch on
synthetic activations (fused kernels when the backend provides them); the compressed row
times the fused `compressed_scores` path of the training step on the CuTe backend and a
torch simulation of the projection elsewhere. `benchmark_scoring.py --gpu N` is applied before
any import: the CuTe DSL initialises the CUDA driver when `drpt.kernels` is
imported, after which `CUDA_VISIBLE_DEVICES` is ignored (setting it inside
`main()` would put every model on GPU 0).

## Scoring Methods

Per-layer complexities as in the paper's Table `tab:scoring-comparison`
(`N = n + m`, `w = √(d/L)` the layer width, `κ` the compressed dimension):

| Method | FLOPs per layer | Memory (entries) | Cheapest when (FLOP count) |
|--------|-----------------|------------------|----------------------------|
| **compress** | O(NT(√(κ·d/L) + κ)) + (2n+m)κ | (n+1)κ | always (approximate scores) |
| **gip** | 4nmT²·√(d/L) | 2nmT² | `mT ≲ √(d/L)/2`: short T and small m |
| **pip** | 2NT·(d/L) + nT·√(d/L) | d/L + nT·√(d/L) | `T ≲ √(d/L)` and outside GIP's regime; overtakes GIP once `m ≳ √(d/L)/(2T)` |
| **direct** | 2NT·(d/L) + n·(d/L) | (n+1)·(d/L) | `T ≳ √(d/L)` (PIP and Direct share the leading term; only the lower-order terms differ) |

### Crossover Thresholds

**pip vs gip** — crossover at `V* = Σ(O·I) / (T · Σ(O+I))`:

| Model            | V* at T=512 | V* at T=1024 |
|------------------|-------------|--------------|
| SmolLM2-360M     | 1.1         | 0.6          |
| TinyLlama-1.1B   | 2.4         | 1.2          |
| Llama-3.2-3B     | 3.6         | 1.8          |
| Qwen3-1.7B       | 2.5         | 1.3          |
| Qwen3-4B         | 3.4         | 1.7          |
| Qwen3-8B         | 5.0         | 2.5          |

**pip vs direct** — crossover at `T* = Σ(O·I) / Σ(√(O·I))`:

| Model            | T*    |
|------------------|-------|
| SmolLM2-360M     | 1271  |
| TinyLlama-1.1B   | 2799  |
| Llama-3.2-3B     | 4069  |
| Qwen3-1.7B       | 2854  |
| Qwen3-4B         | 4088  |
| Qwen3-8B         | 5747  |

**Measured with the fused kernels** (`results/paper/scoring/`, A40, `n=8`;
Direct / GIP / PIP ms):

- `T=512, m=1`: GIP < PIP < Direct on all three models (56/29/40 SmolLM2,
  132/39/88 TinyLlama, 414/73/251 Llama-3.2-3B), as the FLOP count predicts
  (`T ≪ T*` and `mT < √(d/L)/2`). PIP is never the most expensive exact method.
- `m`-sweep at `T=512`: PIP overtakes GIP at `m=2` on SmolLM2 (43 vs 55),
  `m=4` on TinyLlama (121 vs 138) and `m=8` on Llama-3.2-3B (429 vs 508),
  matching `m ≳ √(d/L)/(2T)` (`V* = 1.1 / 2.4 / 3.6`). This needs the target
  gradient GEMM on the fused `total_wgrad` kernel; with the cuBLAS einsum
  PIP at `m=8` on Llama-3.2-3B takes 515 ms and GIP 487 ms.
- `T`-sweep at `m=1`: the FLOP-count crossover to Direct at `T*` does **not**
  show up. PIP stays the cheapest exact method through `T=8192` on all three
  models (463 vs 575, 1351 vs 1353, 3880 vs 4091 ms). PIP's lower-order term
  `nT√(d/L)` is the `[n, T, O]` intermediate, which the fused kernel keeps in
  registers, so it costs nothing; Direct's `n·(d/L)` term is a materialisation
  (a memory-bound write and read of `n` gradient matrices) that no kernel
  removes. Both share the leading `2NT(d/L)`, so at long `T` the two are within
  1–10 %. GIP is 3–14× PIP for `T ≥ 2048`; its `T=8192` cells move by
  15–30 % between runs (the `S=8192` kernel's per-call time fluctuates 80–137 ms
  on an idle GPU), every other cell by < 5 %.
- Compressed scoring is the cheapest everywhere (25–413 ms).

Compared with the PyTorch reference ops, the fused kernels keep the `T=512`
ordering and the `m`-crossovers, but Direct is not the cheapest at `T=2048`
(SmolLM2, TinyLlama) or at `T=8192` (all models), and GIP and PIP tie on
Llama-3.2-3B at `T=2048` (1001 vs 961 ms, within noise).

## Fused Kernels (drpt/kernels)

The custom Linear backward is GPU-bound at these sizes (kernel idle < 1.5 % of
the backward at `n=8, T=512`), so the lever is kernel time, not launch overhead.
Three fused tensor-core kernels replace the multi-kernel PyTorch paths of the
scoring and curated-w.grad steps. They are written in the **CuTe DSL** (CUTLASS
Python DSL, `drpt/kernels/cute_ops.py`; Ampere `cp.async` + `ldmatrix` +
`mma.sync` mainloop with predicated boundaries) with a Triton port kept as a
fallback backend (`drpt/kernels/triton_ops.py`). Backend selection:
`DRPT_KERNEL_BACKEND=cute|triton|off` (default `cute` when
`nvidia-cutlass-dsl` + `apache-tvm-ffi` are installed), or `--kernel-backend` /
`--no-fused` in the benchmark scripts. `off` runs the PyTorch reference ops.
`DRPT_FUSED_DISABLE=wgrad` (or `gip`, `pip`, `proj`, `gather`) sends one operation to the reference
path while the backend stays fused.

```bash
pip install nvidia-cutlass-dsl apache-tvm-ffi   # CuTe backend (CUDA 12+ driver)
```

| Kernel | Replaces | What is fused |
|--------|----------|---------------|
| `gip_scores` | `einsum` ×2 + mul + sum | both `[S, S]` dot-product tiles formed in registers (two mainloops, K = O then K = I), Hadamard + CTA reduction in the epilogue; no `[B, V, S, S]` intermediates, fp32 scores |
| `pip_scores` | `matmul` + fused mul-sum | GEMM `inp @ G_valᵀ` whose `[B, S, O]` result is dotted against the `go` tile in registers |
| `selected_wgrad` | gather ×2 + `einsum` + scale (+ bias sum) | GEMM whose K loop walks the selected samples through the index list (M-/N-major operands, no gathered copies); item-count scale fused into the bf16 store |
| `total_wgrad` | `einsum('vto,vti->oi')` | the target gradient `G_val = Σ_v go_vᵀ inp_v` (and every other batch-summed weight gradient, `compute_total_gradient`) through the `selected_wgrad` mainloop with all samples selected: ~20 % faster than the cuBLAS einsum for `K = m·T ≳ 2k` and immune to the `[3072, 8192]` heuristic bug below; used by the PIP and Direct scoring paths and the separate-batch validation cache |
| `compressed_grad` (CuTe only) | `Compressor.forward`: 2 skinny GEMMs + `bmm` + scale (torch.compile) | both `κ^{1/2}`-wide projections of a 64-row tile and their outer product `(go P_O)ᵀ(inp P_I)` formed on-chip; one launch + one tile-sum per layer, replaces the five-kernel compiled graph and its guard overhead |
| `reduce_select` (CuTe only) | partial sum, correction, `topk`, `sort` | single-CTA epilogue for the exact methods: the pip / gip kernels now expose their per-CTA partial sums (`pip_partials`, `gip_partials`); one launch turns them (plus the bias-gradient column when present) into corrected scores and the sorted top-k (`drpt.selection.backward.exact_scores_fused`, same `DRPT_FUSED_SELECT` switch) |
| `gather_rows` | `index_select` x2 + scale | the selected samples of a `[B, S, F]` activation copied into the contiguous `[K*S, F]` operand of the reference w.grad GEMM, 128-bit vectors, item-count scale fused into the gradient-output copy (3.1 TB/s on H200 vs ~1 TB/s for `index_select`) |
| `hopper_ops.selected_wgrad` (sm_90) | the `selected_wgrad` above on Hopper | TMA + `wgmma` persistent kernel: A = `go` viewed as `(O, S, B)` (M-major), B = `inp` as `(I, S, B)` (N-major), the producer warp walks K as (selected sample, 64-token tile) with the sample index as the third TMA coordinate, fp32 accumulation, scale in the epilogue, TMA store; within a few percent of cuBLAS on a contiguous copy of the selected rows |
| `hopper_ops.gip_partials` (sm_90) | the `gip_scores` above on Hopper | dual TMA + `wgmma` GEMM (K = O, then K = I, one smem ring, two fp32 accumulators), Hadamard + CTA reduction to one fp32 per 128×128 tile; 1.7× the cuBLAS Gram-matrix path at T = 512 |
| `hopper_ops.compressed_partials` / `compressed_scores` (sm_90) | the `compressed_grad` projection above and `score_select` on Hopper | two kernels: `HopperDualProj` forms both projections (`go P_O`, `inp P_I`) over the flattened tokens in one persistent TMA + `wgmma` launch (64×64×64 tiles, no K split, output rounded once to bf16 like the reference), `HopperOuterScore` forms the per-sample 64×64 outer products on tensor cores (K = the tokens of one sample) and, one CTA per training row, the scores `corr · <c_b, c_val>`; the last CTA to finish selects the sorted top-k. Two launches per layer instead of five; the projection runs at 55–70 % of HBM bandwidth |
| `score_select` (CuTe only) | tile-sum, bf16 cast, val sum, GEMV, correction, `topk`, `sort` | single-CTA epilogue over the projection's fp32 tile partials: validation vector, `corr · <c_b, c_val>` for every training row and the sorted top-k indices, in one launch (≈ 8 launches before). Used by the Layer-Wise and Global compress paths (`drpt.selection.backward.compressed_scores_fused`) on Ampere; on `sm_90` the scores and the selection come out of `HopperOuterScore` instead. `DRPT_FUSED_SELECT=0` keeps only the projection kernel |

Per-layer kernel time on A40 (`benchmark_kernels.py`, Qwen3-1.7B shapes,
`n=8 T=512 m=1`, summed over the 28 blocks; reference → CuTe → Triton):
gip 55 → 53 → 47 ms, pip 144 → 107 → 104 ms, selected w.grad 83 → 58 → 56 ms.
The CuTe kernels are within ~8 % of Triton on the large layers (pip and w.grad
at parity) and 2-3× faster on small ones (≈ 25 µs host overhead per call vs
≈ 120 µs). `act_grad` (`grad_output @ W`) stays on cuBLAS (≈ 78 % of bf16 peak).

Full A40 suite with the CuTe backend (`results/paper/`; the paper tables are written from it by
`SFT/tables/system_efficiency.py`; compressed scoring, m=1, k=n/2; Standard → Layer-Wise →
Global 1P step time, overhead vs Standard in parentheses):

| Model | n=8, T=512 | n=2, T=1024 | n=8, T=512 + ckpt |
|-------|-----------|-------------|--------------------|
| SmolLM2-360M | 260 → 299 (+15%) → 298 (+15%) | 158 → 224 (+42%) → 223 (+41%) | 323 → 371 (+15%) → 369 (+14%) |
| TinyLlama-1.1B | 493 → 512 (+4%) → 513 (+4%) | 300 → 383 (+28%) → 383 (+28%) | 623 → 643 (+3%) → 643 (+3%) |
| Llama-3.2-3B | 1397 → 1326 (−5%) → 1329 (−5%) | 846 → 967 (+14%) → 967 (+14%) | 1689 → 1658 (−2%) → 1656 (−2%) |

For comparison, the PyTorch reference ops give +34% / +18% on
SmolLM2, +10% / +8% on TinyLlama and +3% / +3% on Llama-3.2-3B at n=8, T=512.
Llama-3.2-3B's Standard baseline with `--cublaslt` is 1328 ms (1622 ms with
checkpointing), against which the curated steps are −0.1% / +0.1% (+2% with
checkpointing). Repeated runs of the suite reproduce within 1%.

Exact scoring, same suite (`python SFT/tables/system_efficiency.py --scoring pip` prints the PIP grids to stdout; the paper
tables use `gip`, the default; Layer-Wise overhead vs Standard, n=8 T=512 / n=2 T=1024):

| Model | PIP | GIP |
|-------|-----|-----|
| SmolLM2-360M | +22% / +42% | +18% / +43% |
| TinyLlama-1.1B | +19% / +46% | +8% / +35% |
| Llama-3.2-3B | +13% / +33% | −2% / +21% |

PIP's extra cost over compressed scoring is the full-rank score GEMM
(`2·n·T·O·I` FLOPs per layer, 40 / 93 / 282 ms of `scoring` per step on the
three models at n=8) plus the target-gradient GEMM (14–35 ms per step at m=1).
At `T=512, m=1` GIP needs fewer FLOPs by the factor `V*·N/n` ≈ 1.2× (SmolLM2),
2.7× (TinyLlama) and 4× (Llama-3.2-3B) — the paper's `mT ≲ √(d/L)/2` regime —
which is why GIP lands within 2 % of Standard on Llama-3.2-3B while the two exact
methods are within a few percent of each other on SmolLM2.

On Llama-3.2-3B the curated step is faster than Standard: the fused w.grad
avoids the cuBLAS kernel-selection bug below and only computes the selected half
of the batch. Tests: `python tests/test_fused_kernels.py` (kernels vs fp64, both backends) and
`python tests/test_fused_parity_gpu.py` (fused vs reference gradients through the
training strategies, with and without checkpointing).
`benchmark_kernels.py` times the kernels against the reference ops on synthetic
layer shapes and `profile_step.py` reproduces the per-phase / per-layer kernel
attribution on a real model step.

### Hopper note (H200)

The `mma.sync` CuTe kernels above are an Ampere design (`cp.async` + `ldmatrix` +
`mma.sync`); on H100/H200 they run through the compatibility path and pass every test,
but lose to cuBLAS, which uses `wgmma`. Per-layer kernel time at Qwen3-14B shapes on one
H200 (`benchmark_kernels.py --model-tag qwen3-14b`, `n=8 T=512 m=1 k=4`, ms summed over
the 40 blocks; reference = PyTorch ops on cuBLAS):

| op                          | reference | CuTe (`mma.sync`) | Triton |
|-----------------------------|-----------|-------------------|--------|
| PIP scores                  | 185       | 274               | 201    |
| GIP scores                  | 38        | 52                | 46     |
| selected w.grad (4 of 8)    | 78 (mm)   | 158               | 125    |
| compressed projection 64x64 | 32        | 30                | —      |

The full-batch w.grad (8 samples, one cuBLAS GEMM) is 142 ms, so with the Ampere
kernels the halved w.grad of the curated update saved nothing on Hopper. On `sm_90` the
CuTe backend therefore runs **`drpt/kernels/hopper_ops.py`**: two TMA + `wgmma` kernels
written after NVIDIA's `cute/hopper/kernel/dense_gemm` persistent example (one TMA
producer warp, two consumer warpgroups, 4–7-stage smem ring, persistent tile scheduler,
`setmaxnreg` register split), measured standalone on the four block shapes of Qwen3-14B:

- **`selected_wgrad`** — `go` viewed as `(O, S, B)` (M-major A) and `inp` as `(I, S, B)`
  (N-major B); the producer walks K as (selected sample, 64-token tile), the sample index
  being the third TMA coordinate, so the selected rows are never gathered; the item-count
  scale multiplies the fp32 accumulators before the single rounding; TMA-store epilogue.
  128×256 tile, 2×1 cluster (B-tile multicast), swizzle 4 for problems with ≥ 8 waves of
  tiles (MLP projections, lm_head), 128×128 / 2×1 / 8 below (q/o, k/v), and 128×256 in a 2×2
  cluster (both operands multicast) for problems of 1–1.5 waves of 128×128 tiles with an even
  tile grid (the MLP projections of sub-billion models: 20–25 % faster than the nearly empty
  second wave). At `k=4, T=512`
  (K = 2048) it is 1.00–1.03× a cuBLAS `mm` on a contiguous copy of the selected rows for
  the MLP shapes, 1.13× for q/o and 1.2× for the small k/v projections (1.05× lm_head); at
  K = 4096 (`n=16`, or `T ≥ 1024`) it is faster than cuBLAS (0.90–0.95×). It removes the
  15 ms gather that the reference path needs per step.
- **`gip_partials`** — the two token-pair Gram matrices are never written: one CTA tile
  streams `go_t[b] · go_v[v]ᵀ` (K = O) and then `inp_t[b] · inp_v[v]ᵀ` (K = I) through
  the same smem ring into two fp32 accumulators, multiplies them elementwise and reduces
  to one fp32 per 128×128 tile (`reduce_select` finishes). 1.7× faster than the cuBLAS
  Gram path at `n=8 T=512` (0.36 vs 0.60 ms over the four shapes), 1.3× at `n=16`, parity
  at `T ≥ 2048` (the 128×128 tile is L2-bound), slower on the 152k-row lm_head, which
  therefore keeps the cuBLAS partials (`_gip_partials_cublas`); rel. error vs fp64 ≤ 1e-5
  against 1–3e-3 for the bf16 Gram path.

- **`compressed_partials` / `compressed_scores`** — the Ampere projection kernel is a
  `64 × 64 × K` GEMM per 64-token tile and launches only `B · S/64` CTAs (72 at `n=8, T=512`),
  so it ran at a fraction of the memory bandwidth (31 ms per step, plus 18 ms in the
  tile-summing `score_select`). On `sm_90` two kernels replace the five launches of that
  chain: `HopperDualProj` forms both projections over the flattened tokens in one persistent
  TMA + `wgmma` launch (64×64×64 tiles, the two problems' 64-row tiles fill the GPU without a
  K split, output rounded once to bf16 as the reference's matmul does), and
  `HopperOuterScore` forms the per-sample 64×64 outer products on tensor cores (K = the tokens
  of a sample) — as fp32 compressed gradients for `compressed_grad`, or, one CTA per training
  row, straight into `corr · <c_b, c_val>` with the sorted top-k selected by the last CTA to
  finish (arrival counter, acquire/release at GPU scope). Per layer at `n=8, T=512` on the 8B
  q/o shape: 28 + 7 µs of GPU time against 66 µs before; on SmolLM2's 960-wide layers with
  `n=32`: 21 + 13 µs against 96 µs (the single-CTA `score_select` alone took 51 µs at B = 33).

- **`pip_partials`** — the per-token inner product as one TMA + `wgmma` GEMM
  (`inp[b] · Gᵀ`, K-major operands, 128×256 tiles, persistent scheduler rastered along the
  token tiles so the CTAs sharing a tile of the weight-sized `G` run together) whose fp32
  accumulator tile is dotted with the matching `go` tile in registers; the reference's
  `[B, S, O]` temporary and its casts disappear. PIP stays compute-bound by construction
  (`G · x` for every token is a forward pass' worth of GEMM, 62 TFLOP at 8B and 4k tokens),
  so the kernel runs at cuBLAS speed rather than below it: ~105 ms per step either way.

On the vocabulary-sized lm_head the GIP problem stays on cuBLAS GEMMs plus an fp32
product-reduction (`_gip_partials_cublas`); PIP runs the Hopper kernel at every width. The
`reduce_select` epilogue of pip/gip is unchanged.
`DRPT_FUSED_DISABLE=hopper` keeps the `mma.sync` w.grad, `DRPT_CUTE_SCORING_ON_HOPPER=1`
the `mma.sync` pip/gip kernels; `DRPT_FUSED_DISABLE=<op>[,<op>]` (`gip`, `pip`, `wgrad`,
`proj`, `gather`) still routes single operations to the reference path.

The reference path (`KERNEL_BACKEND=off`) is itself Hopper-aware: `compute_selected_gradients`
runs one cuBLAS `mm` on contiguous operands produced by the **row-gather kernel**
(`GatherRows` in `cute_ops.py`, Triton port in `triton_ops.py`): the selected samples of a
`[B, S, F]` activation are copied into a `[K*S, F]` tensor with one 128-bit vector per
thread and the item-count scale fused into the copy of the gradient output (33 GB of
selected rows in 15 ms at the 14B shapes, 3.1 TB/s; `index_select` needs 32 ms plus 11 ms
for a separate scale). The gather backend is chosen independently of the GEMM backend
(`kernels.gather_backend()`: CuTe, else Triton, else `index_select`).

**Two measurement effects at this scale.** (i) The per-layer timing wrapper that splits
Full-Training's backward into `act_grad` / `w.grad` (a Python autograd Function with four
CUDA-event marks per Linear) costs the 14B step about 4 % on an H200: 1047–1051 ms wrapped
against 1003–1007 ms for the native step on the same GPU and batches. Every method's
result therefore also carries `plain_step_ms`, the same step timed once per iteration with
every patch removed; overheads should be quoted from those numbers (at `n=8, T=512`:
plain Full-Training 1004 ms, plain Layer-Wise / GIP 1094 ms, +9 %). (ii) cuBLAS picks its
kernels per shape: the block GEMMs run at 34.5 µs per token at `M = 4096` rows (Full-Training,
`n=8, T=512`) but at 36.5–38.5 µs per token at `M = 4608` (the merged batch with the
target sequence), 39–40 µs at `M = 8192` and `8704`. The ninth sequence at `n=8, T=512`
therefore costs its 1/8 share of forward, `a.grad` and recompute (≈ 87 ms) plus a
≈ 35 ms tile-shape penalty on the other eight, while at `n = 16` and `n = 32` the merged
and the plain batch sit on the same efficiency plateau (`gemm_shape_probe` in the
benchmark notes; cuBLASLt does not remove the effect).

**What is left is the merged target example.** With the Hopper kernels the curated GPU
work is already below Full-Training's `w.grad` (14B, `n=8, T=512`: selected `w.grad` 89 ms +
scoring 28–29 ms against 162 ms); the overhead is the target sequence's `1/n` share of
forward, `a.grad` and (with checkpointing) recompute, minus that saving. Without
checkpointing there is no recompute and no lost early stop, and the `w.grad` share of the
step doubles: Qwen3-8B, one H200, plain step times, Layer-Wise / compress **+3.1 % at
`n=8, T=512`** (479 ms baseline), **−2.0 % at `n=16, T=512`** (887 ms) and **−5.3 % at
`n=32, T=256`**; exact GIP reads +7.0 % / +1.7 % / −3.3 % on the same steps (its score is a
forward-sized GEMM per candidate, quadratic in the sequence length); the two-pass Global
variant stays at +38–41 % (its second forward/backward). With checkpointing at 14B the
`n=8` / `n=16` configurations read +8.0 % / +4.3 % (plain) or +4.1 % / −0.2 % (wrapped
baseline). Longer sequences at fixed tokens per step make the target share larger (`n = 2`,
`T = 2048`: one sequence in three, +29 % compress / +44 % GIP), more candidates per target
make it smaller. Small models follow the same curve once they leave the launch-bound
regime: at `n = 32, T = 512` the Layer-Wise compress step is +6 % on SmolLM2-360M, −2 % on
TinyLlama-1.1B and −3 % on Llama-3.2-3B (GIP +10 % / +4 % / 0 %), against +33 % / +11 % / +10 %
at `n = 8`. Per-configuration numbers: `results/h200/`.

Small models are a different regime: on SmolLM2-360M the curated backward was
**CPU-bound** with the reference ops (their +34 % there is Python and launch overhead of the
per-layer custom backward, not GPU work). Besides the fused compress and
score-select kernels, the hot path avoids per-layer launches where the math
allows: the item-count scale is a constant per selection size under
`sample_mean` (computed once per step, no index/sum/divide chain per layer),
the `lr` multiply before top-k is skipped (top-k and sign filtering are
scale-invariant), `m = 1` needs no reduction over validation rows, and the
merged-batch correction is cached as an fp32 tensor (a `float()` of a bf16
device scalar per layer used to stall the CPU on the GPU). With these the step
is GPU-bound on all three models.

**What is left is mostly inherent to merged-batch scoring.** With `m = 1`
validation sample in a batch of `n = 8`, the forward, `a.grad` and the
non-linear backward run on 9/8 of the tokens (+12.5 %), while the curated
`w.grad` halves (`k = n/2`). The floor for a curated step is therefore
`Standard × (1 + 0.125·(fwd + a.grad + autograd share) − 0.5·w.grad share) +
scoring`, which is where Llama-3.2-3B already is (the halved `w.grad` outweighs
the extra sample) and within ~5 % of where SmolLM2 and TinyLlama are. Going
below Standard on the small models would need a cheaper validation signal
(e.g. reusing a cached validation gradient across steps in separate-batch mode)
rather than faster kernels.

## Real-Job Step Throughput (alpaca → samsum)

`run_qa_throughput.sh` runs the paper's QA setting — Llama-3.2-1B on
`alpaca → samsum`, `n=8`, `max_seq_length=512` with dynamic padding, `m=1`,
`k=4`, bf16 + FlashAttention-2 — for 130 optimizer steps per combination with
the production launcher's own command (`SFT/train/train.sh --dry-run`), one job
per GPU (`GPUS=0,1,2,3`). ms/step is the slope of the trainer's CUDA-event
`train_wall_time` between the evaluations at steps 26 and 130 (evaluation time
excluded; the first 26 steps absorb kernel compilation). The four 26-step windows
of a run agree with their mean within 6 % on median and 9 % at worst (the
sequence-length mix changes per window); a run with an anomalous window is repeated.
The scoring method is overridden on the command line; `compress`
uses the compressor of the matching `LayerWiseSubset-<ft>` config — 64×64 for
Full / LoRA and 512×512 for MeSO as in the actual experiments (512×512 is outside
the fused compressor kernel's κ ≤ 64, so both backends run the reference path
there). The Global Subset configs of `alpaca_samsum` use exact PIP scoring, the
Layer-Wise configs compressed scoring; every other cell is the override.
`python SFT/benchmark/qa_throughput_table.py` prints this summary (diagnostic; the paper's per-component grids come from
`SFT/tables/qa_timing.py`, which reads the same runs).

| Fine-tuning | Method | Scoring | cute ms/step | off ms/step | vs Full-Training | final loss |
|---|---|---|---|---|---|---|
| Full | Full-Training |  | 301 | — | — | 1.449 |
| Full | Layer-Wise Subset | Compressed | 325 | 334 | cute: +8.1%, off: +11.0% | 1.450/1.448 |
| Full | Layer-Wise Subset | GIP | 333 | 339 | cute: +10.8%, off: +12.7% | 1.449/1.448 |
| Full | Layer-Wise Subset | PIP | 374 | 377 | cute: +24.3%, off: +25.4% | 1.446/1.447 |
| Full | Global Subset (1P) | Compressed | 322 | 332 | cute: +7.1%, off: +10.3% | 1.450/1.450 |
| Full | Global Subset (1P) | GIP | 333 | 338 | cute: +10.7%, off: +12.5% | 1.439/1.441 |
| Full | Global Subset (1P) | PIP | 375 | 376 | cute: +24.6%, off: +25.0% | 1.443/1.446 |
| LoRA | Full-Training |  | 260 | — | — | 1.414 |
| LoRA | Layer-Wise Subset | Compressed | 369 | 376 | cute: +41.8%, off: +44.4% | 1.380/1.385 |
| LoRA | Layer-Wise Subset | GIP | 314 | 341 | cute: +20.5%, off: +30.8% | 1.365/1.371 |
| LoRA | Layer-Wise Subset | PIP | 309 | 355 | cute: +18.7%, off: +36.2% | 1.366/1.366 |
| LoRA | Global Subset (1P) | Compressed | 351 | 356 | cute: +34.8%, off: +36.6% | 1.398/1.401 |
| LoRA | Global Subset (1P) | GIP | 315 | 328 | cute: +20.8%, off: +26.1% | 1.406/1.404 |
| LoRA | Global Subset (1P) | PIP | 315 | 349 | cute: +20.9%, off: +33.8% | 1.405/1.408 |
| MeSO | Full-Training |  | 244 | — | — | 1.445 |
| MeSO | Layer-Wise Subset | Compressed | 276 | 268 | cute: +13.2%, off: +10.2% | 1.424/1.431 |
| MeSO | Layer-Wise Subset | GIP | 282 | 285 | cute: +15.7%, off: +17.1% | 1.437/1.422 |
| MeSO | Layer-Wise Subset | PIP | 328 | 321 | cute: +34.5%, off: +31.8% | 1.430/1.433 |
| MeSO | Global Subset (1P) | Compressed | 283 | 283 | cute: +16.1%, off: +16.3% | 1.445/1.448 |
| MeSO | Global Subset (1P) | GIP | 282 | 280 | cute: +15.7%, off: +15.1% | 1.441/1.443 |
| MeSO | Global Subset (1P) | PIP | 323 | 318 | cute: +32.7%, off: +30.7% | 1.443/1.458 |

- **Full fine-tuning.** Layer-Wise and Global Subset with compressed scoring cost
  +8 % / +7 % over Full-Training (301 ms), GIP +11 %, PIP +25 % for either
  method. Steps are ~300 ms rather than the synthetic 512-token ~500 ms because
  alpaca batches pad only to their longest sequence.
- **Fused vs reference ops.** Under full fine-tuning the kernels save 3–10 ms/step
  (1–3 %) with compressed and GIP scoring and nothing with PIP: at S ≈ 250 the
  PIP GEMM is compute-bound on both paths and the `[n, S, O]` intermediate the
  kernel removes is small. Under LoRA they save 13–46 ms/step (4–13 %) for GIP
  and PIP, where the hooked rank-8 factors leave the reference einsum path
  launch- and bandwidth-bound.
- **LoRA.** Compressed 64×64 scoring is the *most* expensive method (+42 % / +35 %)
  and exact GIP / PIP the cheapest (+19–21 %): the per-sample gradient of a LoRA
  factor is `[8, d_in]`, so exact scoring costs `2NT·8·d_in` per layer while the
  dense first-stage projection costs `NT·64·d_in` — the paper's `κ ≪ d/L`
  premise does not hold for rank-8 adapters.
- **MeSO.** +10–16 % for compressed (512×512) and GIP scoring, +31–35 % for PIP,
  over a 244 ms MeSO baseline.

**Decomposition of real steps.** `PROFILE=1 bash SFT/benchmark/run_qa_throughput.sh`
re-runs every combination for 60 steps and profiles steps 30–39 with
`torch.profiler` (opt-in in the trainer: `DRPT_PROFILE_STEPS=30:40
DRPT_PROFILE_TRACE=<dir>/trace.json`, see `SFT/benchmark/trace_attribution.py`).
Every CUDA kernel is attributed to the phase that launched it: kernels under the
autograd engine are backward, split by the `record_function` labels the profiler
patches into the drpt custom backward (`drpt/score`, `drpt/select` → `scoring`;
`drpt/wgrad`, `drpt/assembly`, MeSO's `drpt/store_update` → `w.grad`; the
un-labelled remainder of the custom Linear backward → `a.grad`) and, for plain
`nn.Linear` layers, by the output shape of the two backward GEMMs (weight-shaped →
`w.grad`); kernels under `Optimizer.step`/`zero_grad` and the `_foreach_` gradient
clipping → `optimizer`; everything else → `forward`. `SFT/tables/qa_timing.py`
combines the GPU-busy ms per phase with the un-profiled wall time of the matching
throughput run (`Host / idle` = wall − Σ busy) into the real-step grids
`qa_timing_grid_{full,lora,meso}.tex` (Full fine-tuning below; `profile.json` per run in
`results/paper/qa_profile/`). These grids are not part of the paper (registry kind `shelved`);
the generator writes them into `Paper/<venue>/Tables/` when run directly.

| Component | Full-Training | LW Compressed | LW GIP | LW PIP | GS Compressed | GS GIP | GS PIP |
|---|---|---|---|---|---|---|---|
| Forward | 61.7 | 67.3 | 67.3 | 67.3 | 67.4 | 67.3 | 67.3 |
| Backward | 120.4 | 132.3 | 141.0 | 182.3 | 132.1 | 140.5 | 181.6 |
| ⤷ a.grad | 34.6 | 38.4 | 38.5 | 38.7 | 38.5 | 38.4 | 38.6 |
| ⤷ scoring | — | 13.3 | 21.9 | 62.7 | 10.5 | 19.1 | 59.9 |
| ⤷ w.grad | 32.8 | 23.6 | 23.6 | 23.7 | 29.2 | 29.2 | 29.3 |
| ⤷ autograd | 53.0 | 57.1 | 57.1 | 57.1 | 53.8 | 53.8 | 53.8 |
| Optimizer (+ grad clipping) | 94.9 | 95.0 | 94.9 | 95.0 | 94.9 | 94.9 | 94.9 |
| Host / idle | 23.7 | 30.4 | 29.9 | 29.1 | 27.7 | 30.2 | 30.9 |
| **Total (wall)** | **301** | **325** (+8.1%) | **333** (+10.8%) | **374** (+24.3%) | **322** (+7.1%) | **333** (+10.7%) | **375** (+24.6%) |

The extra target sample is 9/8 of `Forward` and `a.grad`; curated `w.grad` falls
from 33 to 24 ms (Layer-Wise) and 29 ms (Global Subset, whose post-hoc assembly
adds a few ms of elementwise work); the three scoring backends cost 13 / 22 / 63
ms. The AdamW step with gradient clipping is 95 ms of every step. Under LoRA the
optimizer is 1 ms and `w.grad` 5–6 ms, so the curated overhead is scoring plus
the extra sample in `Forward`/`autograd`, and exact scoring (16–18 ms) beats the
64×64 projection (32–34 ms). Under MeSO with the shared 512×512 compressor,
Layer-Wise's compressed update gradient is produced by the scoring projection, so
its `w.grad` is 0 by construction. The profiler itself inflates the step by
15–120 % (most under LoRA), which is why the components are GPU-busy times and
the totals come from the un-profiled runs.

**Paper tables.** `python SFT/tables/system_efficiency.py` writes `system_overhead`, the 12 `timing_grid*`
tables, `peak_memory` and `score_cost` from `results/paper/{breakdown, breakdown_checkpointing,scoring}` straight
into `Paper/ICLR/Tables/` with exact-GIP curated columns (`--check` diffs instead of writing; `--scoring compress` / `pip`
prints the grids of the other backends to stdout). `python SFT/tables/qa_timing.py` writes the shelved real-step grids
`qa_timing_grid_*` from `results/paper/{qa_profile,qa_throughput}`; `python tables/make_all.py --check` runs every generator
(`tables/README.md`). The Full-Training rows of Llama-3.2-3B can be cross-checked against the `--cublaslt`
baseline JSONs in the same directories (quoted as a comment line in `system_overhead.tex`). `results/paper/`
(JSON, not logs or traces) is exempted from the repository's `**/results/` ignore rule so these numbers travel
with the code; every other `results/` directory stays untracked.
`benchmark_run.py --only-scoring <method>` re-runs just Full-Training and that
method's combos and merges them into an existing JSON.

**cuBLAS heuristic bug (A40, torch 2.6 / cu124).** For the `[3072, 8192]`
weight-gradient GEMM of Llama-3.2-3B `down_proj` with `K ∈ [2048, 4608]`
tokens, cuBLAS picks a kernel that runs at 35 TFLOPS instead of 120 TFLOPS.
This inflates the *Standard* baseline's w.grad by ≈ 4 ms per block (also inside
PyTorch's own `F.linear` backward) and the reference curated w.grad by 2.4 ms per
block; the fused w.grad kernels are unaffected. `--cublaslt`
(`torch.backends.cuda.preferred_blas_library("cublaslt")`) avoids it for all
GEMMs and is within noise on every other shape of the suite — use it to get an
un-inflated Llama-3.2-3B baseline (1405 → 1340 ms).

## Quick Start

```bash
# A40 suite (run_benchmarks.sh launches breakdown + checkpointing + scoring)
bash SFT/benchmark/run_benchmarks.sh

# Slurm submission (A40)
bash SFT/benchmark/slurm/launch_all.sh

# Slurm submission (H200)
bash SFT/benchmark/slurm/launch_h200.sh

# Single method benchmark
python benchmark.py --method layer_wise_subset --model HuggingFaceTB/SmolLM2-360M \
    --batch-size 8 --seq-length 512 --scoring-method compress

# Same with the Triton backend, or with the PyTorch reference ops
python benchmark.py --method layer_wise_subset --model HuggingFaceTB/SmolLM2-360M \
    --batch-size 8 --seq-length 512 --scoring-method gip --kernel-backend triton
python benchmark.py --method layer_wise_subset --model HuggingFaceTB/SmolLM2-360M \
    --batch-size 8 --seq-length 512 --scoring-method gip --no-fused

# Aggregate tables
python aggregate_breakdown.py
```

## File Structure

```
benchmark/
├── benchmark.py              # Core timing engine (per-method)
├── benchmark_run.py          # Runner (all combos for one config)
├── benchmark_scoring.py      # Standalone scoring timer (synthetic tensors)
├── benchmark_kernels.py      # Fused kernels (cute / triton) vs reference ops per layer shape (synthetic tensors)
├── profile_step.py           # torch.profiler step profile with per-phase / per-layer kernel attribution
├── benchmark.sh              # Clean-process wrapper
├── utils.py                  # Config, datasets, helpers
├── aggregate_breakdown.py    # Table generator
├── aggregate_fused_ab.py     # Fused-kernel vs reference A/B table (results/fused_ab/)
├── qa_throughput_table.py    # Markdown summary of results/paper/qa_throughput (paper tables: SFT/tables/{system_efficiency,qa_timing}.py)
├── check_pip_gip_equivalence.py   # PIP == GIP on random factors (the matrix's exact backend is GIP)
├── diag_compression_agreement.py  # exact vs 64..512 compressed Layer-Wise scores on identical batches (configs/alpaca_samsum_diag)
├── run_benchmarks.sh         # A40 launcher (breakdown + scoring)
├── README.md
├── slurm/
│   ├── launch_all.sh         # A40 Slurm submitter (uses configs/*.json)
│   ├── launch_h200.sh        # H200 Slurm submitter (inline config matrix)
│   └── configs/              # Per-model JSON configs (gitignored)
└── results/
    ├── breakdown/                 # A40, no gradient checkpointing
    ├── breakdown_checkpointing/   # A40, with gradient checkpointing
    ├── h200/                      # H200 suite (breakdown, breakdown_checkpointing, scoring; manifest + task.sh of the launcher)
    ├── scoring/                   # Scoring-comparison results
    ├── fused_ab/                  # Fused kernels (cute, triton) vs PyTorch reference (A40)
    └── paper/                     # Full suite with the CuTe backend (breakdown, checkpointing, scoring, qa_throughput, qa_profile)
```
