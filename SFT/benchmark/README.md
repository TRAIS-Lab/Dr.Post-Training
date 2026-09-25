# Timing Benchmark

Per-component runtime analysis for **Standard**, **LayerWiseSubset**,
**GlobalSubset (two-pass)**, and **GlobalSubset (one-pass)** with four
scoring mechanisms (`compress`, `gip`, `pip`, `direct`; `pip` = per-token inner product, `gip` = ghost inner product).

Two suites are maintained for different hardware:

- **A40 suite** — 3 small/mid models from different families, packed
  one-model-per-GPU on a single A40x4 node.
- **H200 suite** — Qwen3 family at larger sizes and longer sequences,
  one-job-per-(model, config) on H200 nodes.

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

Qwen3 family at scale (one GPU per job).

| Model      | hidden | intermediate | L  | heads (q / kv) | head dim | GQA |
|------------|--------|--------------|----|----------------|----------|-----|
| Qwen3-1.7B | 2048   |  6144        | 28 | 16 / 8         | 128      | 2:1 |
| Qwen3-4B   | 2560   |  9728        | 36 | 32 / 8         | 128      | 4:1 |
| Qwen3-8B   | 4096   | 12288        | 36 | 32 / 8         | 128      | 4:1 |

Configs: `(n=8, T=1024)`, `(n=4, T=2048)`, `(n=2, T=4096)`, `(n=16, T=512)`,
each with `m=1`. Per (model, config) → 2 jobs (with/without gradient
checkpointing) → 24 breakdown jobs + 3 scoring jobs = 27 total.

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

Results: `results/scoring/`. `benchmark_scoring.py --gpu N` is applied before
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
| `score_select` (CuTe only) | tile-sum, bf16 cast, val sum, GEMV, correction, `topk`, `sort` | single-CTA epilogue over the projection's fp32 tile partials: validation vector, `corr · <c_b, c_val>` for every training row and the sorted top-k indices, in one launch (~30 µs; ≈ 8 launches before). Used by the Layer-Wise and Global compress paths (`drpt.selection.backward.compressed_scores_fused`); `DRPT_FUSED_SELECT=0` keeps only the projection kernel |

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

Hopper note: the CuTe kernels run on H100/H200 through the Ampere-compatible
path; a wgmma/TMA mainloop (see NVIDIA's `cute/hopper/dense_gemm.py`) is the
natural next step for the H200 suite and was not written here (no Hopper GPU to
validate on).

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
sequence-length mix changes per window); one run in 41 had a single 3× slower
window and was replaced by a repeat (373 / 374 ms in two repeats). The scoring method is overridden on the command line; `compress`
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
    ├── h200/                      # H200 results
    ├── scoring/                   # Scoring-comparison results
    ├── fused_ab/                  # Fused kernels (cute, triton) vs PyTorch reference (A40)
    └── paper/                     # Full suite with the CuTe backend (breakdown, checkpointing, scoring, qa_throughput, qa_profile)
```
