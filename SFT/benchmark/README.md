# Timing Benchmark

Per-component runtime analysis for **Standard**, **LayerWiseSubset**,
**GlobalSubset (two-pass)**, and **GlobalSubset (one-pass)** with four
scoring mechanisms (`compress`, `full_ghost`, `reduced_ghost`, `direct`).

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

- **m-sweep** (T=512, m=1..16): full_ghost vs reduced_ghost crossover
- **T-sweep** (m=1, T=256..8192): reduced_ghost vs direct crossover

```bash
bash SFT/benchmark/run_benchmarks.sh scoring
```

Results: `results/scoring/`.

## Scoring Methods

| Method | FLOPs (score-specific) | Memory | Wins when |
|--------|----------------------|--------|-----------|
| **compress** | O(NTκ) | O(nκ) | Always cheapest (approximate) |
| **full_ghost** | O(nmT²√(d/L)) | O(nmT²) | Small T, small m |
| **reduced_ghost** | O(nT·d/L) | O(nT√(d/L)) | Small T, large m |
| **direct** | O(n·d/L) | O(n·d/L) | Large T |

### Crossover Thresholds

**reduced_ghost vs full_ghost** — crossover at `V* = Σ(O·I) / (T · Σ(O+I))`:

| Model            | V* at T=512 | V* at T=1024 |
|------------------|-------------|--------------|
| SmolLM2-360M     | 1.1         | 0.6          |
| TinyLlama-1.1B   | 2.4         | 1.2          |
| Llama-3.2-3B     | 3.6         | 1.8          |
| Qwen3-1.7B       | 2.5         | 1.3          |
| Qwen3-4B         | 3.4         | 1.7          |
| Qwen3-8B         | 5.0         | 2.5          |

**reduced_ghost vs direct** — crossover at `T* = Σ(O·I) / Σ(√(O·I))`:

| Model            | T*    |
|------------------|-------|
| SmolLM2-360M     | 1271  |
| TinyLlama-1.1B   | 2799  |
| Llama-3.2-3B     | 4069  |
| Qwen3-1.7B       | 2854  |
| Qwen3-4B         | 4088  |
| Qwen3-8B         | 5747  |

## Quick Start

```bash
# A40 suite (run_benchmarks.sh launches breakdown + checkpointing + scoring)
bash SFT/benchmark/run_benchmarks.sh

# Slurm submission (A40)
bash SFT/benchmark/slurm/launch_all.sh

# Slurm submission (H200)
bash SFT/benchmark/slurm/launch_h200.sh

# Single method benchmark
python benchmark.py --method layer-wise-subset --model HuggingFaceTB/SmolLM2-360M \
    --batch-size 8 --seq-length 512 --scoring-method compress

# Aggregate tables
python aggregate_breakdown.py
```

## File Structure

```
benchmark/
├── benchmark.py              # Core timing engine (per-method)
├── benchmark_run.py          # Runner (all combos for one config)
├── benchmark_scoring.py      # Standalone scoring timer (synthetic tensors)
├── benchmark.sh              # Clean-process wrapper
├── utils.py                  # Config, datasets, helpers
├── aggregate_breakdown.py    # Table generator
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
    └── scoring/                   # Scoring-comparison results
```
