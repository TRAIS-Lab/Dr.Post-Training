# Timing Benchmark

Per-component runtime analysis for **Standard**, **LayerWiseSubset**, **GlobalSubset (two-pass)**, and **GlobalSubset (one-pass)** across three model sizes, with four scoring mechanisms.

## Models

| Model | hidden | intermediate | Layers | Params |
|-------|--------|-------------|--------|--------|
| Qwen3-0.6B | 1024 | 3072 | 28 | 0.6B |
| Qwen3-1.7B | 2048 | 6144 | 28 | 1.7B |
| Llama-3.2-3B | 3072 | 8192 | 28 | 3.2B |

## Benchmarks

### Benchmark 1: Per-Component Breakdown

End-to-end training step timing with per-component decomposition for all 4 selection methods. Same `(n, T, m)` across all models for fair comparison.

**Configs**: `n=8 T=512 m=1` and `n=2 T=1024 m=1`

```bash
bash SFT/benchmark/run_benchmarks.sh breakdown
```

Results: `results/breakdown/`

### Benchmark 2: Scoring Comparison

Standalone scoring function timing with synthetic tensors. No model loading — tests all scoring regimes at scale.

- **m-sweep** (T=512, m=1..16): full_ghost vs reduced_ghost crossover
- **T-sweep** (m=1, T=256..8192): reduced_ghost vs direct crossover

```bash
bash SFT/benchmark/run_benchmarks.sh scoring
```

Results: `results/scoring/`

## Scoring Methods

| Method | FLOPs (score-specific) | Memory | Wins when |
|--------|----------------------|--------|-----------|
| **compress** | O(NTκ) | O(nκ) | Always cheapest (approximate) |
| **full_ghost** | O(nmT²√(d/L)) | O(nmT²) | Small T, small m |
| **reduced_ghost** | O(nT·d/L) | O(nT√(d/L)) | Small T, large m |
| **direct** | O(n·d/L) | O(n·d/L) | Large T |

### Crossover Thresholds

**reduced_ghost vs full_ghost** — crossover at `V* = Σ(O·I) / (T · Σ(O+I))`:

| Model | V* at T=512 |
|-------|------------|
| Qwen3-0.6B | 1.4 |
| Qwen3-1.7B | 2.5 |
| Llama-3.2-3B | 3.6 |

**reduced_ghost vs direct** — crossover at `T* = Σ(O·I) / Σ(√(O·I))`:

| Model | T* |
|-------|-----|
| Qwen3-0.6B | 1532 |
| Qwen3-1.7B | 2854 |
| Llama-3.2-3B | 4069 |

## Quick Start

```bash
# Run both benchmarks (3 GPUs)
bash SFT/benchmark/run_benchmarks.sh

# Run on Slurm cluster
bash SFT/benchmark/slurm/launch_all.sh

# Single method benchmark
python benchmark.py --method layer-wise-subset --model Qwen/Qwen3-0.6B \
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
├── run_benchmarks.sh         # Main launcher (breakdown + scoring)
├── README.md
├── slurm/
│   ├── launch_all.sh         # Slurm job submitter
│   └── configs/              # Per-model JSON configs
└── results/
    ├── breakdown/            # Benchmark 1 results + tables
    └── scoring/              # Benchmark 2 results + tables
```
