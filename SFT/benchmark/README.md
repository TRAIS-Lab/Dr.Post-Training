# Memory and Performance Benchmark

Comprehensive benchmarking suite for comparing memory usage and throughput across different fine-tuning methods.

## Methods Tested

### Gradient Streaming Methods

| #   | Method                       | Description                               |
| --- | ---------------------------- | ----------------------------------------- |
| 1a  | `NA_NA_Full`                 | Baseline full fine-tuning (AdamW)         |
| 1b  | `NA_NA_LoRA`                 | Baseline LoRA fine-tuning (AdamW)         |
| 2a  | `Streaming_NA_Full`          | Per-layer selection, full gradients       |
| 2b  | `Streaming_NA_LoRA`          | Per-layer selection, full gradients, LoRA |
| 3a  | `GREATS_NA_Full`             | Global selection, full gradients          |
| 3b  | `GREATS_NA_LoRA`             | Global selection, full gradients, LoRA    |
| 4   | `Streaming_LoGra_Full`       | Per-layer selection + MeSO                |
| 5   | `GREATS_LoGra_Full`          | Global selection + MeSO                   |
| 6a  | `Streaming_LoGra_Full` + 2nd | Per-layer + MeSO + second-order selection |
| 6b  | `GREATS_LoGra_Full` + 2nd    | Global + MeSO + second-order selection    |

### Gradient Checkpointing Variants

All selection methods have `_gc` variants with gradient checkpointing enabled:

| Method                    | Description                     |
| ------------------------- | ------------------------------- |
| `Streaming_NA_Full_gc`    | Streaming + full gradients + GC |
| `Streaming_NA_LoRA_gc`    | Streaming + LoRA + GC           |
| `Streaming_LoGra_Full_gc` | Streaming + MeSO + GC           |
| `GREATS_NA_Full_gc`       | GREATS + full gradients + GC    |
| `GREATS_NA_LoRA_gc`       | GREATS + LoRA + GC              |
| `GREATS_LoGra_Full_gc`    | GREATS + MeSO + GC              |

### External Baselines (for comparison)

We additionally compare full fine-tuning with other optimizers and LoRA variants:
| Method                 | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `Full_sgd`             | Full fine-tuning with SGD                       |
| `Full_sgd_momentum`    | Full fine-tuning with SGD + momentum (0.9)      |
| `Full_sgd_gc`          | SGD with gradient checkpointing                 |
| `Full_sgd_momentum_gc` | SGD-momentum with gradient checkpointing        |
| `LoRA_sgd`             | LoRA with SGD                                   |
| `LoRA_sgd_momentum`    | LoRA with SGD + momentum                        |
| `LoRA_sgd_gc`          | LoRA + SGD with gradient checkpointing          |
| `LoRA_sgd_momentum_gc` | LoRA + SGD-momentum with gradient checkpointing |

## Quick Start

### Running Benchmarks

> [!Note]
> Always use `benchmark.sh` instead of running `benchmark.py` directly. The shell script runs each method in a separate Python process and clears GPU memory between runs, ensuring accurate memory measurements.

```bash
# List available methods
bash benchmark.sh --list

# Run default methods (NA_NA_Full, Streaming_LoGra_Full)
bash benchmark.sh

# Run specific methods
bash benchmark.sh --methods NA_NA_Full Streaming_LoGra_Full GREATS_LoGra_Full

# Run all methods
bash benchmark.sh --all

# Run with custom configuration
bash benchmark.sh --methods GREATS_LoGra_Full --batch-size 32 --num-iterations 20

# Run with second-order selection enabled
bash benchmark.sh --methods GREATS_LoGra_Full --use-second-order
```

<details>
  <summary>Benchmark Configurations</summary>

1. Method Selection
   - `--methods <method1> <method2> ...` - Methods to benchmark (default: `NA_NA_Full Streaming_LoGra_Full`)
   - `--all` - Run all available methods
   - `--list` - List all available methods and exit
2. Model Configuration
   - `--model <name>` - Model name (default: `meta-llama/Llama-3.2-3B`)
   - `--dtype <type>` - Data type: `float32`, `bfloat16`, `float16` (default: `bfloat16`)
   - `--no-flash-attention` - Disable flash attention
3. Training Configuration
   - `--batch-size <n>` - Training batch size (default: 64)
   - `--seq-length <n>` - Sequence length (default: 64)
   - `--val-batch-size <n>` - Validation batch size for selection (default: 1)
4. Benchmark Configuration
   - `--num-warmup <n>` - Warmup iterations (default: 10)
   - `--num-iterations <n>` - Timed iterations (default: 10)
   - `--use-second-order` - Enable second-order selection (greedy, slower)
5. Output
   - `--output <file>` - Save results to JSON file
   - `--results-file <file>` - Append results to JSONL file (for aggregation)
   - `--print-summary <file>` - Print summary from results file and exit
</details>

### Example Output

```
==============================================================================================================
BENCHMARK SUMMARY (Aggregated)
Model: meta-llama/Llama-3.2-1B | Batch: 16 | Val Batch: 1 | Seq: 512 | Dtype: bfloat16
==============================================================================================================
Method                       Peak Mem     Setup Mem    Time/Iter      Throughput       Total Time
--------------------------------------------------------------------------------------------------------------
NA_NA_Full                   34.11 GB     2.30 GB      976.7 ms       16.38 samp/s     19.5 s
NA_NA_LoRA                   28.79 GB     2.51 GB      844.2 ms       18.95 samp/s     16.9 s
NA_NA_Full_gc                21.25 GB     2.30 GB      1190.9 ms      13.44 samp/s     23.8 s
NA_NA_LoRA_gc                17.19 GB     2.51 GB      1125.8 ms      14.21 samp/s     22.5 s
Streaming_NA_Full            35.80 GB     2.31 GB      1228.3 ms      13.03 samp/s     24.6 s
Streaming_NA_LoRA            30.01 GB     2.41 GB      850.8 ms       18.81 samp/s     17.0 s
Streaming_NA_Full_gc         22.14 GB     2.31 GB      1449.0 ms      11.04 samp/s     29.0 s
Streaming_NA_LoRA_gc         17.78 GB     2.41 GB      1111.4 ms      14.40 samp/s     22.2 s
Streaming_LoGra_Full         32.11 GB     3.11 GB      889.1 ms       18.00 samp/s     17.8 s
Streaming_LoGra_Full_gc      18.45 GB     3.11 GB      1110.5 ms      14.41 samp/s     22.2 s
GREATS_NA_Full               36.66 GB     2.31 GB      1413.0 ms      11.32 samp/s     28.3 s
GREATS_NA_LoRA               30.26 GB     2.41 GB      1260.8 ms      12.69 samp/s     25.2 s
GREATS_NA_Full_gc            23.00 GB     2.31 GB      1753.4 ms      9.12 samp/s      35.1 s
GREATS_NA_LoRA_gc            18.04 GB     2.41 GB      1655.7 ms      9.66 samp/s      33.1 s
GREATS_LoGra_Full            32.11 GB     3.11 GB      1291.2 ms      12.39 samp/s     25.8 s
GREATS_LoGra_Full_gc         18.45 GB     3.11 GB      1626.1 ms      9.84 samp/s      32.5 s
Full_sgd                     29.50 GB     2.30 GB      897.7 ms       17.82 samp/s     18.0 s
Full_sgd_momentum            31.80 GB     2.30 GB      909.3 ms       17.60 samp/s     18.2 s
LoRA_sgd                     28.38 GB     2.51 GB      839.0 ms       19.07 samp/s     16.8 s
LoRA_sgd_momentum            28.59 GB     2.51 GB      840.8 ms       19.03 samp/s     16.8 s
Full_sgd_gc                  16.64 GB     2.30 GB      1117.7 ms      14.32 samp/s     22.4 s
Full_sgd_momentum_gc         18.95 GB     2.30 GB      1123.9 ms      14.24 samp/s     22.5 s
LoRA_sgd_gc                  16.78 GB     2.51 GB      1120.1 ms      14.28 samp/s     22.4 s
LoRA_sgd_momentum_gc         16.99 GB     2.51 GB      1120.7 ms      14.28 samp/s     22.4 s
==============================================================================================================
```
