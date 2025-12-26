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
4. Dataset Configuration
   - `--dataset <name>` - Dataset for benchmarking (default: `alpaca`)
     - `dummy` - Synthetic repeated sentence (fast, for debugging)
     - `alpaca` - Stanford Alpaca instruction dataset
     - `gsm8k` - Grade school math problems
     - `dolly` - Databricks Dolly 15k
     - `openhermes` - OpenHermes 2.5 conversations
5. Benchmark Configuration
   - `--num-warmup <n>` - Warmup iterations (default: 10)
   - `--num-iterations <n>` - Timed iterations (default: 10)
   - `--use-second-order` - Enable second-order selection (greedy, slower)
6. Output
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
NA_NA_Full                   34.11 GB     2.30 GB      981.0 ms       16.31 samp/s     19.6 s
NA_NA_LoRA                   27.90 GB     2.32 GB      828.0 ms       19.32 samp/s     16.6 s
NA_NA_Full_gc                21.25 GB     2.30 GB      1195.8 ms      13.38 samp/s     23.9 s
NA_NA_LoRA_gc                16.62 GB     2.32 GB      1110.0 ms      14.41 samp/s     22.2 s
Streaming_NA_Full            33.45 GB     2.31 GB      1074.6 ms      14.89 samp/s     21.5 s
Streaming_NA_LoRA            27.12 GB     2.32 GB      873.6 ms       18.31 samp/s     17.5 s
Streaming_NA_Full_gc         20.06 GB     2.31 GB      1295.7 ms      12.35 samp/s     25.9 s
Streaming_NA_LoRA_gc         15.41 GB     2.32 GB      1118.3 ms      14.31 samp/s     22.4 s
Streaming_LoGra_Full         29.76 GB     3.11 GB      903.3 ms       17.71 samp/s     18.1 s
Streaming_LoGra_Full_gc      16.37 GB     3.11 GB      1128.2 ms      14.18 samp/s     22.6 s
GREATS_NA_Full               30.69 GB     2.31 GB      1333.8 ms      12.00 samp/s     26.7 s
GREATS_NA_LoRA               27.13 GB     2.32 GB      1218.9 ms      13.13 samp/s     24.4 s
GREATS_NA_Full_gc            17.30 GB     2.31 GB      1667.2 ms      9.60 samp/s      33.3 s
GREATS_NA_LoRA_gc            15.42 GB     2.32 GB      1621.4 ms      9.87 samp/s      32.4 s
GREATS_LoGra_Full            29.76 GB     3.11 GB      1296.7 ms      12.34 samp/s     25.9 s
GREATS_LoGra_Full_gc         16.37 GB     3.11 GB      1634.7 ms      9.79 samp/s      32.7 s
Full_sgd                     29.50 GB     2.30 GB      900.4 ms       17.77 samp/s     18.0 s
Full_sgd_momentum            31.80 GB     2.30 GB      909.5 ms       17.59 samp/s     18.2 s
LoRA_sgd                     27.87 GB     2.32 GB      828.4 ms       19.31 samp/s     16.6 s
LoRA_sgd_momentum            27.89 GB     2.32 GB      828.0 ms       19.32 samp/s     16.6 s
Full_sgd_gc                  16.64 GB     2.30 GB      1114.9 ms      14.35 samp/s     22.3 s
Full_sgd_momentum_gc         18.95 GB     2.30 GB      1125.0 ms      14.22 samp/s     22.5 s
LoRA_sgd_gc                  16.59 GB     2.32 GB      1102.7 ms      14.51 samp/s     22.1 s
LoRA_sgd_momentum_gc         16.61 GB     2.32 GB      1107.9 ms      14.44 samp/s     22.2 s
==============================================================================================================
```
