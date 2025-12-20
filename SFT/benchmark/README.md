# Memory and Performance Benchmark

Comprehensive benchmarking suite for comparing memory usage and throughput across different fine-tuning methods.

## Overview

This benchmark measures:
- **Peak memory usage** during training
- **Training throughput** (samples/second)
- **Average time per iteration** (milliseconds)

## Naming Convention

Methods follow the pattern: `{selection}_{compression}_{training}`

| Component     | Options                     | Description                                   |
| ------------- | --------------------------- | --------------------------------------------- |
| `selection`   | `NA`, `Streaming`, `GREATS` | Data selection method                         |
| `compression` | `NA`, `LoGra`               | Gradient compression (implies MeSO optimizer) |
| `training`    | `full`, `lora`              | Full fine-tuning or LoRA                      |

> **Note:** GraSS compression is also available but LoGra is used in default experiments.

## Methods Tested

### Gradient Streaming Methods (10 configurations)

| #   | Method                       | Description                               |
| --- | ---------------------------- | ----------------------------------------- |
| 1a  | `NA_NA_full`                 | Baseline full fine-tuning (AdamW)         |
| 1b  | `NA_NA_lora`                 | Baseline LoRA fine-tuning (AdamW)         |
| 2a  | `Streaming_NA_full`          | Per-layer selection, full gradients       |
| 2b  | `Streaming_NA_lora`          | Per-layer selection, full gradients, LoRA |
| 3a  | `GREATS_NA_full`             | Global selection, full gradients          |
| 3b  | `GREATS_NA_lora`             | Global selection, full gradients, LoRA    |
| 4   | `Streaming_LoGra_full`       | Per-layer selection + MeSO                |
| 5   | `GREATS_LoGra_full`          | Global selection + MeSO                   |
| 6a  | `Streaming_LoGra_full` + 2nd | Per-layer + MeSO + second-order selection |
| 6b  | `GREATS_LoGra_full` + 2nd    | Global + MeSO + second-order selection    |

**Key design principles:**
- **LoRA doesn't need compression** - already low-rank
- **Compression controls both scoring AND optimizer** - with compression uses MeSO, without uses AdamW

### Gradient Checkpointing Variants

All selection methods have `_gc` variants with gradient checkpointing enabled:

| Method                    | Description                     |
| ------------------------- | ------------------------------- |
| `Streaming_NA_full_gc`    | Streaming + full gradients + GC |
| `Streaming_NA_lora_gc`    | Streaming + LoRA + GC           |
| `Streaming_LoGra_full_gc` | Streaming + MeSO + GC           |
| `GREATS_NA_full_gc`       | GREATS + full gradients + GC    |
| `GREATS_NA_lora_gc`       | GREATS + LoRA + GC              |
| `GREATS_LoGra_full_gc`    | GREATS + MeSO + GC              |

**Note:** GC = Gradient Checkpointing (reduces memory at cost of ~30% slower training)

### External Baselines (for comparison)

#### Full Fine-Tuning
| Method                 | Description                                |
| ---------------------- | ------------------------------------------ |
| `full_sgd`             | Full fine-tuning with SGD                  |
| `full_sgd_momentum`    | Full fine-tuning with SGD + momentum (0.9) |
| `full_sgd_gc`          | SGD with gradient checkpointing            |
| `full_sgd_momentum_gc` | SGD-momentum with gradient checkpointing   |

#### LoRA
| Method                 | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `lora_sgd`             | LoRA with SGD                                   |
| `lora_sgd_momentum`    | LoRA with SGD + momentum                        |
| `lora_sgd_gc`          | LoRA + SGD with gradient checkpointing          |
| `lora_sgd_momentum_gc` | LoRA + SGD-momentum with gradient checkpointing |

## Quick Start

### Using benchmark.sh (Recommended)

**IMPORTANT:** Always use `benchmark.sh` instead of running `benchmark.py` directly. The shell script runs each method in a separate Python process and clears GPU memory between runs, ensuring accurate memory measurements.

```bash
# List available methods
bash benchmark.sh --list

# Run default methods (NA_NA_full, Streaming_LoGra_full)
bash benchmark.sh

# Run specific methods
bash benchmark.sh --methods NA_NA_full Streaming_LoGra_full GREATS_LoGra_full

# Run all methods
bash benchmark.sh --all

# Run with custom configuration
bash benchmark.sh --methods GREATS_LoGra_full --batch-size 32 --num-iterations 20

# Run with second-order selection enabled
bash benchmark.sh --methods GREATS_LoGra_full --use-second-order
```

### Why Use benchmark.sh?

Running multiple methods in a single Python process can lead to:
- **Memory not being fully released** between runs
- **Inaccurate peak memory measurements** due to cached allocations
- **Inconsistent timing** due to JIT compilation state

The `benchmark.sh` script solves these issues by:
1. Running each method in a fresh Python process
2. Explicitly clearing GPU memory between runs
3. Aggregating results into a single summary table

### Direct Python Usage (Single Method Only)

For running a single method (where memory cleanup is not an issue):

```bash
python benchmark.py --methods NA_NA_full --output results.json
```

## Command-Line Arguments

### Method Selection
- `--methods <method1> <method2> ...` - Methods to benchmark (default: `NA_NA_full Streaming_LoGra_full`)
- `--all` - Run all available methods
- `--list` - List all available methods and exit

### Model Configuration
- `--model <name>` - Model name (default: `meta-llama/Llama-3.2-3B`)
- `--dtype <type>` - Data type: `float32`, `bfloat16`, `float16` (default: `bfloat16`)
- `--no-flash-attention` - Disable flash attention

### Training Configuration
- `--batch-size <n>` - Training batch size (default: 64)
- `--seq-length <n>` - Sequence length (default: 64)
- `--val-batch-size <n>` - Validation batch size for selection (default: 1)

### Benchmark Configuration
- `--num-warmup <n>` - Warmup iterations (default: 10)
- `--num-iterations <n>` - Timed iterations (default: 10)
- `--use-second-order` - Enable second-order selection (greedy, slower)

### Output
- `--output <file>` - Save results to JSON file
- `--results-file <file>` - Append results to JSONL file (for aggregation)
- `--print-summary <file>` - Print summary from results file and exit

## Example Output

```
==============================================================================================================
BENCHMARK SUMMARY (Aggregated)
Model: meta-llama/Llama-3.2-3B | Batch: 4 | Val Batch: 1 | Seq: 512 | Dtype: bfloat16
==============================================================================================================
Method                       Peak Mem     Setup Mem    Time/Iter      Throughput       Total Time
--------------------------------------------------------------------------------------------------------------
NA_NA_full                   30.65 GB     5.98 GB      842.0 ms       4.75 samp/s      8.4 s
NA_NA_lora                   18.07 GB     6.53 GB      553.6 ms       7.23 samp/s      5.5 s
NA_NA_full_gc                30.43 GB     5.98 GB      1010.6 ms      3.96 samp/s      10.1 s
NA_NA_lora_gc                11.42 GB     6.53 GB      763.4 ms       5.24 samp/s      7.6 s
Streaming_NA_full            31.24 GB     5.99 GB      1057.5 ms      3.78 samp/s      10.6 s
Streaming_NA_lora            19.56 GB     6.28 GB      619.3 ms       6.46 samp/s      6.2 s
Streaming_NA_full_gc         30.55 GB     5.99 GB      1240.1 ms      3.23 samp/s      12.4 s
Streaming_NA_lora_gc         11.56 GB     6.28 GB      827.4 ms       4.83 samp/s      8.3 s
Streaming_LoGra_full         21.16 GB     7.58 GB      677.9 ms       5.90 samp/s      6.8 s
Streaming_LoGra_full_gc      12.63 GB     7.58 GB      861.0 ms       4.65 samp/s      8.6 s
GREATS_NA_full               33.06 GB     6.00 GB      1174.1 ms      3.41 samp/s      11.7 s
GREATS_NA_lora               20.09 GB     6.28 GB      899.4 ms       4.45 samp/s      9.0 s
GREATS_NA_full_gc            31.96 GB     6.00 GB      1440.7 ms      2.78 samp/s      14.4 s
GREATS_NA_lora_gc            12.13 GB     6.28 GB      1210.2 ms      3.31 samp/s      12.1 s
GREATS_LoGra_full            21.16 GB     7.58 GB      935.7 ms       4.28 samp/s      9.4 s
GREATS_LoGra_full_gc         12.63 GB     7.58 GB      1211.2 ms      3.30 samp/s      12.1 s
full_sgd                     16.58 GB     5.98 GB      648.8 ms       6.17 samp/s      6.5 s
full_sgd_momentum            22.57 GB     5.98 GB      700.5 ms       5.71 samp/s      7.0 s
lora_sgd                     16.96 GB     6.53 GB      537.5 ms       7.44 samp/s      5.4 s
lora_sgd_momentum            17.51 GB     6.53 GB      542.7 ms       7.37 samp/s      5.4 s
full_sgd_gc                  13.94 GB     5.98 GB      813.3 ms       4.92 samp/s      8.1 s
full_sgd_momentum_gc         19.93 GB     5.98 GB      868.6 ms       4.61 samp/s      8.7 s
lora_sgd_gc                  10.32 GB     6.53 GB      746.6 ms       5.36 samp/s      7.5 s
lora_sgd_momentum_gc         10.87 GB     6.53 GB      753.3 ms       5.31 samp/s      7.5 s
```

## Key Features

### 1. MeSO + Selection Optimization
When using compression (`LoGra`), the MeSO optimizer reuses compressed gradients:
- Gradients computed during selection are reused for optimization
- Avoids redundant forward/backward pass
- Memory-efficient stateful optimization

### 2. Selection Methods
- **Streaming**: Per-layer selection (single-pass, lower overhead)
- **GREATS**: Global selection (two-pass, better selection quality)
- **Second-order**: Greedy selection with pairwise similarity (slower but more accurate)

### 3. Comparison with Baselines
External baselines (SGD variants) are included for comprehensive comparison:
- Memory usage across different optimizers
- Throughput comparison with/without gradient checkpointing
- LoRA vs full fine-tuning trade-offs
