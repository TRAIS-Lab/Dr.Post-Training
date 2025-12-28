# Memory and Performance Benchmark

Comprehensive benchmarking suite for comparing memory usage and throughput across different fine-tuning methods.

## Methods Tested

### Gradient Streaming Methods

| #   | Experiment                   | Description                               |
| --- | ---------------------------- | ----------------------------------------- |
| 1a  | `NA-NA-Full`                 | Baseline full fine-tuning (AdamW)         |
| 1b  | `NA-NA-LoRA`                 | Baseline LoRA fine-tuning (AdamW)         |
| 2a  | `Streaming-NA-Full`          | Per-layer selection, full gradients       |
| 2b  | `Streaming-NA-LoRA`          | Per-layer selection, full gradients, LoRA |
| 3a  | `GREATS-NA-Full`             | Global selection, full gradients          |
| 3b  | `GREATS-NA-LoRA`             | Global selection, full gradients, LoRA    |
| 4   | `Streaming-LoGra-Full`       | Per-layer selection + MeSO                |
| 5   | `GREATS-LoGra-Full`          | Global selection + MeSO                   |
| 6a  | `Streaming-LoGra-Full` + 2nd | Per-layer + MeSO + second-order selection |
| 6b  | `GREATS-LoGra-Full` + 2nd    | Global + MeSO + second-order selection    |

### Gradient Checkpointing Variants

All selection methods have `_gc` variants with gradient checkpointing enabled (use `--gc` flag):

| Method                      | Description                     |
| --------------------------- | ------------------------------- |
| `Streaming-NA-Full` + gc    | Streaming + full gradients + GC |
| `Streaming-NA-LoRA` + gc    | Streaming + LoRA + GC           |
| `Streaming-LoGra-Full` + gc | Streaming + MeSO + GC           |
| `GREATS-NA-Full` + gc       | GREATS + full gradients + GC    |
| `GREATS-NA-LoRA` + gc       | GREATS + LoRA + GC              |
| `GREATS-LoGra-Full` + gc    | GREATS + MeSO + GC              |

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

# Run default experiments (NA-NA-Full, Streaming-LoGra-Full)
bash benchmark.sh

# Run specific methods (comma-separated)
bash benchmark.sh --methods "NA-NA-Full,Streaming-LoGra-Full,GREATS-LoGra-Full"

# Run all methods
bash benchmark.sh --methods all

# Run by category
bash benchmark.sh --methods baseline      # NA-NA-Full, NA-NA-LoRA
bash benchmark.sh --methods streaming     # All Streaming-* variants
bash benchmark.sh --methods compression   # *-LoGra-* variants

# Run with custom configuration
bash benchmark.sh --methods GREATS-LoGra-Full --batch-size 32 --num-iterations 20

# Run with second-order selection enabled
bash benchmark.sh --methods GREATS-LoGra-Full --use-second-order

# Dry run - preview commands without executing
bash benchmark.sh --methods all --dry-run
```

<details>
  <summary>Benchmark Configurations</summary>

1. Method Selection
   - `--methods <list>` - Experiments to benchmark (comma-separated or category)
   - `--list` - List all available methods and exit
   - `--dry-run` - Print commands without executing

   Available categories: `all`, `baseline`, `streaming`, `greats`, `full`, `lora`, `compression`, `no-compression`
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
Model: meta-llama/Llama-3.2-1B | Dataset: alpaca | Batch: 16 | Val Batch: 1 | Seq: 512 | Dtype: bfloat16 | Val Strategy: merged
==============================================================================================================
Method                       Peak Mem     Setup Mem    Time/Iter      Throughput       Total Time
--------------------------------------------------------------------------------------------------------------
NA_NA_Full                   33.04 GB     2.30 GB      481.9 ms       33.20 samp/s     48.2 s
NA_NA_LoRA                   27.59 GB     2.33 GB      384.0 ms       41.67 samp/s     38.4 s
Streaming_NA_Full            32.34 GB     2.31 GB      591.9 ms       27.03 samp/s     59.2 s
Streaming_NA_LoRA            26.76 GB     2.32 GB      388.3 ms       41.20 samp/s     38.8 s
GREATS_NA_Full               29.57 GB     2.31 GB      696.0 ms       22.99 samp/s     69.6 s
GREATS_NA_LoRA               26.77 GB     2.32 GB      583.9 ms       27.40 samp/s     58.4 s
Streaming_LoGra_Full         28.64 GB     3.11 GB      430.8 ms       37.14 samp/s     43.1 s
GREATS_LoGra_Full            28.64 GB     3.11 GB      612.7 ms       26.12 samp/s     61.3 s
==============================================================================================================
```
