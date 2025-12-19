# Memory and Performance Benchmark

Comprehensive benchmarking suite for comparing memory usage and throughput across different fine-tuning methods on Llama-3.2-1B.

## Overview

This benchmark measures:
- **Peak memory usage** during training
- **Training throughput** (samples/second)
- **Detailed timing breakdown** (forward, backward, optimizer, data selection)
- **Memory growth analysis** across iterations

## Methods Tested

### Full Fine-Tuning Methods
| Index | Method                   | Description                              |
| ----- | ------------------------ | ---------------------------------------- |
| 0     | Full + SGD               | Standard SGD optimizer                   |
| 1     | Full + SGD + GC          | SGD with gradient checkpointing          |
| 2     | Full + SGD-momentum      | SGD with momentum (0.9)                  |
| 3     | Full + SGD-momentum + GC | SGD-momentum with gradient checkpointing |
| 4     | Full + AdamW             | Standard AdamW optimizer                 |
| 5     | Full + AdamW + GC        | AdamW with gradient checkpointing        |
| 6     | Full + MeSO              | Memory-efficient Stateful Optimizer      |
| 7     | Full + MeSO + GC         | MeSO with gradient checkpointing         |
| 8     | Full + GaLore            | GaLore optimizer (rank=128)              |
| 9     | Full + GaLore + GC       | GaLore with gradient checkpointing       |

### LoRA Methods
| Index | Method                   | Description                                       |
| ----- | ------------------------ | ------------------------------------------------- |
| 10    | LoRA + SGD               | LoRA (r=256) with SGD                             |
| 11    | LoRA + SGD + GC          | LoRA with SGD and gradient checkpointing          |
| 12    | LoRA + SGD-momentum      | LoRA with SGD-momentum                            |
| 13    | LoRA + SGD-momentum + GC | LoRA with SGD-momentum and gradient checkpointing |
| 14    | LoRA + AdamW             | LoRA with AdamW                                   |
| 15    | LoRA + AdamW + GC        | LoRA with AdamW and gradient checkpointing        |

### Data Selection Methods (GREATS)
| Index | Method             | Description                                         |
| ----- | ------------------ | --------------------------------------------------- |
| 16    | Full + GREATS      | Full fine-tuning with GREATS data selection         |
| 17    | Full + GREATS + GC | GREATS with gradient checkpointing                  |
| 18    | LoRA + GREATS      | LoRA with GREATS data selection                     |
| 19    | LoRA + GREATS + GC | LoRA + GREATS with gradient checkpointing           |
| 20    | MeSO + GREATS      | MeSO with GREATS (33% speedup from gradient reuse!) |
| 21    | MeSO + GREATS + GC | MeSO + GREATS with gradient checkpointing           |

**Note:** GC = Gradient Checkpointing (reduces memory at cost of ~30% slower training)

## Quick Start

### List Available Methods
```bash
python benchmark.py --list
```

### Run Single Method
```bash
# Test MeSO (method index 6)
python benchmark.py --method 6

# Test MeSO + GREATS (method index 20)
python benchmark.py --method 20
```

### Run All Methods (Sequential)
```bash
# Run all 22 methods sequentially (takes ~2-3 hours on A100)
python benchmark.py
```

### Parallel Execution (Recommended)
For faster results, run methods in parallel across multiple GPUs:

```bash
# GPU 0: Run methods 0-5
CUDA_VISIBLE_DEVICES=0 python benchmark.py --method 0 &
CUDA_VISIBLE_DEVICES=0 python benchmark.py --method 1 &
# ... (repeat for methods 0-5)

# GPU 1: Run methods 6-11
CUDA_VISIBLE_DEVICES=1 python benchmark.py --method 6 &
CUDA_VISIBLE_DEVICES=1 python benchmark.py --method 7 &
# ... (repeat for methods 6-11)
```

Or use the provided script:
```bash
bash benchmark.sh
```

### Aggregate Results
After running individual methods, combine results:
```bash
python benchmark.py --aggregate
```

This generates:
- `benchmark.json`: Aggregated results from all runs
- Console output: Formatted comparison table

## Results Folder Structure

Results are automatically organized by configuration settings:

```
results/
├── fp32/                    # Float32 precision (default)
│   ├── result_00.json
│   ├── result_01.json
│   └── ...
├── bf16/                    # BFloat16 precision
│   ├── result_00.json
│   └── ...
└── bf16_flashattn/          # BFloat16 + Flash Attention
    ├── result_00.json
    └── ...
```

**Subfolder naming:**
- `fp32` - Float32 precision
- `fp16` - Float16 precision
- `bf16` - BFloat16 precision
- `*_flashattn` - Configuration with Flash Attention enabled

**Example workflow:**
```bash
# Run with default settings (fp32)
python benchmark.py --method 6
# Results saved to: results/fp32/result_06.json

# Switch to Flash Attention + bfloat16
# Edit benchmark.py:
#   USE_FLASH_ATTENTION = True
#   MODEL_DTYPE = torch.bfloat16

python benchmark.py --method 6
# Results saved to: results/bf16_flashattn/result_06.json

# Aggregate all results (includes all configurations)
python benchmark.py --aggregate
# Shows comparison table with Config column
```

## Configuration

### Model & Training Settings
Edit constants in `benchmark.py` (lines 75-80):
```python
device = 'cuda'
batch_size = 8              # Training batch size
seq_length = 512            # Sequence length
num_warmup_iterations = 10  # Warmup (not timed)
num_timed_iterations = 10   # Timed iterations for throughput
```

### Flash Attention & Precision
Edit model configuration in `benchmark.py` (lines 82-85):
```python
USE_FLASH_ATTENTION = False  # Set to True to enable Flash Attention 2
MODEL_DTYPE = torch.float32  # Options: torch.float32, torch.bfloat16, torch.float16
# Note: Flash Attention requires bfloat16 or float16 (not float32)
```

**Examples:**
```python
# Default: Standard attention with float32
USE_FLASH_ATTENTION = False
MODEL_DTYPE = torch.float32

# Flash Attention with bfloat16 (recommended for A100/H100)
USE_FLASH_ATTENTION = True
MODEL_DTYPE = torch.bfloat16

# Flash Attention with float16
USE_FLASH_ATTENTION = True
MODEL_DTYPE = torch.float16
```

### Profiling Flags
Edit flags in `benchmark.py` (lines 87-90):
```python
ENABLE_DETAILED_PROFILING = True   # Show millisecond-level timing
ENABLE_DATA_SELECTION = False      # Test with GREATS (deprecated, use --method 16-21)
ENABLE_MEMORY_SNAPSHOT = False     # Capture memory snapshots for visualization
```

Or use command-line flags:
```bash
# Enable memory snapshot recording
python benchmark.py --method 6 --memory-snapshot

# Visualize memory snapshot
python -m torch.utils.viz_memory memory_snapshots/Full_MeSO.pickle
```

## Benchmark Results

After running `--aggregate`, results are displayed with configuration info:

### Single Configuration
```
=======================================================================================================================================
Benchmark Results (Total methods: 22 | Results directory: SFT/benchmark/results)
=======================================================================================================================================
Method                       Peak Mem   Sel(ms)   Fwd(ms)   Bwd(ms)   Opt(ms)   Total(ms)  Throughput
-----------------------------------------------------------------------------------------------------------------------
Full + SGD                   15.62 GB   -         154.7     282.4     14.3      454.4      17.55 samp/s
Full + SGD + GC              9.48 GB    -         152.3     397.0     14.4      567.2      14.07 samp/s
Full + SGD-momentum          17.92 GB   -         155.4     282.5     37.5      478.4      16.64 samp/s
Full + SGD-momentum + GC     11.78 GB   -         152.5     397.6     37.3      590.6      13.52 samp/s
Full + AdamW                 20.22 GB   -         154.5     283.8     91.6      532.9      14.96 samp/s
Full + AdamW + GC            14.08 GB   -         154.2     400.2     91.7      649.2      12.30 samp/s
Full + MeSO                  17.03 GB   -         154.5     325.0     41.1      523.6      15.22 samp/s
Full + MeSO + GC             12.64 GB   -         152.5     439.6     40.6      635.9      12.55 samp/s
Full + GaLore                15.98 GB   -         156.2     285.5     40.7      485.4      16.41 samp/s
Full + GaLore + GC           9.84 GB    -         153.2     403.9     40.6      602.0      13.26 samp/s
LoRA + SGD                   15.01 GB   -         171.6     211.5     1.2       387.4      20.55 samp/s
LoRA + SGD + GC              9.55 GB    -         170.5     346.3     1.4       521.3      15.31 samp/s
LoRA + SGD-momentum          15.11 GB   -         172.0     211.3     2.3       388.7      20.47 samp/s
LoRA + SGD-momentum + GC     9.65 GB    -         172.7     346.1     2.4       524.3      15.22 samp/s
LoRA + AdamW                 15.21 GB   -         172.7     211.2     5.3       392.1      20.26 samp/s
LoRA + AdamW + GC            9.75 GB    -         169.8     346.0     4.9       523.6      15.24 samp/s
Full + GREATS                18.23 GB   712.5     83.8      144.3     20.1      964.4      8.28 samp/s
Full + GREATS + GC           13.78 GB   888.1     83.4      207.7     20.1      1202.4     6.65 samp/s
LoRA + GREATS                18.88 GB   797.6     93.0      158.8     0.3       1052.9     7.58 samp/s
LoRA + GREATS + GC           14.52 GB   1008.1    92.0      235.0     0.4       1338.4     5.97 samp/s
MeSO + GREATS                17.36 GB   715.7     0.0       0.0       42.5      761.1      9.41 samp/s
MeSO + GREATS + GC           12.91 GB   887.7     0.0       0.0       40.0      930.8      7.86 samp/s
=======================================================================================================================
Note: Sel(ms) = Data selection time (GREATS/GradNorm), includes val/train fwd/bwd + dot product

=======================================================================================================================
DETAILED SELECTION BREAKDOWN
=======================================================================================================================
Method                       Val Fwd    Val Bwd    Train Fwd   Train Bwd   Dot Prod    Greedy
-----------------------------------------------------------------------------------------------------------------------
Full + GREATS                84.6ms     141.4ms    155.0ms     324.9ms     5.4ms       0.4ms
Full + GREATS + GC           83.8ms     205.0ms    153.7ms     439.6ms     5.4ms       0.4ms
LoRA + GREATS                94.4ms     158.7ms    172.9ms     359.6ms     10.4ms      0.4ms
LoRA + GREATS + GC           93.1ms     234.9ms    171.3ms     497.5ms     10.7ms      0.4ms
MeSO + GREATS                85.1ms     141.8ms    155.7ms     325.0ms     5.4ms       0.4ms
MeSO + GREATS + GC           82.7ms     204.7ms    153.1ms     439.8ms     4.9ms       0.3ms
=======================================================================================================================
Selection Components:
  Val Fwd/Bwd:    Validation forward/backward pass (small batch)
  Train Fwd/Bwd:  Training forward/backward pass (per-sample gradients)
  Dot Prod:       Gradient similarity computation (layer-by-layer)
  Greedy:         Greedy selection algorithm (choose top samples)
=======================================================================================================================
```

### Multiple Configurations (Comparison)
When aggregating results from multiple configurations (e.g., fp32 vs bf16_flashattn):
```
Configurations found:
  bf16_flashattn: 3 methods
  fp32: 3 methods

==================================================================================================================================
Benchmark Results (Total methods: 6 | Results directory: results)
==================================================================================================================================
Config           Method                       Peak Mem   Fwd(ms)   Bwd(ms)   Opt(ms)   Total(ms)  Throughput
----------------------------------------------------------------------------------------------------------------------------------
bf16_flashattn   Full + AdamW                 18.23 GB   421.3     685.2     108.5     1215.0     6.58 samp/s
bf16_flashattn   Full + MeSO                  14.92 GB   492.1     656.8     41.7      1190.6     6.72 samp/s
bf16_flashattn   Full + MeSO + GC             9.87 GB    493.5     1083.6    38.9      1616.0     4.95 samp/s
fp32             Full + AdamW                 35.13 GB   825.2     1326.8    217.3     2374.6     3.34 samp/s
fp32             Full + MeSO                  27.88 GB   984.4     1313.2    83.4      2386.3     3.35 samp/s
fp32             Full + MeSO + GC             18.50 GB   985.5     2167.1    77.8      3235.1     2.47 samp/s
==================================================================================================================================
```

**Key observations:**
- Flash Attention + bfloat16 reduces memory by ~48% (18.23 GB vs 35.13 GB for Full + AdamW)
- Speed improvement of ~2x (6.58 samp/s vs 3.34 samp/s)
- Config column allows easy comparison across different precision/attention settings
```

## Key Features

### 1. MeSO + GREATS Optimization
Methods 20-21 implement the **gradient reuse optimization**:
- Gradients computed during GREATS selection are reused for optimization
- Avoids redundant forward/backward pass (33% speedup!)
- Matches real `trainer.py` behavior

### 2. Memory Snapshots
Capture detailed memory allocation traces:
```bash
python benchmark.py --method 6 --memory-snapshot
python -m torch.utils.viz_memory memory_snapshots/Full_MeSO.pickle
```

### 3. Detailed Profiling
When `ENABLE_DETAILED_PROFILING = True`:
- Millisecond-level timing for each component
- Memory delta tracking (forward/backward/optimizer)
- Bottleneck analysis (identifies slowest component)
- Data selection breakdown (val fwd/bwd, train fwd/bwd, dot product, greedy)