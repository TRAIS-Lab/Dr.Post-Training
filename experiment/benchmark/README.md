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

## Configuration

### Model & Training Settings
Edit constants in `benchmark.py` (lines 72-77):
```python
device = 'cuda'
batch_size = 8              # Training batch size
seq_length = 512            # Sequence length
num_warmup_iterations = 10  # Warmup (not timed)
num_timed_iterations = 10   # Timed iterations for throughput
```

### Profiling Flags
Edit flags in `benchmark.py` (lines 79-83):
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

After running `--aggregate`:
```
=======================================================================================================================
Benchmark Results (Total methods: 22 | Results directory: experiment/benchmark/benchmark_results)
=======================================================================================================================
Method                       Peak Mem   Sel(ms)   Fwd(ms)   Bwd(ms)   Opt(ms)   Total(ms)  Throughput
-----------------------------------------------------------------------------------------------------------------------
Full + SGD                   25.93 GB   -         862.6     1459.0    36.2      2364.2     3.38 samp/s
Full + SGD + GC              13.14 GB   -         654.3     1656.1    27.8      2342.1     3.41 samp/s
Full + SGD-momentum          30.53 GB   -         652.3     1116.6    73.2      1846.1     4.29 samp/s
Full + SGD-momentum + GC     17.74 GB   -         811.8     2070.7    99.2      2987.0     2.68 samp/s
Full + AdamW                 35.13 GB   -         825.2     1326.8    217.3     2374.6     3.34 samp/s
Full + AdamW + GC            24.99 GB   -         656.2     1661.6    207.7     2529.3     3.16 samp/s
Full + MeSO                  27.88 GB   -         984.4     1313.2    83.4      2386.3     3.35 samp/s
Full + MeSO + GC             18.50 GB   -         985.5     2167.1    77.8      3235.1     2.47 samp/s
Full + GaLore                27.84 GB   -         1038.2    1711.5    144.8     2899.9     2.76 samp/s
Full + GaLore + GC           15.24 GB   -         1033.8    2541.2    129.4     3708.7     2.16 samp/s
LoRA + SGD                   24.76 GB   -         1240.0    1309.5    3.2       2559.7     3.12 samp/s
LoRA + SGD + GC              13.18 GB   -         1047.5    2037.0    2.3       3091.7     2.59 samp/s
LoRA + SGD-momentum          24.96 GB   -         1063.1    1129.5    6.3       2205.0     3.62 samp/s
LoRA + SGD-momentum + GC     13.39 GB   -         1055.3    2034.7    5.7       3100.6     2.58 samp/s
LoRA + AdamW                 25.17 GB   -         1060.2    1123.9    14.2      2204.6     3.62 samp/s
LoRA + AdamW + GC            13.59 GB   -         1071.0    2000.2    13.5      3089.9     2.59 samp/s
Full + GREATS                30.28 GB   3490.2    505.5     671.6     62.8      4735.5     1.69 samp/s
Full + GREATS + GC           20.78 GB   4971.0    560.3     1151.1    70.7      6758.9     1.18 samp/s
LoRA + GREATS                30.91 GB   3765.2    558.9     686.7     0.3       5016.1     1.59 samp/s
LoRA + GREATS + GC           22.02 GB   5013.3    547.3     1124.6    0.4       6690.0     1.20 samp/s
MeSO + GREATS                28.55 GB   3472.3    0.0       0.0       81.3      3559.0     1.95 samp/s
MeSO + GREATS + GC           19.04 GB   4646.9    0.0       0.0       80.6      4733.3     1.52 samp/s
=======================================================================================================================
Note: Sel(ms) = Data selection time (GREATS/GradNorm), includes val/train fwd/bwd + dot product

=======================================================================================================================
DETAILED SELECTION BREAKDOWN
=======================================================================================================================
Method                       Val Fwd    Val Bwd    Train Fwd   Train Bwd   Dot Prod    Greedy
-----------------------------------------------------------------------------------------------------------------------
Full + GREATS                521.7ms    643.7ms    995.7ms     1312.6ms    13.5ms      2.0ms
Full + GREATS + GC           513.2ms    1087.0ms   1029.1ms    2321.1ms    14.0ms      6.2ms
LoRA + GREATS                556.0ms    692.6ms    1074.6ms    1407.8ms    29.9ms      2.9ms
LoRA + GREATS + GC           516.6ms    1103.7ms   1077.9ms    2278.4ms    33.4ms      3.0ms
MeSO + GREATS                512.8ms    663.4ms    969.5ms     1302.5ms    14.7ms      3.1ms
MeSO + GREATS + GC           496.4ms    1044.4ms   975.3ms     2107.4ms    14.0ms      3.9ms
=======================================================================================================================
Selection Components:
  Val Fwd/Bwd:    Validation forward/backward pass (small batch)
  Train Fwd/Bwd:  Training forward/backward pass (per-sample gradients)
  Dot Prod:       Gradient similarity computation (layer-by-layer)
  Greedy:         Greedy selection algorithm (choose top samples)
=======================================================================================================================
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