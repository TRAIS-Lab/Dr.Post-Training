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

## Output

### Per-Iteration Output
```
--- Iteration 11/20 (Timed 1/10) ---
  Forward:   Current =  5.234 GB, Max =  6.123 GB (peak: +0.889 GB, delta: +0.234 GB, time: 245.3ms, model: 240.1ms, loss: 5.2ms)
  Backward:  Current =  5.456 GB, Max =  7.890 GB (peak: +1.767 GB, delta: +0.222 GB, time: 312.4ms, compute: 310.2ms)
  Optimizer: Current =  5.234 GB, Max =  8.012 GB (peak: +0.122 GB, delta: -0.222 GB, time: 89.7ms, step: 85.3ms, zero_grad: 4.4ms)
```

### Final Summary
```
================================================================================
Full + MeSO - FINAL SUMMARY
================================================================================
Absolute maximum memory EVER reached: 8.012 GB
Current memory at end:                5.234 GB
Peak occurred during:                 BACKWARD pass

Memory growth during timed iterations (iter 1 to 10):
  Iter 1 max:  7.998 GB
  Iter 10 max: 8.012 GB
  Growth:      +0.014 GB
  Status:      ✓ STABLE

Timing (average per iteration):
  Setup time:      2.345s
  Forward:         0.245s (245.3ms)
  Backward:        0.312s (312.4ms)
  Optimizer:       0.090s (89.7ms)

================================================================================
Total/iter:     0.647s (647.4ms)
Total (10 timed iters): 6.474s
Throughput:     12.36 samples/s
================================================================================
```

### Aggregated Results
After running `--aggregate`:
```
===============================================================================
Benchmark Results (Total methods: 22 | Results directory: benchmark_results)
===============================================================================
Method                       Peak Mem   Fwd(ms)   Bwd(ms)   Opt(ms)   Total(ms)  Throughput
-------------------------------------------------------------------------------------------
Full + SGD                   15.234 GB  245.3     512.4     89.7      847.4      9.44 samp/s
Full + MeSO                  8.012 GB   245.3     312.4     89.7      647.4      12.36 samp/s
MeSO + GREATS                8.145 GB   0.0       0.0       89.7      789.8      16.46 samp/s
...
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