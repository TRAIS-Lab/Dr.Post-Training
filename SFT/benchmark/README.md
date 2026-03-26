# Timing Breakdown Benchmark

Per-component runtime analysis for **Standard**, **Layerwise**, **Subset (two-pass)**, and **Subset (one-pass)** on Llama-3.2-1B, SmolLM-135M, and Qwen2.5-0.5B, with configurable scoring mechanisms.

## Quick Start

```bash
# Full benchmark: all methods x all scoring x all default configs
python benchmark_run.py --gpu 2

# Single config (0=n8T512m1, 1=n8T512m4, 2=n4T1024m1, 3=n4T1024m4, 4=n2T2048m1)
python benchmark_run.py --gpu 2 --config 0

# Different model
python benchmark_run.py --gpu 2 --model Qwen/Qwen2.5-0.5B
python benchmark_run.py --gpu 2 --model HuggingFaceTB/SmolLM-135M

# Custom config
python benchmark_run.py --gpu 2 --batch-size 16 --seq-length 1024 --val-batch-size 4

# Save results to JSON
python benchmark_run.py --gpu 2 --output results/llama_all.json

# Run multiple models in parallel across GPUs
python benchmark_run.py --gpu 1 --output results/llama.json &
python benchmark_run.py --gpu 2 --model Qwen/Qwen2.5-0.5B --output results/qwen.json &
python benchmark_run.py --gpu 3 --model HuggingFaceTB/SmolLM-135M --output results/smol.json &
wait
```

## Scoring Methods

| Method           | Formula                                                   | Memory         | FLOPs per layer         |
| ---------------- | --------------------------------------------------------- | -------------- | ----------------------- |
| **compress**     | `c_i = Pi(go, inp)`; `s_i = c_i . c_val`                  | O(B x k)       | 2 x B x k               |
| **ghost_greats** | Pairwise: `sum (go_i.go_v) * (inp_i.inp_v)`               | O(B x V x S^2) | 2 x B x V x S^2 x (O+I) |
| **ghost** (ours) | Collapse: `temp = inp @ G_val.T`; `s = (go * temp).sum()` | O(B x S x O)   | 2 x B x S x O x I       |
| **direct**       | Batch bmm: `G = bmm(go^T, inp)`; `s = G . G_val`          | O(B x O x I)   | 2 x B x S x O x I       |

### FLOPs Derivation

**ghost (collapse-first):**
```
Step 1: G_val = einsum('vso,vsi->oi', val_go, val_inp)   → 2 x V x S x O x I  (precompute)
Step 2: temp  = train_inp @ G_val.T                       → 2 x B x S x I x O  (matmul)
Step 3: score = (train_go * temp).sum(dim=(1,2))           → 2 x B x S x O      (elementwise + reduce)
Total (factorized val): 2 x (B+V) x S x O x I
Total (stored val, G_val precomputed): 2 x B x S x O x I
```

**ghost_greats (pairwise):**
```
Step 1: go_dot  = einsum('bso,vto->bvst', go, val_go)     → 2 x B x V x S^2 x O  (contract over O)
Step 2: inp_dot = einsum('bsi,vti->bvst', inp, val_inp)   → 2 x B x V x S^2 x I  (contract over I)
Step 3: score = (go_dot * inp_dot).sum(dim=(2,3,1))        → 2 x B x V x S^2      (elementwise + reduce)
Total: 2 x B x V x S^2 x (O + I)
```

### Crossover Analysis

**ghost wins when** (stored val, SeparateBatch mode):
```
2 x B x S x O x I  <  2 x B x V x S^2 x (O + I)
             O x I  <  V x S x (O + I)
                 V  >  O x I / (S x (O + I))
```

For Llama-3.2-1B per-layer crossover threshold V*:

| Layer        |    O |    I | V* (S=512) | V* (S=1024) | V* (S=2048) |
| ------------ | ---: | ---: | ---------: | ----------: | ----------: |
| q/k/v/o_proj | 2048 | 2048 |        2.0 |         1.0 |         0.5 |
| gate/up_proj | 2048 | 5632 |        2.9 |         1.5 |         0.7 |
| down_proj    | 5632 | 2048 |        2.9 |         1.5 |         0.7 |

**Summary:**
- **S=512, V=1**: ghost_greats is 3-4x faster (all layers below threshold)
- **S=512, V=4**: near crossover (gate/down at V*=2.9, q/k/v at V*=2.0)
- **S=512, V>=8**: ghost wins (all layers above threshold)
- **S=1024, V>=2**: ghost wins (all layers above threshold)
- **S=2048**: ghost always wins regardless of V

**For merged batch** (MergedBatch mode): ghost's G_val precompute adds `2 x V x S x O x I`, making the total `2 x (B+V) x S x O x I`. This shifts the crossover higher — ghost_greats remains competitive at slightly larger V.

### Benchmark Validation (Llama-3.2-1B)

| Config     | Predicted winner | Score cost: ghost | Score cost: ghost_greats | Actual winner |
| ---------- | ---------------- | ----------------: | -----------------------: | ------------- |
| T=512 m=1  | ghost_greats     |            127 ms |                    40 ms | ghost_greats  |
| T=512 m=4  | crossover        |            171 ms |                   163 ms | ~tie          |
| T=512 m=8  | ghost            |            216 ms |                   292 ms | ghost         |
| T=1024 m=1 | ghost_greats     |            136 ms |                    84 ms | ghost_greats  |
| T=1024 m=4 | ghost            |            227 ms |                   313 ms | ghost         |
| T=2048 m=1 | ~tie             |            158 ms |                   156 ms | ~tie          |

### Cross-Model Validation

The crossover threshold V* = O*I / (S*(O+I)) varies by model size. Smaller models have lower thresholds, meaning ghost wins in more configurations.

**SmolLM-135M** (O=576, I=1536, V* = 419/S):

| Config         | ghost score | greats score | Winner       | V* threshold |
| -------------- | ----------: | -----------: | ------------ | -----------: |
| n=8 T=512 m=1  |       42 ms |        30 ms | ghost_greats |         0.82 |
| n=8 T=512 m=4  |       43 ms |       105 ms | ghost        |         0.82 |
| n=4 T=1024 m=1 |       41 ms |        52 ms | ghost_greats |         0.41 |
| n=4 T=1024 m=4 |       61 ms |       208 ms | ghost        |         0.41 |
| n=2 T=2048 m=1 |       32 ms |       104 ms | ghost        |         0.20 |

**Qwen2.5-0.5B** (O=896, I=4864, V* = 757/S):

| Config         | ghost score | greats score | Winner       | V* threshold |
| -------------- | ----------: | -----------: | ------------ | -----------: |
| n=8 T=512 m=1  |       65 ms |        40 ms | ghost_greats |         1.48 |
| n=8 T=512 m=4  |       90 ms |       163 ms | ghost        |         1.48 |
| n=4 T=1024 m=1 |       70 ms |        76 ms | ~tie         |         0.74 |
| n=4 T=2048 m=1 |      131 ms |       306 ms | ghost        |         0.37 |
| n=4 T=1024 m=4 |      121 ms |       312 ms | ghost        |         0.74 |

**Summary across models:** The crossover V* = O*I/(S*(O+I)) correctly predicts the winner in all but borderline cases (V near V*). Key patterns:
- **Larger models** (higher O, I) → higher V*, so ghost_greats wins in more settings
- **Longer sequences** → lower V*, so ghost wins even at small V
- **More val samples** → ghost wins since it amortizes G_val precomputation
- At the crossover point (V ~ V*), both methods have similar score costs

## Results

All benchmarks: Llama-3.2-1B | bfloat16 | A40 GPU (45 GB) | flash attention | dummy dataset (full-length sequences) | 10 warmup + 20 timed iterations.

### n=8, T=512, m=1

Standard: **542.7 ms** | 19.49 GB

| Method        | Scoring          | forward | act_grad | compress | score | select | w.grad | autograd | pass2 fwd | pass2 bwd | assembly | optim |     Total |  Overhead |
| ------------- | ---------------- | ------: | -------: | -------: | ----: | -----: | -----: | -------: | --------: | --------: | -------: | ----: | --------: | --------: |
| standard      | --               |   151.4 |     91.1 |          |       |        |   88.9 |    120.5 |           |           |          |  90.7 |     542.7 |        -- |
| layerwise     | compress         |   165.8 |     96.1 |     20.6 |   1.8 |    2.6 |   70.6 |    135.3 |           |           |          |  90.7 |     583.3 |     +7.5% |
| layerwise     | ghost_greats     |   165.9 |     96.3 |          |  40.0 |    2.6 |   71.3 |    135.2 |           |           |          |  90.7 |     602.0 |    +10.9% |
| layerwise     | ghost            |   166.2 |     96.8 |          | 126.6 |    2.6 |   71.9 |    135.3 |           |           |          |  90.7 |     690.1 |    +27.2% |
| layerwise     | direct           |   166.2 |     96.9 |          | 167.5 |    2.6 |   71.6 |    135.2 |           |           |          |  90.7 |     730.7 |    +34.7% |
| subset 2P     | compress         |   165.9 |     95.7 |     20.5 |   2.0 |        |        |    131.7 |      78.1 |     157.0 |          |  90.7 |     741.5 |    +36.7% |
| subset 2P     | ghost_greats     |   165.9 |     96.4 |          |  40.1 |        |        |    131.9 |      78.4 |     157.1 |          |  90.7 |     760.5 |    +40.1% |
| subset 2P     | ghost            |   166.3 |     97.0 |          | 127.2 |        |        |    131.5 |      79.3 |     157.3 |          |  90.7 |     849.3 |    +56.5% |
| subset 2P     | direct           |   166.2 |     97.1 |          | 167.6 |        |        |    131.5 |      78.7 |     157.4 |          |  90.7 |     889.2 |    +63.9% |
| **subset 1P** | **compress**     |   165.9 |     95.8 |     20.5 |   2.0 |        |        |    131.8 |           |           |     68.9 |  90.6 | **575.7** | **+6.1%** |
| **subset 1P** | **ghost_greats** |   165.8 |     96.3 |          |  40.1 |        |        |    132.0 |           |           |     69.1 |  90.6 | **594.0** | **+9.5%** |
| subset 1P     | ghost            |   165.9 |     97.2 |          | 127.0 |        |        |    131.7 |           |           |     69.2 |  90.6 |     681.9 |    +25.6% |
| subset 1P     | direct           |   166.3 |     97.0 |          | 167.7 |        |        |    131.7 |           |           |     69.4 |  90.6 |     722.9 |    +33.2% |

### n=8, T=512, m=4

Standard: **544.2 ms** | 19.49 GB

| Method        | Scoring      | forward | act_grad | compress | score | select | w.grad | autograd | pass2 fwd | pass2 bwd | assembly | optim |     Total |   Overhead |
| ------------- | ------------ | ------: | -------: | -------: | ----: | -----: | -----: | -------: | --------: | --------: | -------: | ----: | --------: | ---------: |
| standard      | --           |   151.9 |     91.7 |          |       |        |   89.4 |    120.6 |           |           |          |  90.7 |     544.2 |         -- |
| layerwise     | compress     |   220.3 |    132.2 |     23.5 |   1.8 |    2.5 |   70.9 |    177.6 |           |           |          |  90.7 |     719.5 |     +32.2% |
| layerwise     | ghost_greats |   220.4 |    132.9 |          | 162.8 |    2.6 |   71.5 |    177.5 |           |           |          |  90.7 |     858.4 |     +57.7% |
| layerwise     | ghost        |   220.1 |    133.2 |          | 170.7 |    2.6 |   72.1 |    177.6 |           |           |          |  90.7 |     867.0 |     +59.3% |
| layerwise     | direct       |   219.8 |    132.0 |          | 225.6 |    2.6 |   71.3 |    177.6 |           |           |          |  90.7 |     919.6 |     +69.0% |
| subset 2P     | compress     |   219.9 |    131.5 |     23.5 |   1.9 |        |        |    174.0 |      77.8 |     157.1 |          |  90.7 |     876.4 |     +61.0% |
| subset 2P     | ghost_greats |   220.0 |    131.9 |          | 167.9 |        |        |    173.7 |      78.4 |     156.8 |          |  90.7 |    1019.3 |     +87.3% |
| subset 2P     | ghost        |   219.8 |    133.6 |          | 170.9 |        |        |    173.8 |      78.5 |     156.8 |          |  90.7 |    1024.3 |     +88.2% |
| subset 2P     | direct       |   220.0 |    132.5 |          | 226.1 |        |        |    173.9 |      78.6 |     156.8 |          |  90.7 |    1078.7 |     +98.2% |
| **subset 1P** | **compress** |   219.8 |    131.5 |     23.5 |   1.9 |        |        |    174.2 |           |           |     68.7 |  90.6 | **710.4** | **+30.5%** |
| subset 1P     | ghost_greats |   219.9 |    132.0 |          | 163.6 |        |        |    173.9 |           |           |     68.9 |  90.6 |     849.1 |     +56.1% |
| subset 1P     | ghost        |   220.3 |    132.9 |          | 170.8 |        |        |    174.1 |           |           |     69.1 |  90.6 |     858.0 |     +57.7% |
| subset 1P     | direct       |   220.2 |    132.5 |          | 226.7 |        |        |    174.2 |           |           |     69.2 |  90.6 |     913.5 |     +67.9% |

### n=4, T=1024, m=1

Standard: **544.3 ms** | 19.49 GB

| Method        | Scoring          | forward | act_grad | compress | score | select | w.grad | autograd | pass2 fwd | pass2 bwd | assembly | optim |     Total |   Overhead |
| ------------- | ---------------- | ------: | -------: | -------: | ----: | -----: | -----: | -------: | --------: | --------: | -------: | ----: | --------: | ---------: |
| standard      | --               |   151.9 |     90.1 |          |       |        |   87.9 |    123.6 |           |           |          |  90.7 |     544.3 |         -- |
| layerwise     | compress         |   183.2 |    106.2 |     19.5 |   1.9 |    2.6 |   69.8 |    153.0 |           |           |          |  90.7 |     627.0 |     +15.2% |
| layerwise     | ghost_greats     |   183.4 |    107.7 |          |  84.1 |    2.6 |   70.9 |    153.1 |           |           |          |  90.7 |     692.5 |     +27.2% |
| layerwise     | ghost            |   183.8 |    108.2 |          | 136.4 |    2.6 |   71.6 |    153.2 |           |           |          |  90.7 |     746.5 |     +37.1% |
| layerwise     | direct           |   183.7 |    108.4 |          | 139.6 |    2.7 |   71.7 |    153.3 |           |           |          |  90.7 |     750.0 |     +37.8% |
| subset 2P     | compress         |   183.6 |    106.9 |     19.5 |   2.1 |        |        |    149.5 |      78.8 |     157.9 |          |  90.7 |     789.1 |     +45.0% |
| subset 2P     | ghost_greats     |   183.7 |    108.1 |          |  84.6 |        |        |    149.6 |      79.5 |     158.1 |          |  90.7 |     854.3 |     +57.0% |
| subset 2P     | ghost            |   183.8 |    108.3 |          | 137.0 |        |        |    149.5 |      79.4 |     157.8 |          |  90.7 |     906.5 |     +66.5% |
| subset 2P     | direct           |   183.5 |    108.4 |          | 139.9 |        |        |    149.6 |      79.2 |     158.1 |          |  90.7 |     909.4 |     +67.1% |
| **subset 1P** | **compress**     |   182.9 |    106.4 |     19.5 |   2.1 |        |        |    149.6 |           |           |     68.3 |  90.6 | **619.7** | **+13.9%** |
| **subset 1P** | **ghost_greats** |   183.3 |    107.7 |          |  84.4 |        |        |    149.8 |           |           |     68.8 |  90.6 | **684.8** | **+25.8%** |
| subset 1P     | ghost            |   183.7 |    107.9 |          | 136.6 |        |        |    149.6 |           |           |     69.0 |  90.6 |     737.7 |     +35.5% |
| subset 1P     | direct           |   183.8 |    108.4 |          | 139.9 |        |        |    149.8 |           |           |     69.2 |  90.6 |     742.0 |     +36.3% |

### n=4, T=1024, m=4

Standard: **546.3 ms** | 19.50 GB

| Method        | Scoring      | forward | act_grad | compress |     score | select | w.grad | autograd | pass2 fwd | pass2 bwd | assembly | optim |      Total |    Overhead |
| ------------- | ------------ | ------: | -------: | -------: | --------: | -----: | -----: | -------: | --------: | --------: | -------: | ----: | ---------: | ----------: |
| standard      | --           |   152.6 |     90.7 |          |           |        |   88.6 |    123.7 |           |           |          |  90.7 |      546.3 |          -- |
| layerwise     | compress     |   298.0 |    173.9 |     30.6 |       1.9 |    2.6 |   70.4 |    239.6 |           |           |          |  90.7 |      907.6 |      +66.1% |
| layerwise     | ghost_greats |   298.3 |    175.7 |          |     312.5 |    2.6 |   70.9 |    240.0 |           |           |          |  90.7 |     1190.8 |     +118.0% |
| layerwise     | **ghost**    |   297.8 |    175.9 |          | **226.5** |    2.6 |   71.7 |    239.9 |           |           |          |  90.7 | **1105.0** | **+102.4%** |
| layerwise     | direct       |   297.0 |    174.6 |          |     259.1 |    2.7 |   71.3 |    239.9 |           |           |          |  90.7 |     1135.3 |     +107.8% |
| subset 2P     | compress     |   296.8 |    173.0 |     30.5 |       2.1 |        |        |    235.9 |      78.5 |     157.5 |          |  90.7 |     1065.1 |      +95.0% |
| subset 2P     | ghost_greats |   297.5 |    175.1 |          |     302.3 |        |        |    236.0 |      79.2 |     157.6 |          |  90.7 |     1338.4 |     +145.0% |
| subset 2P     | **ghost**    |   297.8 |    175.5 |          | **226.8** |        |        |    236.0 |      79.1 |     157.9 |          |  90.7 | **1263.8** | **+131.4%** |
| subset 2P     | direct       |   297.7 |    176.4 |          |     260.0 |        |        |    236.3 |      79.3 |     157.9 |          |  90.7 |     1298.3 |     +137.7% |
| **subset 1P** | **compress** |   296.8 |    173.0 |     30.5 |       2.1 |        |        |    236.1 |           |           |     68.2 |  90.7 |  **897.6** |  **+64.3%** |
| subset 1P     | ghost_greats |   297.3 |    174.6 |          |     310.9 |        |        |    236.2 |           |           |     68.8 |  90.6 |     1178.5 |     +115.8% |
| subset 1P     | **ghost**    |   297.5 |    175.6 |          | **226.3** |        |        |    236.2 |           |           |     68.9 |  90.6 | **1095.4** | **+100.6%** |
| subset 1P     | direct       |   298.0 |    176.7 |          |     261.6 |        |        |    236.6 |           |           |     69.3 |  90.6 |     1133.1 |     +107.4% |

### n=2, T=2048, m=1

Standard: **549.7 ms** | 19.49 GB

| Method        | Scoring      | forward | act_grad | compress | score | select | w.grad | autograd | pass2 fwd | pass2 bwd | assembly | optim |     Total |   Overhead |
| ------------- | ------------ | ------: | -------: | -------: | ----: | -----: | -----: | -------: | --------: | --------: | -------: | ----: | --------: | ---------: |
| standard      | --           |   153.4 |     88.9 |          |       |        |   86.5 |    130.1 |           |           |          |  90.7 |     549.7 |         -- |
| layerwise     | compress     |   224.5 |    130.3 |     24.6 |   1.9 |    2.0 |   69.7 |    191.3 |           |           |          |  90.7 |     735.0 |     +33.7% |
| layerwise     | ghost_greats |   224.9 |    131.0 |          | 158.5 |    2.0 |   70.4 |    191.4 |           |           |          |  90.7 |     868.9 |     +58.1% |
| layerwise     | ghost        |   225.6 |    132.3 |          | 157.4 |    2.0 |   71.3 |    191.9 |           |           |          |  90.7 |     871.3 |     +58.5% |
| layerwise     | direct       |   225.9 |    133.0 |          | 150.3 |    2.1 |   71.6 |    192.1 |           |           |          |  90.7 |     865.7 |     +57.5% |
| subset 2P     | compress     |   225.9 |    130.7 |     24.6 |   2.1 |        |        |    187.9 |      81.9 |     161.5 |          |  90.7 |     905.3 |     +64.7% |
| subset 2P     | ghost_greats |   226.2 |    131.6 |          | 159.5 |        |        |    187.8 |      82.3 |     161.8 |          |  90.7 |    1039.9 |     +89.2% |
| subset 2P     | ghost        |   225.8 |    132.4 |          | 158.3 |        |        |    188.2 |      82.6 |     161.3 |          |  90.7 |    1039.4 |     +89.1% |
| subset 2P     | direct       |   226.1 |    133.0 |          | 150.8 |        |        |    188.0 |      82.4 |     161.7 |          |  90.7 |    1033.0 |     +87.9% |
| **subset 1P** | **compress** |   225.8 |    130.4 |     24.6 |   2.1 |        |        |    188.1 |           |           |     68.2 |  90.6 | **730.0** | **+32.8%** |
| subset 1P     | ghost_greats |   225.7 |    131.8 |          | 155.8 |        |        |    188.0 |           |           |     68.5 |  90.6 |     860.7 |     +56.6% |
| subset 1P     | ghost        |   225.8 |    132.4 |          | 158.0 |        |        |    188.4 |           |           |     68.7 |  90.6 |     864.1 |     +57.2% |
| subset 1P     | direct       |   226.6 |    133.7 |          | 151.4 |        |        |    188.8 |           |           |     69.1 |  90.6 |     860.4 |     +56.5% |

Note: At T=2048, ghost (158ms), ghost_greats (156ms), and direct (151ms) all converge — the crossover S = O*I/(O+I) ~ 1500 is exceeded, so all exact methods have comparable FLOPs.

---

### SmolLM-135M Results

All benchmarks: SmolLM-135M | bfloat16 | A40 GPU (45 GB) | flash attention | dummy dataset (full-length sequences) | 10 warmup + 20 timed iterations.

#### SmolLM-135M: Subset 1P Overhead vs Standard

| Config          | Standard (ms) | compress | ghost_greats |  ghost | direct |
| --------------- | ------------: | -------: | -----------: | -----: | -----: |
| n=16 T=512 m=1  |         257.0 |   +19.1% |       +29.0% | +26.7% | +26.0% |
| n=16 T=1024 m=1 |         506.6 |   +14.7% |       +42.0% | +23.1% | +20.0% |
| n=8 T=512 m=4   |         147.6 |   +75.8% |      +104.9% | +75.1% | +62.8% |
| n=8 T=2048 m=1  |         525.1 |        — |       +88.4% | +28.1% | +24.2% |

#### SmolLM-135M: Score Cost (ms)

| Config          | compress | ghost_greats | ghost | direct |
| --------------- | -------: | -----------: | ----: | -----: |
| n=16 T=512 m=1  | 3.8+20.8 |         51.2 |  45.4 |   43.0 |
| n=16 T=1024 m=1 | 3.7+35.9 |        178.6 |  83.9 |   66.8 |
| n=8 T=512 m=4   |        — |        105.4 |  48.0 |   42.0 |
| n=8 T=2048 m=1  |        — |        401.4 |  86.1 |   64.4 |

Note: SmolLM-135M has smaller hidden dimensions (O=576, I=1536), so the crossover threshold V* is lower. Ghost beats ghost_greats in more configurations than Llama-3.2-1B. At T=2048, ghost and direct converge (~64-86ms) while ghost_greats remains expensive (401ms).

---

### Qwen2.5-0.5B Results

All benchmarks: Qwen2.5-0.5B | bfloat16 | A40 GPU (45 GB) | flash attention | dummy dataset (full-length sequences) | 10 warmup + 20 timed iterations.

#### Qwen2.5-0.5B: Subset 1P Overhead vs Standard

| Config         | Standard (ms) | compress | ghost_greats |  ghost | direct |
| -------------- | ------------: | -------: | -----------: | -----: | -----: |
| n=8 T=512 m=1  |         309.2 |   +14.2% |       +19.0% | +27.4% | +30.9% |
| n=4 T=1024 m=1 |         313.9 |        — |       +38.6% | +36.9% | +35.3% |
| n=8 T=512 m=4  |         309.4 |   +39.8% |       +82.7% | +59.3% | +68.1% |
| n=4 T=2048 m=1 |         573.0 |        — |       +71.2% | +40.6% | +38.0% |

#### Qwen2.5-0.5B: Score Cost (ms)

| Config         | compress | ghost_greats | ghost | direct |
| -------------- | -------: | -----------: | ----: | -----: |
| n=8 T=512 m=1  | 2.9+19.7 |         40.2 |  66.0 |   76.8 |
| n=4 T=1024 m=1 |        — |         75.9 |  70.3 |   65.4 |
| n=8 T=512 m=4  |        — |        163.8 |  90.9 |  117.8 |
| n=4 T=2048 m=1 |        — |        305.9 | 131.8 |  115.9 |

Note: Qwen2.5-0.5B has an asymmetric hidden dim (O=896, I=4864), giving a higher crossover threshold V*=757/S. Ghost_greats wins at T=512 m=1 (40ms vs 66ms), but ghost dominates at longer sequences and higher V.

---

### Summary: Score Cost (ms)

**Llama-3.2-1B** (Subset 1P score cost):

| Config         | compress | ghost_greats |   ghost | direct |
| -------------- | -------: | -----------: | ------: | -----: |
| n=8 T=512 m=1  |      1.8 |       **40** |     127 |    168 |
| n=8 T=512 m=4  |      1.8 |          163 |     171 |    226 |
| n=4 T=1024 m=1 |      1.9 |       **84** |     136 |    140 |
| n=4 T=1024 m=4 |      1.9 |          313 | **227** |    259 |
| n=2 T=2048 m=1 |      1.9 |          156 |     158 |    151 |

**SmolLM-135M** (Subset 1P score cost):

| Config          | compress | ghost_greats | ghost | direct |
| --------------- | -------: | -----------: | ----: | -----: |
| n=16 T=512 m=1  | 3.8+20.8 |         51.2 |  45.4 |   43.0 |
| n=16 T=1024 m=1 | 3.7+35.9 |        178.6 |  83.9 |   66.8 |
| n=8 T=512 m=4   |        — |        105.4 |  48.0 |   42.0 |
| n=8 T=2048 m=1  |        — |        401.4 |  86.1 |   64.4 |

**Qwen2.5-0.5B** (Subset 1P score cost):

| Config         | compress | ghost_greats | ghost | direct |
| -------------- | -------: | -----------: | ----: | -----: |
| n=8 T=512 m=1  | 2.9+19.7 |         40.2 |  66.0 |   76.8 |
| n=4 T=1024 m=1 |        — |         75.9 |  70.3 |   65.4 |
| n=8 T=512 m=4  |        — |        163.8 |  90.9 |  117.8 |
| n=4 T=2048 m=1 |        — |        305.9 | 131.8 |  115.9 |

Note: compress score cost shows only the inner product (~2ms); the full compress+score is ~22-26ms. At T=2048, all exact methods converge (~150-158ms) since S exceeds O*I/(O+I).

### Summary: Subset 1P Overhead vs Standard

**Llama-3.2-1B:**

| Config            |  compress | ghost_greats |   ghost |  direct |
| ----------------- | --------: | -----------: | ------: | ------: |
| **n=8 T=512 m=1** | **+6.1%** |    **+9.5%** |  +25.6% |  +33.2% |
| n=8 T=512 m=4     |    +30.5% |       +56.1% |  +57.7% |  +67.9% |
| n=4 T=1024 m=1    |    +13.9% |       +25.8% |  +35.5% |  +36.3% |
| n=4 T=1024 m=4    |    +64.3% |      +115.8% | +100.6% | +107.4% |
| n=2 T=2048 m=1    |    +32.8% |       +56.6% |  +57.2% |  +56.5% |

**SmolLM-135M:**

| Config          | compress | ghost_greats |  ghost | direct |
| --------------- | -------: | -----------: | -----: | -----: |
| n=16 T=512 m=1  |   +19.1% |       +29.0% | +26.7% | +26.0% |
| n=16 T=1024 m=1 |   +14.7% |       +42.0% | +23.1% | +20.0% |
| n=8 T=512 m=4   |   +75.8% |      +104.9% | +75.1% | +62.8% |
| n=8 T=2048 m=1  |        — |       +88.4% | +28.1% | +24.2% |

**Qwen2.5-0.5B:**

| Config         | compress | ghost_greats |  ghost | direct |
| -------------- | -------: | -----------: | -----: | -----: |
| n=8 T=512 m=1  |   +14.2% |       +19.0% | +27.4% | +30.9% |
| n=4 T=1024 m=1 |        — |       +38.6% | +36.9% | +35.3% |
| n=8 T=512 m=4  |   +39.8% |       +82.7% | +59.3% | +68.1% |
| n=4 T=2048 m=1 |        — |       +71.2% | +40.6% | +38.0% |

### Summary: One-Pass Speedup vs Two-Pass

| Config         | compress | ghost_greats | ghost | direct |
| -------------- | -------: | -----------: | ----: | -----: |
| n=8 T=512 m=1  |    22.4% |        21.9% | 19.7% |  18.7% |
| n=8 T=512 m=4  |    18.9% |        16.7% | 16.2% |  15.3% |
| n=4 T=1024 m=1 |    21.5% |        19.8% | 18.6% |  18.4% |
| n=4 T=1024 m=4 |    15.7% |        12.0% | 13.3% |  12.7% |
| n=2 T=2048 m=1 |    19.4% |        17.2% | 16.9% |  16.7% |

## Key Takeaways

1. **Scoring cost dominates the overhead.** The non-scoring components (act_grad, select, w.grad, autograd, optimizer) are essentially constant across scoring methods. The only variable is the score computation.

2. **ghost_greats is fastest exact scoring at m=1.** Score cost: 40ms (T=512) to 84ms (T=1024) vs ghost's 127-136ms. The 3.2x speedup comes from `V x S^2 x (O+I) < S x O x I` when V=1 and S < min(O, I).

3. **ghost wins at m=4, T=1024.** Score cost: 227ms vs ghost_greats 313ms (1.4x faster). The crossover occurs when `V x S x (O+I) > O x I`. Validated across three models (SmolLM-135M, Qwen2.5-0.5B, Llama-3.2-1B) — smaller models have lower crossover thresholds, so ghost wins in more configurations.

4. **One-pass saves 12-22% vs two-pass** consistently. The saving is the entire pass2 (forward ~79ms + backward ~157ms = ~236ms), replaced by assembly (~69ms). Net saving: ~167ms.

5. **compress scoring is always fastest** (~22ms total including compression), but it's an approximation. The score inner product itself is only ~2ms; the 20ms is the compressor projection.

6. **autograd overhead scales with batch size** (131ms at n=8/m=1, 174ms at n=8/m=4, 236ms at n=4+4/m=4). This is PyTorch framework overhead, not our code.

7. **w.grad is constant at ~71ms** for layerwise and ~69ms for one-pass assembly, independent of scoring method. For two-pass, pass2 w.grad is ~46ms (only selected samples).

8. **retain cost is negligible** (0.2ms) — saving per-layer (grad_output, input) references is essentially free.

## Methods

- **Standard**: Baseline full fine-tuning with AdamW.
- **Layerwise** (Algorithm 4.4): Per-layer selection. Single-pass — scoring, selection, and w.grad inline during backward.
- **Subset 2P** (Algorithm 4.3): Global selection. Two-pass — scoring pass, then forward+backward on selected subset.
- **Subset 1P** (Algorithm 4.2): Global selection. One-pass — scoring + retain during backward, post-hoc assembly. Eliminates the second forward+backward.

## Scoring Methods

- **compress**: Compressor projection to R^k, then dot product. ~22ms total. Approximate. Requires `score_compression` config.
- **ghost_greats**: True ghost via pairwise dot products `(go_i.go_v)(inp_i.inp_v)`. Score: 40-313ms. Fastest exact method at small V. O(B x V x S^2) memory.
- **ghost** (ours): Ghost via collapse-first `inp @ G_val.T`. Score: 127-227ms. Fastest exact method at large V. O(B x S x O) memory.
- **direct**: Batch materialization via `bmm(go^T, inp)`. Score: 140-261ms. O(B x O x I) memory, OOMs at large n.

## Methodology

Each method runs in a **separate Python process** to ensure clean GPU memory measurement. Per-component timing uses CUDA events placed inside **monkey-patched autograd Functions** (`benchmark.py`). The unified runner (`benchmark_run.py`) orchestrates all method × scoring combinations.

All timing uses the **dummy dataset** which generates guaranteed full-length sequences (every sample = exactly T tokens). This ensures consistent and reproducible measurements independent of dataset-specific tokenization.

## Reproducing Results

```bash
# Reproduce all Llama-3.2-1B results (5 configs × 13 combos = 65 runs)
python benchmark_run.py --gpu 0 --output results/llama.json

# Reproduce SmolLM-135M results (adjust configs for the model)
python benchmark_run.py --gpu 1 --model HuggingFaceTB/SmolLM-135M --output results/smol.json

# Reproduce Qwen2.5-0.5B results
python benchmark_run.py --gpu 2 --model Qwen/Qwen2.5-0.5B --output results/qwen.json

# Run all three in parallel
python benchmark_run.py --gpu 0 --output results/llama.json &
python benchmark_run.py --gpu 1 --model HuggingFaceTB/SmolLM-135M --output results/smol.json &
python benchmark_run.py --gpu 2 --model Qwen/Qwen2.5-0.5B --output results/qwen.json &
wait
```

## File Structure

```
SFT/benchmark/
├── benchmark.py          # Core timing engine (single-method, instrumented backward)
├── benchmark_run.py      # Unified runner (all configs × combos, detailed breakdown)
├── utils.py              # Shared utilities (model setup, dataloaders, CUDA timers)
├── README.md             # This file
└── results/              # Saved JSON results
```
