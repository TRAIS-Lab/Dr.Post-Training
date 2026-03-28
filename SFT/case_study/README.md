# Case Study: Layerwise vs Subset Selection Analysis

## Overview

This case study examines **why Layerwise selection outperforms Subset selection** by analyzing the per-layer influence scores during standard training. We find two key results:

1. **Score magnitudes are heavily skewed across layer types and blocks.** `down_proj` layers produce scores 40–114× larger than `k_proj` layers, despite some having identical parameter counts. Block 1's `down_proj` alone dominates the entire score landscape.

2. **Different layer types would select different training samples.** The within-attention rank correlation is only 0.33–0.55, meaning Q/K/V/O projections frequently disagree on which samples are most useful. Subset selection, dominated by `down_proj`'s large scores, effectively ignores the preferences of Q and K projections (Jaccard with Subset ≈ 0.45, barely above random).

Together, these explain why Subset underperforms: it forces all layers to train on data chosen by a single layer type, even though different layers benefit from different samples. Layerwise selection respects each layer's individual preference.

## Experimental Setup

The case study uses the **same training configuration as the main SFT experiments** — same model, data percentage, learning rate, and hyperparameters. The only addition is that at each training step, we compute what both Layerwise and Subset selection **would** pick (without applying the selection), recording per-layer scores for post-hoc analysis.

### Model
- **Llama-3.2-1B** (16 transformer blocks, 113 linear layers)
- Each block has 7 linear layers: Q, K, V, O (attention) + Gate, Up, Down (MLP)
- Full-parameter fine-tuning (no LoRA)

### Dataset Combinations

|                | tulu3 → tydiqa              | alpaca → samsum                 |
| -------------- | --------------------------- | ------------------------------- |
| Training data  | 1% of tulu3 (~9.4k samples) | 40% of alpaca (~20.8k samples)  |
| Eval task      | TyDiQA (cross-lingual QA)   | SAMSum (dialogue summarization) |
| Learning rate  | 4.96e-05                    | 1e-06                           |
| Training steps | 1,175 (1 epoch)             | 2,600 (1 epoch)                 |
| Records        | Every step (1,175 total)    | Every 2 steps (1,300 total)     |

### Common Hyperparameters
- Batch size: 8
- Optimizer: AdamW (weight_decay=0.0)
- LR scheduler: linear with warmup_ratio=0.03
- Max sequence length: 512
- BF16 training, flash attention
- Seed: 42 (data seed: 43)
- n_val: 16 (validation samples for scoring)
- n_eval: 500 (evaluation samples for perplexity)
- Selection fraction: 0.5 (select top 4 of 8)
- Val batch size for scoring: 1

### Scoring Protocol
At each training step:
1. Standard forward + backward pass (no hooks) → real gradients for optimizer
2. Save real gradients
3. Layerwise scoring pass → per-layer influence scores and selections for all 113 layers
4. Subset scoring pass → global accumulated scores and selection
5. Restore real gradients → optimizer step proceeds normally

The scoring uses `lr=1.0` (since only relative rankings matter) and factorized validation gradient storage.

## Key Findings

### Finding 1: Score magnitude is heavily skewed

The influence score magnitude varies by **2 orders of magnitude** across layer types, and this is **not explained by parameter count**:

| Layer type | Shape     | Score std (tulu3) | Score std (alpaca) | Down/This ratio |
| ---------- | --------- | ----------------- | ------------------ | --------------- |
| Down       | 2048×8192 | 0.254             | 0.711              | 1×              |
| V          | 512×2048  | 0.034             | 0.165              | 4–7×            |
| O          | 2048×2048 | 0.013             | 0.064              | 11–20×          |
| Up         | 8192×2048 | 0.010             | 0.060              | 12–25×          |
| Q          | 2048×2048 | 0.005             | 0.018              | 40–54×          |
| Gate       | 8192×2048 | 0.003             | 0.023              | 31–77×          |
| K          | 512×2048  | 0.002             | 0.006              | **114×**        |

Key observations:
- **K and V have identical shapes** (512×2048) but V's score is 7–26× larger
- **Gate, Up, Down have identical shapes** (8192×2048 / 2048×8192) but Down's score is 12–77× larger than Gate
- The skew is driven by **position in the computation graph**: output projections (Down for MLP, O for attention) produce larger scores because they write directly to the residual stream
- **Block 1's `down_proj`** alone has score std ~3.0, dwarfing all other layers

### Finding 2: Layer types disagree on sample selection

| Metric                             | tulu3 → tydiqa | alpaca → samsum |
| ---------------------------------- | -------------- | --------------- |
| Within-attention correlation       | 0.33           | 0.55            |
| Within-MLP correlation             | 0.52           | 0.67            |
| Unique selection patterns per step | 5.3            | 4.3             |
| K Jaccard with Subset              | 0.45           | 0.50            |
| Down Jaccard with Subset           | 0.90           | 0.86            |
| Down Spearman with Subset          | 0.96           | 0.94            |

Key observations:
- Q and K projections have **near-random overlap with Subset** (Jaccard 0.45–0.50 vs random baseline ~0.43)
- Down has **near-perfect correlation with Subset** (Spearman 0.96/0.94)
- On average, **5+ distinct selection patterns** exist among the 7 layer types at each step
- Subset's selection is effectively determined by `down_proj`, which has the largest score magnitude

### Implication

Subset selection is not a balanced vote across layers — it is a **magnitude-weighted average dominated by `down_proj`**. Layers like K and Q, which represent 32 of the model's 112 block-level linear layers, have essentially no influence on which data they receive. Layerwise selection gives each layer equal voice, allowing it to train on the data most beneficial for its specific function.

## Files

| File                          | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| `analyze_selection.py`        | Case study trainer: standard training + dual Layerwise/Subset scoring |
| `run.sh`                      | Launch script with correct hyperparameters                            |
| `plot_case_study.py`          | Analysis and plotting (produces individual subplot PDFs)              |
| `figures/magnitude_tulu3.pdf` | Score magnitude by block × type (tulu3→tydiqa)                        |
| `figures/magnitude_alpaca.pdf`| Score magnitude by block × type (alpaca→samsum)                       |
| `figures/correlation_tulu3.pdf` | Rank correlation with Subset by block × type (tulu3→tydiqa)         |
| `figures/correlation_alpaca.pdf`| Rank correlation with Subset by block × type (alpaca→samsum)        |
| `figures/legend.pdf`          | Standalone legend for all subplots                                    |

### Raw data (on scratch)
| Path                                                                    | Description    |
| ----------------------------------------------------------------------- | -------------- |
| `.../case_study_tulu3_tydiqa-Llama-3.2-1B-p0.01-lr4.96e-05-b8-v16-s42/` | tulu3 results  |
| `.../case_study_alpaca_samsum-Llama-3.2-1B-p0.4-lr1e-06-b8-v16-s42/`    | alpaca results |

Each directory contains `selection_records.json` with per-step, per-layer scores and selections.

## Reproduction

```bash
cd $CODE_DIR/Dr.Post-Training

# tulu3 → tydiqa (matching Standard-Full config)
bash SFT/case_study/run.sh

# alpaca → samsum (matching Standard-Full config)
bash SFT/case_study/run.sh --train alpaca --task samsum --percentage 0.4 --lr 1e-06 --record_freq 2

# Generate figure
python3 SFT/case_study/plot_case_study.py
```
