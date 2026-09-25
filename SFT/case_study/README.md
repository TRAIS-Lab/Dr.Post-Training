# Case Study: LayerWiseSubset vs GlobalSubset Selection Analysis

## Overview

This case study examines **why LayerWiseSubset selection outperforms
GlobalSubset selection** by analyzing the per-layer influence scores
during standard training. The findings are:

1. **Score magnitudes are heavily skewed across layer types and blocks.**
   `down_proj` layers produce scores one to two orders of magnitude
   larger than other layer types, despite some having identical
   parameter counts. A small number of early-block `down_proj` layers
   dominate the entire score landscape.

2. **Different layer types would select different training samples.**
   Within-attention and within-MLP rank correlations are well below 1,
   meaning Q/K/V/O projections frequently disagree on which samples are
   most useful. GlobalSubset selection, dominated by `down_proj`'s large
   scores, has near-random overlap with what Q and K would have picked.

Together, these explain why GlobalSubset underperforms: it forces all
layers to train on data chosen by a single layer type, even though
different layers benefit from different samples. LayerWiseSubset
selection respects each layer's individual preference.

The paper figure (fig:case-study) and the numbers quoted with it are produced by
`make_figures.py` from the recorded `selection_records.json` files (`--summary`
prints the type-averaged magnitudes and correlations).

## Experimental Setup

The case study mirrors the main SFT experiments: same model, same data
percentages, same fixed LRs, same batch size, same scheduler. The only
addition is that at each training step we compute what both
LayerWiseSubset and GlobalSubset selection **would** pick (without
applying the selection), recording per-layer scores for post-hoc
analysis.

### Model
- **`meta-llama/Llama-3.2-1B`** (16 transformer blocks, 113 linear layers)
- Each block has 7 linear layers: Q, K, V, O (attention) + Gate, Up, Down (MLP)
- Full-parameter fine-tuning (no LoRA), no compression

### Settings (matches main 4-setting matrix × 5 seeds)

|                | alpaca → samsum | less → tydiqa | triviaqa → nq_open | less → squad |
| -------------- | --------------- | ------------- | ------------------ | ------------ |
| Train pool     | alpaca          | LESS mix      | triviaqa           | LESS mix     |
| Percentage     | 0.4             | 0.005         | 0.05               | 0.005        |
| Step budget    | 2600            | 1225          | 1107               | 1225         |
| Records        | every step      | every step    | every step         | every step   |

LR is `1e-5` for all settings (matches `FullTraining-Full` from the main
experiment). Seeds: `{2, 22, 42, 62, 82}`. Total 20 case-study runs.

### Common Hyperparameters
- Batch size: 8
- Optimizer: AdamW (`weight_decay=0.0`)
- LR scheduler: linear with `warmup_ratio=0.03`
- Max sequence length: 512
- BF16 training, flash attention 2
- Data seed: `seed + 1`
- `n_val=16` (validation samples for scoring)
- `n_eval=500` (evaluation samples for perplexity)
- Selection fraction: `0.5` (select top 4 of 8)
- Val batch size for scoring: `1`, `val_strategy=separate_batch_factorized`

### Scoring Protocol
At each training step:
1. Standard forward + backward pass (no hooks) → real gradients for the optimizer
2. Save real gradients
3. LayerWiseSubset scoring pass → per-layer influence scores and selections for all 113 layers
4. GlobalSubset scoring pass → global accumulated scores and selection
5. Restore real gradients → optimizer step proceeds normally

Scoring uses `lr=1.0` (only relative rankings matter) and factorized
validation gradient storage. The actual training is unaffected.

## Files

| File                          | Description                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| `analyze_selection.py`        | Case-study trainer: standard training + dual LayerWiseSubset/GlobalSubset scoring |
| `run.sh`                      | Single-run launcher (one setting × one seed); also the Slurm array task (manifest lines = its flags) |
| `make_figures.py`             | Loads all 4 settings × 5 seeds, writes the magnitude / correlation panels + legend into `Paper/<venue>/Figures/SFT/case_study/` |
| `result.ipynb`                | Notebook version of the figure code                                        |
| `replot_correlation.py`       | Re-renders the correlation panels from existing figure PDFs (e.g. to relabel rho -> rho_S without the raw records) |

### Raw data

```
$SCRATCH_DIR/Dr.Post-Training/SFT/runs_v2/case_study/        # <results root>/SFT/runs_v2/case_study, see tables/common.py
  alpaca_samsum-Llama-3.2-1B-p0.4-lr1e-05-b8-v16-s{2,22,42,62,82}/
  less_tydiqa-Llama-3.2-1B-p0.005-lr1e-05-b8-v16-s{2,22,42,62,82}/
  triviaqa_nq_open-Llama-3.2-1B-p0.05-lr1e-05-b8-v16-s{2,22,42,62,82}/
  less_squad-Llama-3.2-1B-p0.005-lr1e-05-b8-v16-s{2,22,42,62,82}/
```

Each run dir contains `selection_records.json` (per-step, per-layer
scores and selections) and `evaluation_results.json` (val/eval ppl
trajectories).

## Reproduction

```bash
# Single run (one setting × one seed) — see run.sh for the available flags
bash SFT/case_study/run.sh --train alpaca --task samsum --percentage 0.4 --seed 42

# All 4 settings × 5 seeds as a Slurm array (finished runs are skipped)
python SFT/train/qa_matrix_manifest.py                      # -> SFT/train/manifests/case_study.txt
mkdir -p logs/qa; N=$(wc -l < SFT/train/manifests/case_study.txt)
sbatch --array=0-$((N-1))%2 SFT/train/slurm/qa_array.sbatch SFT/train/manifests/case_study.txt SFT/case_study/run.sh

# Figures into Paper/ICLR/Figures/SFT/case_study/ (+ the numbers quoted in the appendix)
python SFT/case_study/make_figures.py --summary
```
