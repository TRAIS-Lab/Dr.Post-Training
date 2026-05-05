# SFT Experiments

Training and evaluation code for Supervised Fine-Tuning experiments. See
`PROGRESS.md` for the current scope and run inventory.

## Scope

7 train→target settings, Full-FT only first (`*-Full` methods + Target-only
baseline). LoRA / MeSO can be added once the curation gain is verified on
Full-FT for the new settings.

| # | Config dir          | Train pool | Target task | Model        | Step budget | `eval_steps` |
|---|---------------------|------------|-------------|--------------|-------------|--------------|
| 1 | `alpaca_samsum`     | alpaca     | samsum      | Llama-3.2-1B | 2600        | 26           |
| 2 | `tulu3_tydiqa`      | tulu3      | tydiqa      | Llama-3.2-1B | 1174        | 12           |
| 3 | `nq_triviaqa`       | nq_open    | triviaqa    | Llama-3.2-1B | 1100        | 11           |
| 4 | `less_tydiqa`       | less (mix) | tydiqa      | Llama-3.2-1B | 1225        | 12           |
| 5 | `nq_triviaqa_qwen3` | nq_open    | triviaqa    | Qwen3-1.7B-Base | 1100     | 11           |
| 6 | `triviaqa_nq`       | triviaqa   | nq_open     | Llama-3.2-1B | 1107        | 11           |
| 7 | `squad_triviaqa`    | squad      | triviaqa    | Llama-3.2-1B | 1095        | 11           |

LESS-mix = `flan_v2 + cot + dolly + oasst1` (~1.96M).

## Data Preparation

```bash
# Eval splits (val/lr/test)
python SFT/data/prepare_datasets.py --datasets tydiqa samsum triviaqa nq_open_eval

# Training pools
python SFT/data/prepare_datasets.py --datasets nq_open triviaqa_train squad alpaca tulu3

# LESS-mix components (used as a single train pool via train_dataset_names: ['less'])
python SFT/data/prepare_datasets.py --datasets flan_v2 cot dolly oasst1
```

| Dataset            | Role  | Size  | Description                                      |
| ------------------ | ----- | ----- | ------------------------------------------------ |
| `tydiqa`           | eval  | 4877 (test) | Multilingual extractive QA                  |
| `samsum`           | eval  | 719 (test)  | Dialogue summarization (also has train split) |
| `triviaqa`         | eval  | 1000 (test) | Closed-book factoid QA                        |
| `nq_open_eval`     | eval  | 1000 (test) | NaturalQuestions-open closed-book QA          |
| `nq_open`          | train | 88K   | NQ-open Q→A pairs                                |
| `triviaqa_train`   | train | 138K  | TriviaQA `rc.nocontext` Q→A pairs                |
| `squad`            | train | 88K   | SQuAD with-context reading-comprehension         |
| `tulu3`            | train | 939K  | Tulu-3 SFT mixture                               |
| `alpaca`           | train | 52K   | Stanford Alpaca instruction-following            |
| `dolly`/`flan_v2`/`cot`/`oasst1` | train | 1.96M total | LESS-mix components            |

## Methods (Full-FT only)

| Config              | Curation type    | Description                                  |
| ------------------- | ---------------- | -------------------------------------------- |
| `FullTraining-Full` | NA               | Baseline full fine-tuning, no data curation  |
| `GlobalSubset-Full` | GlobalSubset     | Global top-k curation, full fine-tuning      |
| `LayerWiseSubset-Full` | LayerWiseSubset | Per-layer top-k curation, full fine-tuning |
| Target-only (`FullTraining-Full` via `train_val_ablation.sh`) | NA | Baseline trained directly on the n_val=16 task validation samples |

> Run dirs: `{train}_{task}-{model}-{Method}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

## LR Sweep

Best LRs are written into each method's YAML config (`learning_rate:` field) and
into `Target-only.lr.txt` for target-only.

#### Three-way data split

| Split | File | Purpose |
|-------|------|---------|
| `validation` | `{task}_validation_data.jsonl` | Source of n_val=16 curation reference + target-only training |
| `lr` | `{task}_lr_data.jsonl` | LR sweep evaluation |
| `test` | `{task}_test_data.jsonl` | During-training perplexity (first 500) + final task metric (first 500) |

Regenerate splits with: `python SFT/data/prepare_datasets.py --datasets <task>`

#### Main-run LR sweep

```bash
# 3 methods × 20 LRs = 60 jobs per setting
bash SFT/train/lr/lr_sweep_submit.sh -c configs/<setting> -m all

# After completion, write best LR back to method YAMLs
bash SFT/train/lr/lr_sweep_collect.sh -c configs/<setting> -m all
```

Grid: 20 log-spaced LRs in `[1e-7, 1e-3]`. Collect picks the smallest LR within
1% of best eval_loss (stability margin).

#### Target-only LR sweep

```bash
# Per-task; ms (max_steps) and bs (batch_size) must match the corresponding main run
bash SFT/train/lr/lr_sweep_submit_val.sh --task <task> --max_steps <ms> --batch_size 8 --methods FullTraining-Full

bash SFT/train/lr/lr_sweep_collect_val.sh --task <task> --max_steps <ms> --batch_size 8 \
    --out_file SFT/train/configs/<setting>/Target-only.lr.txt
```

## Training

```bash
bash SFT/train/train.sh -c configs/<setting> -m all
bash SFT/train/train.sh -c configs/<setting> -m FullTraining-Full --seed 42
bash SFT/train/train.sh -c configs/<setting> --list
```

Categories: `all`, `full-training`, `layer-wise-subset`, `global-subset`, `full`, `lora`, `meso`.

## Target-only training

```bash
bash SFT/train/train_val_ablation.sh \
    --task <target_task> --methods FullTraining-Full \
    --max_steps <main_ms> --batch_size 8 --eval_steps <main_eval_steps> \
    --seed <seed>
```

## Evaluation

```bash
# n_test=500 matches the during-training perplexity sample for direct comparison
bash SFT/eval/eval.sh --train <train> --task <task> --batch_size 64 --n_test 500
```

Supported tasks: `samsum`, `tydiqa`, `triviaqa`, `nq_open`.

## Config Directory Structure

Each config dir has `defaults.yaml` (shared) and one YAML per method:

```
configs/<setting>/
  defaults.yaml             # model, train pool, target task, step budget, etc.
  FullTraining-Full.yaml    # learning_rate (LR-swept)
  GlobalSubset-Full.yaml    # learning_rate + scoring
  LayerWiseSubset-Full.yaml # learning_rate + scoring (compress)
  Target-only.lr.txt        # LR-swept value for Target-only baseline
```

`defaults.yaml` example:
```yaml
model: meta-llama/Llama-3.2-1B
train_dataset: nq_open
target_task: triviaqa
percentage: 0.1
seed: 42
batch_size: 8
gradient_accumulation_steps: 1
optim: adamw_torch
max_seq_length: 512
lr_scheduler_type: linear
warmup_ratio: 0.03
weight_decay: 0.0
num_train_epochs: 1
eval_steps: 11        # ~100 ppl points across 1100 steps
use_flash_attention: true
n_eval: 500           # during-training perplexity samples (first 500 of test)
selection_frac: 0.5
selection_mode: topk
n_val: 16
val_batch_size: 1
val_strategy: merged_batch
scoring:
  method: reduced_ghost
```

Load order: defaults → `defaults.yaml` → method YAML → CLI overrides (`--seed`, `--lr`).

#### Adding a new setting

1. Create `configs/<new_setting>/` (e.g., `configs/triviaqa_nq/`).
2. Copy a `defaults.yaml` from an existing setting; update `train_dataset`,
   `target_task`, `percentage`, `eval_steps` to give ~100 ppl points.
3. Copy 3 method YAMLs (LR placeholders).
4. Prepare data: `python SFT/data/prepare_datasets.py --datasets <pool> <task>`.
5. LR sweep: `bash SFT/train/lr/lr_sweep_submit.sh -c configs/<new_setting> -m all`.
6. Collect: `bash SFT/train/lr/lr_sweep_collect.sh -c configs/<new_setting> -m all`.
