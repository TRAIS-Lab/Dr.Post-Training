# SFT Experiments

Training and evaluation code for Supervised Fine-Tuning experiments.
See `PROGRESS.md` for the live run inventory and status.

## Scope (4 active settings)

3 LoRA-only train→target settings + 1 multi-finetuning setting
(`alpaca → samsum`, covering Full/LoRA/MeSO), each at 5 seeds. Per-task
target-only baselines train directly on `n_val=16` validation samples.

| # | Config dir       | Train pool | Target task | Step budget | `eval_steps` | Methods                  |
|---|------------------|------------|-------------|-------------|--------------|--------------------------|
| 1 | `alpaca_samsum`  | alpaca     | samsum      | 2600        | 26           | 9 (Full+LoRA+MeSO × 3 curations) |
| 2 | `less_tydiqa`    | less mix   | tydiqa      | 1225        | 12           | 3 (LoRA × 3 curations)   |
| 3 | `triviaqa_nq`    | triviaqa   | nq_open     | 1107        | 11           | 3 (LoRA × 3 curations)   |
| 4 | `less_squad`     | less mix   | squad       | 1225        | 12           | 3 (LoRA × 3 curations)   |

LESS mix = `flan_v2 + cot + dolly + oasst1` (~1.96M). Run-dir prefix is
`{train}_{task}` so setting 3 produces `triviaqa_nq_open-...`.

## Dolci capability setting (Qwen3 + Dolci-Instruct pools)

A fifth family of settings follows the design of `Dr.Post-Training-Next`: a Qwen3
base model is fine-tuned on a 32K-row pool sampled from `allenai/Dolci-Instruct-SFT`,
curation is steered by a small target set, and the final number is an official
downstream benchmark scored by generation. The harness keeps the data roles and
knobs above; only what the files *mean* changes.

### Data roles

| Role | Knob | File | Size | Meaning |
|---|---|---|---:|---|
| Train pool | `train_dataset`, `percentage: 1.0` | `train/dolci_instruction/dolci_instruction_data.jsonl` | 32,000 | Candidates; `batch_size` per step |
| Target set D* | `n_val`, `val_batch_size` | `eval/precise_if/precise_if_validation_data.jsonl` | 16, 1 per step | Selection gradient; also the `val_loss` curve |
| Target held-out | `n_eval` | `eval/precise_if/precise_if_test_data.jsonl` | 500 | `eval_loss` curve only |
| Benchmark | `eval.sh --target` | `eval/ifeval/ifeval_bench_data.jsonl` | full official set | Post-hoc metric via generation + verifier |

The target's `test` file is **not** the benchmark. It is a loss-only held-out
drawn from the same source as D* (Dolci Precise-IF rows, MATH train, MBPP
train). The benchmark is a different dataset that shares the skill (IFEval and
IFBench, MATH500, MBPP+) and carries verifier metadata instead of reference
answers. One target can map to several benchmarks:

| Target (`target_task`) | Benchmarks | Config dirs |
|---|---|---|
| `precise_if` | `ifeval`, `ifbench` | `dolci_inst_if`, `dolci_mixed_if` |
| `math` | `math500` | `dolci_reason_math`, `dolci_mixed_math` |
| `mbpp` | `mbpp_plus` | `dolci_reason_code` |

If a target file has fewer rows than `n_val`/`n_eval` request, the loader logs a
warning and uses what exists (MBPP train has only 374 rows).

### What the harness adds

- **Chat template.** `SFT/data/chat_format.py` renders every string through the
  tokenizer's own chat template when it has one (Qwen3) and the tulu-style
  fallback otherwise (Llama-3.2-1B). Training supervision of assistant turns is
  computed from the fast tokenizer's offset mapping, which stays correct on Qwen3
  multi-turn data (the empty `<think>` scaffold is only injected on the final
  assistant turn). The scaffold is treated as prompt, and evaluation prompts are
  rendered with thinking disabled to match.
- **Generic target loader.** Any `eval/<task>/<task>_{validation,test}_data.jsonl`
  in `messages` format loads without code changes (`precise_if`, `math`, `mbpp`,
  `truthfulqa` are pre-registered).
- **Knobs.** `gradient_checkpointing: true` (non-reentrant; needed for 1.7B at
  4096 tokens on a 48 GB GPU) and `val_seq_length_multiplier` (D* length
  rejection; `0` disables) are new `defaults.yaml` keys.
- **Benchmarks.** `SFT/eval/tasks/{ifeval,ifbench,math500,mbpp_plus}.py`, with
  `eval.sh --target <target>` running every benchmark of a target. Metrics are
  task-native percentages and are never averaged across tasks.

### Commands

```bash
# Benchmark files (public, pinned revisions). mbpp_plus needs `pip install evalplus==0.3.1`.
python SFT/data/prepare_datasets.py --datasets ifeval ifbench math500 mbpp_plus

# Train pools and targets: see the layout above; a prepare step will follow.

# Train (Qwen3-1.7B-Base, batch 8, 4096 tokens, one epoch over the pool)
bash SFT/train/train.sh -c configs/dolci_inst_if -m all
bash SFT/train/train.sh -c configs/dolci_reason_math -m "FullTraining-Full,LayerWiseSubset-Full"

# Evaluate every run of a setting on its benchmarks
bash SFT/eval/eval.sh --train dolci_instruction --target precise_if --batch_size 16 \
    --ifbench_repo /path/to/IFBench
bash SFT/eval/eval.sh --train dolci_reasoning --target math --batch_size 16
bash SFT/eval/eval.sh --train dolci_reasoning --target mbpp --batch_size 16   # apptainer if available
```

Verifier dependencies: `langdetect`, `immutabledict`, `nltk` (IFEval, vendored
under `SFT/eval/tasks/ifeval_lib`), a checkout of
[`allenai/IFBench`](https://github.com/allenai/IFBench) at the pinned commit
(IFBench), `math-verify` (MATH500; boxed-answer fallback if missing), and
`evalplus` plus apptainer/singularity for a sandboxed MBPP+ run.

## Hyperparameters

Fixed across all settings. No LR tuning per setting.

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B` |
| LR (Full / MeSO) | `1e-5` |
| LR (LoRA) | `1e-4` |
| Scheduler | linear, `warmup_ratio=0.03` |
| Optimizer | AdamW (`weight_decay=0.0`) |
| Precision | bf16, flash-attention-2 |
| LoRA | `r=8`, `alpha=16`, `dropout=0.1`, `target_modules=all-linear` |
| Batch size | `per_device=8`, `gradient_accumulation=1` |
| Seq length | `max_seq_length=512` |
| Curation | `selection_frac=0.5`, `n_val=16`, `val_strategy=merged_batch`, `scoring.method=reduced_ghost` (LayerWiseSubset uses `compress` with `compression=normal-64*64`) |
| MeSO | optimizer `compression=normal-512*512` |
| Eval | `n_eval=500`, `n_test=500`, seeds {2, 22, 42, 62, 82} |

## Chat template

All examples are stored as `messages` JSONL (no template baked in).
Llama-3.2-1B-Base ships without a chat template, so we install an
open-instruct-style fallback (`<|user|>` / `<|assistant|>` plaintext
markers) via `SFT/data/chat_format.py:ensure_chat_template` (re-exported from
`get_val_dataset.py`). Tokenizers that ship a template (Qwen3) use their own.
Both training and eval call `tokenizer.apply_chat_template(...)`; loss is
computed only on the assistant-content tokens.

## Data preparation

```bash
# Eval splits (val/lr/test) for the 4 active target tasks
python SFT/data/prepare_datasets.py --datasets samsum tydiqa nq_open_eval squad_eval

# Training pools
python SFT/data/prepare_datasets.py --datasets alpaca triviaqa_train dolly oasst1 flan_v2 cot
```

`cot` (`kaist-ai/CoT-Collection`) is loaded via
`revision="refs/convert/parquet"` because the script form is rejected by
`datasets >= 3.0`.

| Dataset    | Role  | Lines (post-prep)        | Description                                         |
| ---------- | ----- | ------------------------ | --------------------------------------------------- |
| `samsum`   | eval  | 818 / 100 / 719          | Dialogue summarization (val/lr/test)                |
| `tydiqa`   | eval  | 100 / 100 / 4877         | Multilingual extractive QA (val/lr/test)            |
| `nq_open`  | eval  | val/lr/test from HF validation (~3.6K) | Closed-book factoid QA               |
| `squad`    | eval  | val/lr/test from HF validation         | Closed-book reading-comprehension QA |
| `alpaca`   | train | 52,002                   | Stanford Alpaca instruction-following               |
| `triviaqa` | train | ~138K                    | TriviaQA closed-book Q→A pairs (rc.nocontext)       |
| `flan_v2`  | train | 100,000 (subset)         | LESS-mix component                                  |
| `cot`      | train | 1,837,928                | LESS-mix component (CoT-Collection, parquet rev.)   |
| `dolly`    | train | 15,011                   | LESS-mix component                                  |
| `oasst1`   | train | 9,846                    | LESS-mix component (multi-turn unrolled)            |

## Methods (per setting)

| Config                  | Curation       | Finetuning |
|-------------------------|----------------|------------|
| `FullTraining-Full`     | none           | Full       |
| `FullTraining-LoRA`     | none           | LoRA r=8   |
| `FullTraining-MeSO`     | none           | MeSO       |
| `LayerWiseSubset-Full`  | per-layer top-k| Full       |
| `LayerWiseSubset-LoRA`  | per-layer top-k| LoRA r=8   |
| `LayerWiseSubset-MeSO`  | per-layer top-k| MeSO       |
| `GlobalSubset-Full`     | global top-k   | Full       |
| `GlobalSubset-LoRA`     | global top-k   | LoRA r=8   |
| `GlobalSubset-MeSO`     | global top-k   | MeSO       |

Setting 1 (`alpaca_samsum`) runs all 9; settings 2–4 run only the 3 LoRA
variants. Per-task target-only baselines (`FullTraining-{Full,LoRA,MeSO}`
via `train_val_ablation.sh`) train directly on the `n_val=16` task
validation samples.

> Run dirs: `{train}_{task}-{model}-{Method}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

## Submitting the full sweep

```bash
# 90 main + 30 target-only + 18 eval-main + 6 eval-target = 144 jobs
bash SFT/train/submit_all.sh             # submit
bash SFT/train/submit_all.sh --dry-run   # print sbatch commands only
```

Layout:
- Stage 1: 90 main training jobs (3h walltime)
- Stage 2: 30 target-only jobs (2h walltime)
- Stage 3: 18 main-eval jobs (2h, depends on Stage 1)
- Stage 4: 6 target-eval jobs (2h, depends on Stage 2)

## Single-job training

```bash
bash SFT/train/train.sh -c configs/<setting> -m all
bash SFT/train/train.sh -c configs/<setting> -m FullTraining-Full --seed 42
bash SFT/train/train.sh -c configs/<setting> --list
```

Categories: `all`, `full-training`, `layer-wise-subset`, `global-subset`,
`full`, `lora`, `meso`.

```bash
bash SFT/train/train_val_ablation.sh \
    --task <target_task> --config_dir <setting> \
    --methods FullTraining-Full --eval_steps <n> --seed <seed>
```

## Evaluation

```bash
# n_test=500 matches the during-training perplexity sample for direct comparison
bash SFT/eval/eval.sh --train <train> --task <task> --batch_size 64 --n_test 500
```

Supported tasks: `samsum`, `tydiqa`, `nq_open`, `squad`, `triviaqa`, plus the
Dolci benchmarks `ifeval`, `ifbench`, `math500`, `mbpp_plus` (see above).

`evaluate` and `rouge_score` Python packages must be installed in the
active env (`pip install evaluate rouge_score`).

## Config directory structure

Each config dir has `defaults.yaml` (shared) and one YAML per method:

```
configs/<setting>/
  defaults.yaml              # model, dataset, scheduler, etc.
  FullTraining-{Full,LoRA,MeSO}.yaml
  GlobalSubset-{Full,LoRA,MeSO}.yaml
  LayerWiseSubset-{Full,LoRA,MeSO}.yaml
```

`defaults.yaml`:
```yaml
model: meta-llama/Llama-3.2-1B
train_dataset: <pool>
target_task: <task>
percentage: <pct>

seed: 42
batch_size: 8
gradient_accumulation_steps: 1
optim: adamw_torch
max_seq_length: 512
lr_scheduler_type: linear
warmup_ratio: 0.03
weight_decay: 0.0
num_train_epochs: 1
eval_steps: <n>          # ~100 ppl points across max_steps
use_flash_attention: true

n_eval: 500
selection_frac: 0.5
selection_mode: topk
n_val: 16
val_batch_size: 1
val_strategy: merged_batch
scoring:
  method: reduced_ghost
```

Load order: defaults → `defaults.yaml` → method YAML → CLI (`--seed`, `--lr`).

#### Adding a new setting

1. Create `configs/<new_setting>/` with a `defaults.yaml`.
2. Copy method YAMLs (3 if LoRA-only, 9 if Full+LoRA+MeSO) — LRs are fixed (`1e-5` / `1e-4`).
3. Prep data: `python SFT/data/prepare_datasets.py --datasets <pool> <task>`.
4. Add the setting (and any new target task) to `submit_all.sh`.
