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
drawn from the same source as D* (Dolci Precise-IF rows, Dolci math rows, MBPP
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

### Benchmark decontamination

Benchmarks (`eval/<bench>/<bench>_bench_data.jsonl`) are never loaded by
`train.py`; they are only read by `SFT/eval/eval.sh` after training.
`prepare_datasets.py` additionally drops any pool or target row whose normalised
user prompt exactly matches a benchmark prompt, or shares at least 80% of its
unique word 8-grams with one (the Dr.Post-Training-Next rule); the mbpp target
also excludes every MBPP+ task id. `--datasets dolci_audit` re-checks the files
on disk.

Shared boilerplate is *not* leakage and is deliberately kept: IFEval's constraint
templates ("your answer must contain a title wrapped in double angular
brackets") appear verbatim in Dolci Precise-IF prompts, and generic code/math
idioms share long n-grams with MBPP+/MATH500 solutions. Neither reveals a
benchmark task. What this cannot cover is the base model's pre-training data.

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
  task-native percentages and are never averaged across tasks. Decoding follows
  the Qwen3 non-thinking recommendation by default (`--temperature 0.7 --top_p 0.8
  --top_k 20`, seeded by `--seed`); `--temperature 0` gives greedy decoding. The
  settings are recorded under `generation` in every `<bench>_results.json`. A few
  percent of generations never stop, so `--batch_size 64 --max_new_tokens 2048`
  (the launcher default) is 5-8x faster than batch 16 with a 4096 cap at no cost
  in accuracy for these models.

### Commands

```bash
# Benchmark files (public, pinned revisions). mbpp_plus needs `pip install evalplus==0.3.1`.
python SFT/data/prepare_datasets.py --datasets ifeval ifbench math500 mbpp_plus

# mbpp target (D* 128 + held-out) from MBPP train minus MBPP+ task ids, and math_ref (MATH-train
# reference solutions; no longer a training target, but the pools were decontaminated against it).
# Both are decontaminated against the benchmark prompts, so build the benchmark files first.
python SFT/data/prepare_datasets.py --datasets math_ref mbpp

# Train pools (3 x 32,000 rows from allenai/Dolci-Instruct-SFT, pinned revision) + the precise_if target
# (Dolci Precise-IF rows, as in Dr.Post-Training-Next). Pools are uniform samples over domain groups:
# instruction = Chat/Precise IF/Other/Multilingual/Safety, reasoning = Math/Coding/Reasoning/Science,
# mixed = both. Tool-use and hardcoded rows, conversations that are not plain user/assistant chats, and
# rows over 16K characters are dropped. Every pool row is decontaminated against benchmark AND target
# prompts (exact normalised match or >= 80% shared word 8-grams, the Next rule), and the precise_if
# target rows are additionally removed by id.
python SFT/data/prepare_datasets.py --datasets dolci_pools --num_proc 16
python SFT/data/prepare_datasets.py --datasets dolci_audit   # re-check: must print CLEAN

# `math` target (config dirs `dolci_reason_math`, `dolci_mixed_math`): 128 D* + 500 held-out rows, half from the
# benchmarks' own held-out sets (MATH train via math_ref, 32/125; GSM8K train, 32/125, minus problems that occur
# in a pool) and half from the pools' four math sources (16 / 62-63 each: Tulu 3 Persona MATH, GSM, Algebra and
# the Dolci slice of OpenMathInstruct 2, rows that are in no pool). Rows are copied verbatim. Controls:
# `math_ref` (MATH train only, reference solutions), `math_pool` (pool-side sources only), `math_persona` (the
# three Persona sources only).
python SFT/data/build_math_pool_target.py --data_dir $SCRATCH_DIR/Dr.Post-Training/SFT/data --num_proc 16   # math_pool
python SFT/data/build_math_target.py --data_dir $SCRATCH_DIR/Dr.Post-Training/SFT/data                      # math (needs math_ref + math_pool)
python SFT/data/build_math_persona_target.py --data_dir $SCRATCH_DIR/Dr.Post-Training/SFT/data --num_proc 16

# Strong-model rewritten D* (variants `<target>_gen<tag>` / `<target>_rw<tag>`; SFT/data/gen_target_candidates.py +
# SFT/data/build_rewrite_target.py; verifiers in SFT/data/target_verifiers.py, ported from Next). Stage 1 samples 8
# candidate answers per D* row (2 per held-out row) from a generator model, non-thinking, T 0.7 / top-p 0.8 / top-k 20:
# `gen` = re-solved from the prompt alone (math prompts use the MATH500 boxed template), `rw` = the reference answer
# rewritten into the requested format (constraints satisfied exactly / step-by-step boxed solution). Stage 2 keeps, per
# row, the first (rw) or shortest (gen) candidate that passes the domain verifier: Math-Verify on the reference's final
# answer (numeric / normalised-string fallback for free-form persona answers), IFEval constraints recovered from the
# prompt (strict), MBPP asserts. D* rows with no verified candidate are dropped, held-out rows keep the reference, rows
# the verifier cannot check keep the generated answer flagged `unverifiable`; eval/<name>/manifest.json has the counts.
# eval.py maps the variants to the base target's benchmarks; config dirs dolci_inst_if_{gen,rw}32b and
# dolci_{mixed,reason}_mathpersona_{gen,rw}32b. Stage 1 can run in a vLLM env (only transformers is imported).
DATA=$SCRATCH_DIR/Dr.Post-Training/SFT/data
python SFT/data/gen_target_candidates.py --data_dir $DATA --generator <Qwen3-32B path> --backend vllm --tag 32b \
    --jobs precise_if:solve precise_if:rewrite math_persona:solve math_persona:rewrite --out_dir $DATA/candidates
python SFT/data/build_rewrite_target.py --data_dir $DATA --source_target precise_if --name precise_if_gen32b \
    --candidates $DATA/candidates/precise_if_gen32b.jsonl

# Train (Qwen3-1.7B-Base, batch 8, 2048 tokens, one epoch over the pool)
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

## Loss convention (`loss_reduction`)

drpt reads per-sample gradients off the backward pass of one batch loss, so
whatever that loss averages over is one "item" in every influence score and in
the curated update. `loss_reduction=sample_mean` (default since 2026-09-11) uses
the mean over examples of per-example token-mean losses, computed from the
logits (`drpt.losses.causal_lm_loss`): every example is one item, the score of
sample $b$ is $\langle\nabla\bar\ell_b, \nabla L_{\text{val}}\rangle$ with
$L_{\text{val}}$ the mean over the target batch, and the curated update is the
plain mean $\frac1k\sum_{b\in S}\nabla\bar\ell_b$, as in the paper. The
full-training baseline uses the same loss, so all arms share one objective.
`loss_reduction=token_mean` restores the Hugging Face token mean over the batch
(the convention of every run before the switch), where scores carry a factor of
the sample's response length and selected samples enter the update weighted by
length. Reported val/eval perplexities always use the model's token-mean loss.
`tests/test_loss_convention.py` pins both conventions.

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
| `BlockWiseSubset-Full`  | per-block top-k (Dolci settings)    | Full |
| `SublayerWiseSubset-Full` | per attention/MLP sub-block top-k (Dolci settings) | Full |

Setting 1 (`alpaca_samsum`) runs all 9; settings 2–4 run only the 3 LoRA
variants. The Dolci settings additionally run the two group-wise variants
(see "Group-wise curation" below). Per-task target-only baselines (`FullTraining-{Full,LoRA,MeSO}`
via `train_val_ablation.sh`) train directly on the `n_val=16` task
validation samples.

> Run dirs: `{train}_{task}-{model}-{Method}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

## Submitting the full sweep

`submit_all.sh` writes one manifest per stage and submits each stage as a
single Slurm **job array** (never a loop of `sbatch` calls, which trips the
controller's per-user RPC limit). Your `submit.sh` must honour the
`ARRAY`/`MANIFEST`/`DEPEND` knobs described in the top-level README.

```bash
# Paper suite (Llama-3.2-1B): 90 main + 30 target-only + 18 eval-main + 6 eval-target = 144 tasks, 4 arrays
bash SFT/train/submit_all.sh --suite paper
# Dolci suite (Qwen3-1.7B-Base): 5 settings x 3 Full methods per seed, + one benchmark eval per (setting, method)
bash SFT/train/submit_all.sh --suite dolci --seeds 42
bash SFT/train/submit_all.sh --suite dolci --settings dolci_inst_if --stages 1   # subset / single stage
bash SFT/train/submit_all.sh --suite paper --dry-run                             # manifests + sbatch lines only
```

Options: `--seeds "2 22 42 62 82"`, `--stages 1,2,3,4`, `--settings a,b`,
`--max-concurrent K` (array `%K`, default 8). `QOS` defaults to `high`
because runs do not checkpoint, so a preempted `low` job restarts from scratch.

Stages (paper suite walltimes; Dolci main runs get 16h and evals 8h):
- Stage 1: main training array (3h)
- Stage 2: target-only array (2h; paper suite only)
- Stage 3: eval-main array (2h, `afterok` Stage 1)
- Stage 4: eval-target array (2h, `afterok` Stage 2)

## Single-job training

```bash
bash SFT/train/train.sh -c configs/<setting> -m all
bash SFT/train/train.sh -c configs/<setting> -m FullTraining-Full --seed 42
bash SFT/train/train.sh -c configs/<setting> --list
```

Categories: `all`, `full-training`, `layer-wise-subset`, `global-subset`,
`block-wise-subset`, `sublayer-wise-subset`, `group-wise-subset`, `full`,
`lora`, `meso`.

### Group-wise curation (`GroupWiseSubset`)

`LayerWiseSubset` selects a subset per hooked Linear layer and `GlobalSubset`
selects one subset for the whole model. `GroupWiseSubset` selects per *group*
of hooked layers for any partition in between: scores are accumulated over the
group's layers during backward and, as soon as the last layer of the group has
run, the group selects its samples and assembles the curated gradient for
exactly those layers (single backward, no extra FLOPs; only one group's
activations are retained at a time when groups are contiguous in backward
order, which they are for the presets below).

| YAML | Groups |
|---|---|
| `method: BlockWiseSubset` | one group per decoder block `model.layers.N` (alias for `GroupWiseSubset` + `selection_granularity: block`) |
| `method: SublayerWiseSubset` | two groups per block: `self_attn.*` = {q,k,v,o} and `mlp.*` = {gate,up,down} (alias for `selection_granularity: sublayer`) |
| `method: GroupWiseSubset` + `selection_granularity: layer \| sublayer \| block \| global` | presets; `layer` == `LayerWiseSubset`, `global` == one-pass `GlobalSubset` |
| `method: GroupWiseSubset` + `selection_groups: <rules>` | custom per-block groups |

Custom rules are `"<name>=<member>[,<member>..];<name>=.."`; a member matches a
layer when its dot-separated components appear contiguously in the layer name
after `model.layers.N.` (PEFT names such as `self_attn.q_proj.lora_A.default`
match `q_proj`). Unmatched layers stay singletons; `embed_tokens` and `lm_head`
are always singletons except under `global`. Example
(`configs/dolci_reason_math/GroupWiseSubset-Full-custom4.yaml`, four groups per block):

```yaml
method: GroupWiseSubset
selection_groups: attn.qkv=q_proj,k_proj,v_proj;attn.o=o_proj;mlp.gateup=gate_proj,up_proj;mlp.down=down_proj
```

The value must not contain `:` or quotes (it goes through the line-based YAML
parser in `train.sh`). With `record_selections: true` the run's
`selection_records.json` stores one entry per group (`group`, `layer_indices`,
`selected_indices`, `scores`) and the metadata lists `layer_groups`, the group
key of every hooked layer. `tests/test_groupwise_selection.py` checks that
`selection_frac=1.0` reproduces plain training, `layer` == `LayerWiseSubset`
and `global` == `GlobalSubset` exactly (CPU, tiny model).

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
  BlockWiseSubset-Full.yaml           # Dolci settings only (group-wise curation)
  SublayerWiseSubset-Full.yaml        # Dolci settings only
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

Load order: defaults → `defaults.yaml` → method YAML → CLI (`--seed`, `--lr`, `--n_val`).

#### Adding a new setting

1. Create `configs/<new_setting>/` with a `defaults.yaml`.
2. Copy method YAMLs (3 if LoRA-only, 9 if Full+LoRA+MeSO) — LRs are fixed (`1e-5` / `1e-4`).
3. Prep data: `python SFT/data/prepare_datasets.py --datasets <pool> <task>`.
4. Add the setting (and any new target task) to `submit_all.sh`.
