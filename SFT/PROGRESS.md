# SFT Experiments — Progress Tracker

Living document for the unified SFT experiment plan. Update entries as work progresses.

## Final scope (5 settings, Full-FT only, Llama-3.2-1B)

| # | Config dir | Train pool | Target task | Eval format | Pool size | `percentage` | Step budget | `eval_steps` |
|---|------------|------------|-------------|-------------|-----------|--------------|-------------|--------------|
| 1 | `alpaca_samsum`  | alpaca       | samsum   | summarization (ROUGE) | 52K   | 0.4   | 2600 | 26 |
| 2 | `tulu3_tydiqa`   | tulu3        | tydiqa   | multilingual QA (F1)  | 939K  | 0.01  | 1174 | 12 |
| 3 | `less_tydiqa`    | less (mix)   | tydiqa   | multilingual QA (F1)  | 1.96M | 0.005 | 1225 | 12 |
| 4 | `triviaqa_nq`    | triviaqa     | nq_open  | closed-book QA (EM/F1) | 138K | 0.064 | 1107 | 11 |
| 5 | `nq_squad`       | nq_open      | squad    | closed-book QA, no context (EM/F1) | 88K | 0.1 | 1100 | 11 |

LESS-mix = `flan_v2 + cot + dolly + oasst1` (~1.96M).

### Settings rationale

- **Settings 1, 2**: legacy (instruction → summarization, general SFT → multilingual QA). Both **DONE** (5 seeds × 9 methods).
- **Setting 3**: contrast train-pool style for tydiqa target (LESS-mix vs Tulu3).
- **Setting 4**: QA→QA proof point. TriviaQA train (clean primary answers, no alias-noise issue) → NQ-open eval (clean alias structure).
- **Setting 5**: closed-book SQuAD eval — same Q→A format as NQ-open but questions are typically context-dependent. Performance expected to be lower; tests transfer of NQ-style training to a different question distribution.

### Dropped (and why)

- `nq_triviaqa`, `nq_triviaqa_qwen3`, `squad_triviaqa` — TriviaQA as TARGET. TriviaQA's eval-time alias list is heavily noisy (median 10 aliases, includes wrong items like "I'm not a crook" for Nixon questions); inflates EM/F1 variance. SQuAD provides cleaner closed-book QA eval (median 1 alias).
- `less_mmlu`, MMLU as task — subject plumbing complexity dropped.
- `tulu3_triviaqa` — overlap with `nq_triviaqa` (which also dropped).
- `nq_open → tydiqa` — broken (NQ avg seq_len ~30 too short for tydiqa val seq-length filter).
- `less → samsum` — alpaca_samsum already covers samsum target.

## Unified hyperparameters

Already shared across all settings:
- `model: meta-llama/Llama-3.2-1B`, `batch_size=8`, `max_seq_length=512`
- `optim=adamw_torch`, `lr_scheduler=linear`, `warmup_ratio=0.03`, `weight_decay=0.0`
- Curation: `n_val=16`, `selection_frac=0.5`, `selection_mode=topk`,
  `val_batch_size=1`, `val_strategy=merged_batch`, `scoring.method=reduced_ghost`
- Seeds: {2, 22, 42, 62, 82}

Standardized:
- `eval_steps`: per-setting so each run gets ~100 ppl points.
- `n_eval = 500` (perplexity-during-training pool, drawn from `test` split).
- **`n_test = 500` (final task metric, same first-500 examples of `test` split → same set as `n_eval`).**

## Methods (Full-FT only, 3 + Target-only baseline)

- `FullTraining-Full` — baseline, no curation
- `GlobalSubset-Full` — global top-k curation
- `LayerWiseSubset-Full` — per-layer top-k curation
- Target-only (`FullTraining-Full` via `train_val_ablation.sh`) — trained directly on n_val=16 task validation samples; lower bound

## Reuse vs re-run matrix (Full-FT)

| Setting | Asset | State | Action |
|---------|-------|-------|--------|
| 1 alpaca_samsum | Main 9 methods × 5 seeds | ✓ done at 103 ppl points | **REUSE training**, re-run task eval at `--n_test 500` |
| 1 alpaca_samsum | Target-only-Full × 5 seeds | ✗ stale (eval_steps=400 → 14 points) | **RE-RUN** at eval_steps=26 (LR=6.95e-07 verified) |
| 2 tulu3_tydiqa  | Main 9 methods × 5 seeds | ✓ done at 99 ppl points | **REUSE training**, re-run task eval at `--n_test 500` |
| 2 tulu3_tydiqa  | Target-only-Full × 5 seeds | ✗ stale (eval_steps=400 → 6 points) | **RE-RUN** at eval_steps=12 (LR=4.83e-06) |
| 3 less_tydiqa   | All | LR sweep in flight | After collect: 15 main + 5 target-only |
| 4 triviaqa_nq   | All | LR sweep in flight | After collect: 15 main + 5 target-only |
| 5 nq_squad      | All | LR sweep in flight | After collect: 15 main + 5 target-only |

## Phase 1 (no compute) — DONE

- [x] Cancel obsolete LoRA/MeSO LR sweep jobs
- [x] Patch `train_val_ablation.sh` to accept `--eval_steps`
- [x] Update `result.ipynb` to current scope (cells 2, 4, 5)
- [x] Scaffold `nq_triviaqa/`, `less_tydiqa/`, `triviaqa_nq/`, `nq_squad/`
- [x] Add `nq_open` and `squad` as eval tasks (data prep + task code)
- [x] Drop TriviaQA-target settings (`nq_triviaqa`, `nq_triviaqa_qwen3`, `squad_triviaqa`)
- [x] Remove `subject` parameter throughout (MMLU-specific, no longer needed)
- [x] Trim codebase: `prepare_datasets.py` 1751→843, `get_val_dataset.py` 970→453, drop unused tasks (mmlu/bbh/gsm8k/math500/truthfulqa/hhrlhf/arc) and pools (gsm8k/vicuna/wizardlm/openhermes/hhrlhf)
- [x] Delete obsolete eval shell scripts (8 of them)
- [x] Update README to current scope

## Phase 2 (compute) — IN FLIGHT

### LR sweeps (Full-FT only) — current submission

| Step | Status | Job count |
|------|--------|-----------|
| Setting 3 less_tydiqa main + Target-only-Full | submitted (in flight) | 60 + 20 |
| Setting 4 triviaqa_nq main + Target-only-Full | submitted (in flight) | 60 + 20 |
| Setting 5 nq_squad main + Target-only-Full | submitted (in flight) | 60 + 20 |
| Setting 1 alpaca_samsum Target-only-Full | NOT NEEDED (LR=6.95e-07 verified) | 0 |
| Setting 2 tulu3_tydiqa Target-only-Full | REUSE existing LR (4.83e-06) | 0 |

Currently ~321 jobs in queue (some are leftover from cancelled-but-not-killed dropped settings — wasted compute, indistinguishable by job name).

### Cancelled
- Setting `nq_triviaqa_qwen3` Target-only sweep (20 jobs cancelled by name)
- Setting `squad_triviaqa` Target-only sweep (20 jobs cancelled by name)
- Main sweeps for these dropped settings still running because indistinguishable from kept-setting main sweeps (60+60 = 120 wasted jobs, will finish naturally)

### Next compute (after LR sweeps complete)

1. **Collect LRs:**
   - `lr_sweep_collect.sh -c configs/{less_tydiqa,triviaqa_nq,nq_squad} -m all`
   - `lr_sweep_collect_val.sh --task tydiqa --max_steps 1225 --batch_size 8 --out_file SFT/train/configs/less_tydiqa/Target-only.lr.txt`
   - `lr_sweep_collect_val.sh --task nq_open --max_steps 1100 --batch_size 8 --out_file SFT/train/configs/triviaqa_nq/Target-only.lr.txt`
   - `lr_sweep_collect_val.sh --task squad --max_steps 1100 --batch_size 8 --out_file SFT/train/configs/nq_squad/Target-only.lr.txt`

2. **Submit main training:**
   - Settings 3, 4, 5: 5 seeds × 3 methods × 3 settings = **45 runs**

3. **Submit Target-only-Full training:**
   - Settings 1, 2 (re-run for cadence): 5 seeds × 2 = 10
   - Settings 3, 4, 5 (new): 5 seeds × 3 = 15
   - Total: **25 runs**

4. **Re-eval task metrics at `--n_test 500`:**
   - Settings 1, 2 main checkpoints (existing): re-run `eval.sh --n_test 500`
   - Settings 3, 4, 5: as part of new training pipeline

### Total Phase 2 compute (after sweeps)
- 45 main training runs
- 25 target-only training runs  
- Re-eval: trivial

## Key file changes (Phase 1)

- `SFT/train/train_val_ablation.sh`: added `--eval_steps`, default `batch_size=8`, removed `--subject`, trimmed task maps to {tydiqa, samsum, nq_open, squad}
- `SFT/train/configs/`: 5 active dirs (`alpaca_samsum`, `tulu3_tydiqa`, `less_tydiqa`, `triviaqa_nq`, `nq_squad`); 4 stale dirs deleted
- `SFT/result.ipynb`: cells 2, 4, 5 rewritten for unified 5-setting scope (per-setting model, ds_key=config, target-only LR file lookup)
- `SFT/data/prepare_datasets.py`: trimmed; added `prepare_nq_open_eval`, `prepare_triviaqa_train`, `prepare_squad`, `prepare_squad_eval`; removed mmlu/bbh/gsm8k/math500/truthfulqa/hhrlhf/arc/vicuna/wizardlm/openhermes
- `SFT/data/get_val_dataset.py`: trimmed; loaders only for {tydiqa, samsum, nq_open, squad}; subject param removed
- `SFT/data/get_train_dataset.py`: trimmed mappings (kept alpaca, tulu3, nq_open, triviaqa, squad, samsum, less components)
- `SFT/eval/tasks/`: only {samsum, tydiqa, nq_open, squad}; deleted mmlu, triviaqa
- `SFT/eval/eval.py`: dispatcher only for the 4 tasks above
- `SFT/eval/`: deleted 8 obsolete eval shell scripts; only `eval.sh` remains
- `SFT/README.md`: rewritten for current scope

## Open decisions

None at present — current 5-setting scope is the active plan.
