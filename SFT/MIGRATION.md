# SFT migration notes

State as of 2026-05-04 — switching from LR-sweep workflow to fixed-LR strategy with shuffled eval data.

## 5 SFT settings

`Llama-3.2-1B`, `bs=8`, `1 epoch`, `n_val=16`, `n_test=500`, seeds {2, 22, 42, 62, 82}.

| # | Config dir | Source → Target | Methods |
|---|---|---|---|
| 1 | `alpaca_samsum` | alpaca → samsum | 9 (FullTraining/LayerWiseSubset/GlobalSubset × Full/LoRA/MeSO) |
| 2 | `tulu3_tydiqa` | tulu3 → tydiqa | 9 |
| 3 | `less_tydiqa` | less → tydiqa | 9 |
| 4 | `triviaqa_nq` | triviaqa → nq_open | 3 (Full FT only) |
| 5 | `nq_squad` | nq_open → squad | 3 (Full FT only) |

## Fixed LRs (no sweep)

| FT type | LR | Citation |
|---|---|---|
| Full | 2e-5 | LESS [base_training_args.sh](https://github.com/princeton-nlp/LESS/blob/main/less/scripts/train/base_training_args.sh); Tulu-3 ([2411.15124](https://arxiv.org/abs/2411.15124)) used 5e-6 for 8B but Tulu-2 used 2e-5 for smaller models |
| MeSO | 2e-5 | Mirror Full FT (no canonical reference); empirical sweeps tracked Full FT |
| LoRA | 5e-4 | Adjusted for our α=1, r=32 (effective ≈ LESS's 2e-5 × α/r=4) |

Defined in: `SFT/train/configs/{setting}/*.yaml`, `Target-only.{,.LoRA.,.MeSO.}lr.txt`, fallbacks in `train.sh:319-321` and `train_val_ablation.sh:51-52`.

## Critical code changes

### Eval shuffling (was unbiased before — SQuAD test = all "Super Bowl 50", TyDiQA test = all Arabic)
`SFT/data/prepare_datasets.py`:
```python
EVAL_SHUFFLE_SEED = 42
def shuffled_examples(examples, seed=EVAL_SHUFFLE_SEED):
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    return examples
```
Applied in every `prepare_*_eval(...)` before slicing val/lr/test.

### 2× eval inflation fix
`SFT/train/trainer.py:~592` — custom `evaluate()` was missing the callback chain:
```python
if hasattr(self, 'callback_handler') and hasattr(self, 'state') and hasattr(self, 'control'):
    self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_metrics)
```

### Plot inversion fix
`SFT/result.ipynb`: `y_clip_max` for triviaqa_nq, nq_squad fixed `8.0 → 100.0` (data range was 12-100; matplotlib auto-inverted axis).

### Cleanup (dropped tasks)
- Dropped: mmlu, bbh, gsm8k, math500, truthfulqa, hhrlhf, arc, vicuna, wizardlm, openhermes
- Removed `--subject` parameter throughout
- `prepare_datasets.py`: 1751 → 843 lines; `get_val_dataset.py`: 970 → 453 lines

## Submission

`SFT/train/submit_all.sh` — submits 246 jobs in 4 stages with `afterok:` deps:

| Stage | Jobs | Walltime | Depends on |
|---|---|---|---|
| 1. Main training | 165 | 4h | — |
| 2. Target-only | 40 | 2h | — |
| 3. Eval main | 33 | 1h | Stage 1 |
| 4. Eval target | 8 | 1h | Stage 2 |

Run: `bash SFT/train/submit_all.sh` (try `--dry-run` first). Target-only is **deduplicated by task** (e.g. tulu3_tydiqa + less_tydiqa share tydiqa target-only baselines).

## Migrating to a new cluster

1. **Pull this branch** on the new cluster.
2. **Rewrite `cluster_env.sh`** for the new cluster — set `SCRATCH_DIR`, `SLURM_ACCOUNT`, `SLURM_PARTITION`, `activate_env()`. Current values:
   - `SCRATCH_DIR=/work/hdd/bfwm/phu1/Project`
   - `SLURM_PARTITION=gpuA40x4`
   - `CODE_DIR=/u/phu1/Project`
3. **Regenerate data**: `bash SFT/data/prepare_datasets.py` (uses HF cache; cheap).
4. **Submit fresh jobs**: `bash SFT/train/submit_all.sh`
5. **Don't re-run on old cluster** — output dirs cleaned to just `data/` (5.2G); fresh state.

## Key file paths

| File | Purpose |
|---|---|
| `SFT/train/submit_all.sh` | Master submission script (NEW) |
| `SFT/train/train.sh` | Main training entry |
| `SFT/train/train_val_ablation.sh` | Target-only training |
| `SFT/train/configs/{setting}/` | All 5 setting configs |
| `SFT/data/prepare_datasets.py` | Data prep (regenerate per cluster) |
| `SFT/eval/eval.sh` + `eval.py` | Evaluation |
| `SFT/eval/tasks/{nq_open,squad,samsum,tydiqa,triviaqa}.py` | Per-task eval logic |
| `SFT/result.ipynb` | Plotting notebook |
| `SFT/PROGRESS.md` | Progress tracking doc |
