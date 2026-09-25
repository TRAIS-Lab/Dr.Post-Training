# Paper tables: one generator per table family, nothing typed by hand

Every table in the paper that contains measured numbers is written by a script that reads result files. The
scripts live next to the experiments that produced the numbers, one per table family:

| Family | Generator | Runs on | Reads |
|---|---|---|---|
| Qwen3-1.7B-Base SFT (main + appendix k / target-size tables) | `SFT/tables/qwen_capability.py` | H200 cluster | `$DRPT_RESULTS/SFT/runs_v2/<run>/*_results.greedy.json` |
| Llama-3.2-1B QA (Table 1, perplexity, QA matrices, controls) | `SFT/tables/qa_downstream.py` | A40 cluster | `$DRPT_RESULTS/SFT/runs_v2/<run>/{<target>_results.json, evaluation_results.json}` with `--scan`; otherwise the local (gitignored) per-run export `SFT/tables/data/qa_runs.csv` |
| System-efficiency benchmarks (overhead, score cost, memory, timing grids) | `SFT/tables/system_efficiency.py` | A40 cluster | `SFT/benchmark/results/paper/{breakdown,breakdown_checkpointing,scoring}/*.json` (local, gitignored) |
| Per-component timing of real QA steps | `SFT/tables/qa_timing.py` | A40 cluster | `SFT/benchmark/results/paper/{qa_profile,qa_throughput}/<run>/` (local, gitignored) |
| RLHF final-adapter toxicity under four judges, exact vs compressed | `RLHF/tables/scoring_backend.py` | H200 cluster | `$DRPT_RESULTS/RLHF/<run>/rescore_final.json` |
| RLVR accuracy at selected rounds | `RLVR/tables/accuracy.py` | W&B (`verl_grpo_math`) | read directly from W&B (5 seeds per method, matched by run name) |

The three tables without measurements (`bias_exposure.tex`, `scoring_comparison.tex`, `experiment_overview.tex`) are derived from
the method definitions and the experiment design and are the only hand-maintained bodies. `tables/registry.py` is the authoritative list; it maps every
`Tables/*.tex` to its family, machine and result files.

## Conventions

* **Header.** Every generator starts with a docstring of the form
  ```
  Paper table(s): <file>.tex (<label>), ...
  Experiments run on : <machine / cluster>
  Launcher           : <script + config dir + arms that produced the runs>
  Results read from  : <path pattern under the results root>
  Seeds / statistics : <seeds>; mean +- SE (population std / sqrt n); ...
  Usage              : python <setting>/tables/<family>.py [--results-root DIR] [--paper-root DIR] [--venues ICLR] [--check]
  ```
* **Inputs.** Generators read only from the results root (`--results-root`, else `$DRPT_RESULTS`, else
  `$SCRATCH_DIR/Dr.Post-Training`, with `SCRATCH_DIR` taken from the repo's `cluster_env.sh` if not exported).
  No absolute machine paths in the scripts.
* **Outputs.** `<paper root>/<venue>/Tables/<file>.tex` (`--paper-root`, else `$DRPT_PAPER`, else `<repo>/Paper`).
  The first line of every file is a provenance comment (script, date, results root). The body is the bare
  `tabular` (plus `adjustbox` where the paper expects it); captions and labels stay in the `.tex` sources.
* **Statistics.** Mean over seeds; SE = population std / sqrt(n); cells `\(m\,{\scriptstyle\pm se}\)` at one
  decimal unless the table states otherwise; best per column bold, ties at the displayed precision all bold;
  a cell with fewer seeds than expected carries `(n=k)`; missing runs are never invented.
* **Verification.** `--check` regenerates in memory and diffs against the file on disk, comment lines ignored.
  `python tables/make_all.py --check` runs every generator present on the checkout, lists generators that are
  registered but missing here (their results live on another machine), and audits `Tables/` for unregistered or
  orphaned files. It exits non-zero on any diff or audit finding. Run it before syncing the paper.
* **Shared code.** `tables/common.py` (paths, statistics, cell formatting, `emit`). Import it with
  `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`.

## Where a generator can run

The A40 families read local, gitignored files under the repo (the benchmark JSONs in `SFT/benchmark/results/paper/`; the QA per-run
export `SFT/tables/data/qa_runs.csv`, refreshed with `python SFT/tables/qa_downstream.py --scan` on the A40 checkout after new runs
finish), so they verify on a checkout that has those files.
The H200 families need `$DRPT_RESULTS` with the Qwen / RLHF run dirs; on another machine they report `[DIFF]` (empty tables), so scope
the check there: `python tables/make_all.py --check --only SFT/tables/qa`, `--only system`, `--only RLVR`. The audit of `Tables/`
runs in every case.

## Figures

Figures come from `SFT/result.ipynb`, `RLHF/result.ipynb` and `RLVR/result.ipynb` (run all cells; each writes into
`<paper root>/<venue>/Figures/`), and the case-study panels from `SFT/case_study/make_figures.py`, which
has the same header block as the table generators and reads `<results root>/SFT/runs_v2/case_study/<run>/selection_records.json`, writing
`<paper root>/<venue>/Figures/SFT/case_study/`.
