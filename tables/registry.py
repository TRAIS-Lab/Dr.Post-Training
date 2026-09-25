"""Registry of every table in the paper: which script generates it, where its experiments ran, where the results are.

Convention: every table that contains measured numbers is written by a generator script under
``<setting>/tables/`` that reads result files; the paper's Tables/*.tex are never edited by hand. The only
exceptions are tables with no measurements (``kind = "conceptual"``); ``kind = "shelved"`` marks a family whose generator is kept
but whose tables the paper no longer includes (skipped by the check and the orphan audit). ``python tables/make_all.py --check``
verifies every generator against the files on disk and reports registered tables whose generator is missing on
this checkout, unregistered files in Tables/, and registered files the paper no longer \\input{}s.

Machines: ``h200`` = the H200 Slurm cluster (8x H200 nodes; results under $SCRATCH_DIR/Dr.Post-Training),
``a40`` = the A40 Slurm cluster (one node, 4x A40; results under $SCRATCH_DIR/Dr.Post-Training) that ran every Llama-3.2-1B
question-answering and system-benchmark experiment (its benchmark JSONs live under SFT/benchmark/results/paper, gitignored; the QA tables are built from a local per-run export written by --scan),
``wandb`` = W&B project verl_grpo_math (the RLVR runs).
"""

FAMILIES = [
    # ------------------------------------------------------------------ SFT, Qwen3-1.7B-Base capability settings (h200)
    dict(script="SFT/tables/qwen_capability.py", machine="h200",
         results="$DRPT_RESULTS/SFT/runs_v2/<run>/{ifeval,ifbench,math500,gsm8k}_results.greedy.json",
         tables={"sft_qwen_main.tex": "tab:SFT-qwen",
                 "sft_qwen_nval_if.tex": "adxtab:sft-qwen-nval-if", "sft_qwen_nval_math.tex": "adxtab:sft-qwen-nval-math"}),
    # ------------------------------------------------------------------ SFT, Llama-3.2-1B question answering (a40)
    dict(script="SFT/tables/qa_downstream.py", machine="a40",
         results="$DRPT_RESULTS/SFT/runs_v2/<run>/{<target>_results.json, evaluation_results.json} (--scan) -> SFT/tables/data/qa_runs.csv (local, gitignored per-run export the tables are built from)",
         tables={"sft_downstream.tex": "tab:SFT-downstream", "sft_qa_ppl.tex": "adxtab:sft-qa-ppl",
                 "sft_qa_matrix_lora.tex": "adxtab:sft-qa-matrix-lora", "sft_qa_matrix_full.tex": "adxtab:sft-qa-matrix-full",
                 "sft_qa_matrix_meso.tex": "adxtab:sft-qa-matrix-meso",
                 "sft_ablation.tex": "adxtab:sft-ablation"}),
    # ------------------------------------------------------------------ system efficiency benchmarks (a40)
    dict(script="SFT/tables/system_efficiency.py", machine="a40",
         results="SFT/benchmark/results/paper/{breakdown,breakdown_checkpointing}/<tag>_n<n>_T<T>_m1.json, scoring/scoring_<tag>.json (local, gitignored)",
         tables={"system_overhead.tex": "tab:system-overhead", "score_cost.tex": "tab:score-cost", "peak_memory.tex": "adxtab:peak-memory",
                 **{f"timing_grid_{m}_n{n}.tex": "adxtab:timing-grid" for m in ("smollm2", "tinyllama", "llama3b") for n in (2, 8)},
                 **{f"timing_grid_ckpt_{m}_n{n}.tex": "adxtab:timing-grid-ckpt" for m in ("smollm2", "tinyllama", "llama3b") for n in (2, 8)}}),
    dict(script="SFT/tables/qa_timing.py", machine="a40", kind="shelved",   # not in the paper; generator kept
         results="SFT/benchmark/results/paper/qa_profile/<run>/profile.json + qa_throughput/<run>/evaluation_results.json (local, gitignored)",
         tables={"qa_timing_grid_full.tex": "adxtab:qa-timing-grid", "qa_timing_grid_lora.tex": "adxtab:qa-timing-grid",
                 "qa_timing_grid_meso.tex": "adxtab:qa-timing-grid"}),
    # ------------------------------------------------------------------ RLHF (h200)
    dict(script="RLHF/tables/scoring_backend.py", machine="h200",
         results="$DRPT_RESULTS/RLHF/<run>/rescore_final.json",
         tables={"rlhf_scoring_backend.tex": "adxtab:rlhf-scoring-backend"}),
    # ------------------------------------------------------------------ RLVR (wandb)
    dict(script="RLVR/tables/accuracy.py", machine="wandb",
         results="W&B project verl_grpo_math: runs Qwen3-1.7B_s<seed>, Qwen3-1.7B_Global_s<seed>_reward, Qwen3-1.7B_LayerWise_s<seed>_reward",
         tables={"rlvr_accuracy.tex": "adxtab:rlvr"}),
    # ------------------------------------------------------------------ conceptual tables: no measured numbers
    dict(script=None, machine=None, kind="conceptual", results="derived by hand from the method definitions (no experiment)",
         tables={"bias_exposure.tex": "tab:bias-exposure", "scoring_comparison.tex": "tab:scoring-comparison",
                 "experiment_overview.tex": "adxtab:experiment-overview",
                 "partition_groups.tex": "adxtab:partition-groups"}),
]


def by_table():
    out = {}
    for fam in FAMILIES:
        for tex, label in fam["tables"].items():
            out[tex] = dict(fam, label=label)
    return out
