#!/bin/bash
# Submit benchmark jobs to Slurm (Delta cluster).
#
# Benchmarks:
#   breakdown        — Per-component timing, no checkpointing (78 runs)
#   checkpointing    — Same, with gradient checkpointing (78 runs)
#   scoring          — Standalone scoring comparison (3 runs)
#   aggregate        — Generate result tables (no GPU needed)
#   all              — All of the above
#
# Each benchmark runs 3 models in parallel on GPUs 0-2 of a single A40x4 node.
#
# Usage:
#   bash SFT/benchmark/slurm/launch_all.sh                 # all benchmarks
#   bash SFT/benchmark/slurm/launch_all.sh breakdown        # breakdown only
#   bash SFT/benchmark/slurm/launch_all.sh scoring           # scoring only
#   bash SFT/benchmark/slurm/launch_all.sh --dry-run         # preview only
#   bash SFT/benchmark/slurm/launch_all.sh --dry-run scoring # preview scoring

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── Cluster config ──────────────────────────────────────────────────────────
source "$REPO_ROOT/cluster_env.sh"

ACCOUNT="${SLURM_ACCOUNT:-bfwm-delta-gpu}"
PARTITION="${SLURM_PARTITION:-gpuA40x4}"
CONDA_ENV="/u/phu1/.conda/envs/IF/bin"

CONFIGS_DIR="SFT/benchmark/slurm/configs"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── Args ────────────────────────────────────────────────────────────────────
DRY_RUN=false
MODE="all"
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        breakdown|checkpointing|scoring|aggregate|all) MODE="$arg" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "========================================"
echo "  Dr.Post-Training Benchmark Suite"
echo "  Mode: $MODE  Partition: $PARTITION"
echo "========================================"
echo ""

# ── Common preamble for all jobs ────────────────────────────────────────────
preamble() {
    cat <<PREAMBLE
cd $REPO_ROOT
export PATH="$CONDA_ENV:\$PATH"
export PYTHONPATH="$REPO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
PREAMBLE
}

# ── Submit helper ───────────────────────────────────────────────────────────
submit() {
    local name="$1" ngpus="$2" mem="$3" walltime="$4" script="$5"

    echo "  Job:       $name"
    echo "  Resources: ${ngpus} GPU(s), $mem RAM, $walltime"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would submit"
        echo ""
        return 0
    fi

    local gpu_args=()
    local part="$PARTITION"
    if [[ "$ngpus" -gt 0 ]]; then
        gpu_args=(--gres="gpu:$ngpus")
    else
        part="cpu"
    fi

    sbatch \
        --job-name="$name" \
        --account="$ACCOUNT" \
        --partition="$part" \
        --nodes=1 --ntasks=1 "${gpu_args[@]}" \
        --mem="$mem" --cpus-per-task=16 \
        --time="$walltime" \
        --output="$LOG_DIR/${name}_%j.log" \
        --error="$LOG_DIR/${name}_%j.log" \
        --mail-type=END,FAIL \
        --mail-user="${SLURM_MAIL_USER:-}" \
        --wrap="$(preamble)
$script"
    echo ""
}

# ── Benchmark: Breakdown ────────────────────────────────────────────────────
submit_breakdown() {
    local ckpt_flag="${1:-}"  # "" or "--gradient-checkpointing"
    local suffix="breakdown"
    local job_prefix="bench"
    local walltime="1:00:00"
    if [[ -n "$ckpt_flag" ]]; then
        suffix="breakdown_checkpointing"
        job_prefix="bench-ckpt"
        walltime="1:00:00"
    fi
    local results_dir="SFT/benchmark/results/$suffix"

    echo "--- $suffix (3 models x 2 configs x 13 combos = 78 runs) ---"

    submit "$job_prefix" 3 "192G" "$walltime" "
mkdir -p $results_dir

run_model() {
    local gpu=\$1 model=\$2 tag=\$3 configs_file=\$4
    python3 -c '
import json, subprocess, sys, os, time
model, tag, ckpt = \"'\$model'\", \"'\$tag'\", \"$ckpt_flag\"
with open(\"'\$configs_file'\") as f:
    configs = json.load(f)
for cfg in configs:
    n, T, m = cfg[\"n\"], cfg[\"T\"], cfg[\"m\"]
    out = os.path.join(\"$REPO_ROOT/$results_dir\", f\"{tag}_n{n}_T{T}_m{m}.json\")
    if os.path.exists(out):
        print(f\"SKIP {tag} n={n} T={T} m={m}\", flush=True); continue
    print(f\"Running {tag} n={n} T={T} m={m}...\", flush=True)
    t0 = time.time()
    cmd = [sys.executable, \"SFT/benchmark/benchmark_run.py\",
        \"--gpu\", \"'\$gpu'\", \"--model\", model,
        \"--batch-size\", str(n), \"--seq-length\", str(T), \"--val-batch-size\", str(m),
        \"--direct-batch-size\", \"1\", \"--output\", out]
    if ckpt:
        cmd.append(ckpt)
    r = subprocess.run(cmd)
    status = \"Done\" if r.returncode == 0 else \"FAILED\"
    print(f\"  {status} ({time.time()-t0:.0f}s)\", flush=True)
print(f\"{tag} complete.\", flush=True)
'
}

run_model 0 'Qwen/Qwen3-0.6B'        'qwen3-0.6b'   '$REPO_ROOT/$CONFIGS_DIR/qwen3-0.6b.json'   &
run_model 1 'Qwen/Qwen3-1.7B'        'qwen3-1.7b'    '$REPO_ROOT/$CONFIGS_DIR/qwen3-1.7b.json'   &
run_model 2 'meta-llama/Llama-3.2-3B' 'llama-3.2-3b'  '$REPO_ROOT/$CONFIGS_DIR/llama-3.2-3b.json' &
wait
echo '$suffix complete.'
"
}

# ── Benchmark: Scoring ──────────────────────────────────────────────────────
submit_scoring() {
    local results_dir="SFT/benchmark/results/scoring"

    echo "--- scoring (3 models, standalone) ---"

    submit "bench-scoring" 3 "64G" "0:30:00" "
mkdir -p $results_dir

python3 SFT/benchmark/benchmark_scoring.py \
    --gpu 0 --model-tag qwen3-0.6b \
    --output $results_dir/scoring_qwen3-0.6b.json &

python3 SFT/benchmark/benchmark_scoring.py \
    --gpu 1 --model-tag qwen3-1.7b \
    --output $results_dir/scoring_qwen3-1.7b.json &

python3 SFT/benchmark/benchmark_scoring.py \
    --gpu 2 --model-tag llama-3.2-3b \
    --output $results_dir/scoring_llama-3.2-3b.json &

wait
echo 'Scoring complete.'
"
}

# ── Aggregate results (CPU-only) ───────────────────────────────────────────
submit_aggregate() {
    echo "--- aggregate (generate tables) ---"

    submit "bench-aggregate" 0 "8G" "0:10:00" "
for dir in breakdown breakdown_checkpointing; do
    results_dir=SFT/benchmark/results/\$dir
    [[ -d \$results_dir ]] || continue
    for scoring in compress full_ghost reduced_ghost direct; do
        python3 SFT/benchmark/aggregate_breakdown.py \\
            --results-dir \$results_dir \\
            --output \$results_dir/\${dir}_\${scoring}.txt \\
            --scoring \$scoring
    done
    echo \"Tables written to \$results_dir/\"
done
echo 'Aggregation complete.'
"
}

# ── Dispatch ────────────────────────────────────────────────────────────────
case "$MODE" in
    breakdown)
        submit_breakdown ""
        ;;
    checkpointing)
        submit_breakdown "--gradient-checkpointing"
        ;;
    scoring)
        submit_scoring
        ;;
    aggregate)
        submit_aggregate
        ;;
    all)
        submit_breakdown ""
        submit_breakdown "--gradient-checkpointing"
        submit_scoring
        # Aggregate must run after the others finish — submit manually or
        # use --dependency. Print a reminder.
        echo "NOTE: Run aggregation after benchmark jobs complete:"
        echo "  bash SFT/benchmark/slurm/launch_all.sh aggregate"
        ;;
esac

echo "========================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "  Dry run — no jobs submitted."
else
    echo "  Jobs submitted. Monitor with: squeue -u \$USER"
fi
echo "  Logs: $LOG_DIR/"
echo "========================================"
