#!/bin/bash
# Submit Qwen3 H200 benchmark suite (1-GPU jobs, single-model per job).
#
# Models: Qwen3-1.7B, Qwen3-4B, Qwen3-8B
# Configs (per qwen3-*.json): n×T = (8,1024), (4,2048), (2,4096), (16,512)
# Variants: breakdown (no-ckpt), breakdown_checkpointing
# Total: 3 models × 4 configs × 2 variants = 24 jobs + 3 scoring jobs = 27 jobs
#
# Usage:
#   bash SFT/benchmark/slurm/launch_h200.sh                # submit
#   bash SFT/benchmark/slurm/launch_h200.sh --dry-run       # preview

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }

ACCOUNT="${SLURM_ACCOUNT:-bfwm-delta-gpu}"
PARTITION="gpuH200x8"
CONDA_ENV="/u/phu1/.conda/envs/IF/bin"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

MODELS=(
    "qwen3-1.7b:Qwen/Qwen3-1.7B"
    "qwen3-4b:Qwen/Qwen3-4B"
    "qwen3-8b:Qwen/Qwen3-8B"
)
CONFIGS=(
    "8 1024"
    "4 2048"
    "2 4096"
    "16 512"
)

submit() {
    local jobname="$1" walltime="$2" wrap_cmd="$3"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] $jobname [$walltime]"
        echo "  $wrap_cmd" | head -c 200
        echo "..."
        return 0
    fi
    sbatch \
        --job-name="$jobname" \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --mem=128G --cpus-per-task=8 \
        --time="$walltime" \
        --output="$LOG_DIR/${jobname}_%j.log" \
        --error="$LOG_DIR/${jobname}_%j.log" \
        --wrap="cd $REPO_ROOT
export PATH=\"$CONDA_ENV:\$PATH\"
export PYTHONPATH=\"$REPO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}\"
$wrap_cmd"
}

# ── Breakdown jobs ──────────────────────────────────────────────────────────
echo "=== Breakdown jobs (no-ckpt + ckpt) ==="
for entry in "${MODELS[@]}"; do
    tag="${entry%%:*}"
    name="${entry##*:}"
    for cfg in "${CONFIGS[@]}"; do
        read -r n T <<< "$cfg"
        for variant in "breakdown:" "breakdown_checkpointing:--gradient-checkpointing"; do
            subdir="${variant%%:*}"
            ckpt_flag="${variant##*:}"
            results_dir="SFT/benchmark/results/${subdir}"
            out_file="$results_dir/${tag}_n${n}_T${T}_m1.json"
            jobname="h200-${tag}-${subdir}-n${n}T${T}"
            cmd="mkdir -p $results_dir
python3 SFT/benchmark/benchmark_run.py \
    --gpu 0 --model '$name' \
    --batch-size $n --seq-length $T --val-batch-size 1 \
    --direct-batch-size 1 \
    $ckpt_flag \
    --output $out_file"
            submit "$jobname" "0:30:00" "$cmd"
        done
    done
done

# ── Scoring jobs ────────────────────────────────────────────────────────────
echo ""
echo "=== Scoring jobs ==="
for entry in "${MODELS[@]}"; do
    tag="${entry%%:*}"
    out_file="SFT/benchmark/results/scoring/scoring_${tag}.json"
    jobname="h200-scoring-${tag}"
    cmd="mkdir -p SFT/benchmark/results/scoring
python3 SFT/benchmark/benchmark_scoring.py \
    --gpu 0 --model-tag $tag \
    --output $out_file"
    submit "$jobname" "0:20:00" "$cmd"
done

echo ""
echo "=== Done ==="
[[ "$DRY_RUN" == "true" ]] && echo "Dry run — no jobs submitted." || echo "Monitor with: squeue -u \$USER"
