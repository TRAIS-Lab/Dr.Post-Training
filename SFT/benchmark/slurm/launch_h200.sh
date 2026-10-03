#!/bin/bash
# Submit the H200 benchmark suite as ONE Slurm job array (one GPU per task, one (model, n, T) per task).
#
# Protocol (differs from the A40 suite): bf16 weights, activations and Adam state, fused AdamW (torch.optim.AdamW(fused=True);
# the foreach implementation materialises full-size temporaries), the CuTe backend's Hopper kernels (drpt/kernels/hopper_ops.py:
# TMA + wgmma selected w.grad, GIP, PIP and compressed projection; the Ampere mma.sync kernels lose to cuBLAS on sm_90, see the
# README's "Hopper note"; KERNEL_BACKEND=off runs the PyTorch/cuBLAS reference ops with the CuTe row gather), and NO activation
# checkpointing by default: Qwen3-8B-Base fits up to ~9k tokens per step on one 141 GB H200 without it (CKPT=0). CKPT=1 turns
# checkpointing on (needed for Qwen3-14B-Base beyond ~2.5k tokens per step; results then land in breakdown_checkpointing/).
# Configs: n x T = (8,512), (16,512), (32,256), (8,1024), (4,1024), (16,256), (4,2048), (2,2048), (2,4096), m = 1, k = n/2;
# plus one standalone scoring job per model (benchmark_scoring.py, T-sweep to 32768).
#
# Usage:
#   bash SFT/benchmark/slurm/launch_h200.sh                # submit
#   bash SFT/benchmark/slurm/launch_h200.sh --dry-run       # print the manifest and the sbatch command
# Environment knobs: SLURM_PARTITION, SLURM_QOS, SLURM_ACCOUNT (optional), KERNEL_BACKEND (off|cute|triton), CKPT (0|1),
#   MODELS ("tag:hf-name-or-path ..."), MAX_CONCURRENT (array throttle, default 2), CONDA_ENV (bin dir of the env).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
[[ -f "$REPO_ROOT/cluster_env.sh" ]] && source "$REPO_ROOT/cluster_env.sh"

PARTITION="${SLURM_PARTITION:-gpu}"
QOS="${SLURM_QOS:-}"
ACCOUNT="${SLURM_ACCOUNT:-}"
KERNEL_BACKEND="${KERNEL_BACKEND:-cute}"
CKPT="${CKPT:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
CONDA_ENV="${CONDA_ENV:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"   # empty -> activate_env from cluster_env.sh inside the task
MODELS="${MODELS:-qwen3-8b:Qwen/Qwen3-8B-Base}"
CONFIGS=("8 512" "16 512" "32 256" "8 1024" "4 1024" "16 256" "4 2048" "2 2048" "2 4096")
RESULTS="SFT/benchmark/results/h200"
LOG_DIR="logs"
SUBDIR=$([[ "$CKPT" == "1" ]] && echo breakdown_checkpointing || echo breakdown)
mkdir -p "$LOG_DIR" "$RESULTS/$SUBDIR" "$RESULTS/scoring"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Manifest: one line per task ──────────────────────────────────────────────
MANIFEST="$RESULTS/manifest_$(date +%Y%m%d_%H%M%S).txt"
: > "$MANIFEST"
for entry in $MODELS; do
    tag="${entry%%:*}"; name="${entry#*:}"
    for cfg in "${CONFIGS[@]}"; do
        read -r n T <<< "$cfg"
        echo "breakdown $tag $name $n $T" >> "$MANIFEST"
    done
    echo "scoring $tag" >> "$MANIFEST"
done
NTASKS=$(wc -l < "$MANIFEST")

# ── Task body (runs inside the array task) ───────────────────────────────────
TASK_SCRIPT="$RESULTS/task.sh"
cat > "$TASK_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
cd "$REPO_ROOT"
[[ -f "$REPO_ROOT/cluster_env.sh" ]] && { source "$REPO_ROOT/cluster_env.sh"; declare -F activate_env >/dev/null && activate_env; }
[[ -n "$CONDA_ENV" ]] && export PATH="$CONDA_ENV:\$PATH"
export PYTHONPATH="$REPO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export DRPT_KERNEL_BACKEND="$KERNEL_BACKEND"
line="\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
read -r kind tag name n T <<< "\$line"
echo "[task \$SLURM_ARRAY_TASK_ID] \$line on \$(hostname), backend $KERNEL_BACKEND, checkpointing $CKPT"
case "\$kind" in
  breakdown)
    python3 SFT/benchmark/benchmark_run.py --model "\$name" --batch-size "\$n" --seq-length "\$T" --val-batch-size 1 \\
        --direct-batch-size 1 $([[ "$CKPT" == "1" ]] && echo --gradient-checkpointing) --fused-adamw --kernel-backend "$KERNEL_BACKEND" \\
        --output "$RESULTS/$SUBDIR/\${tag}_n\${n}_T\${T}_m1.json" ;;
  scoring)
    python3 SFT/benchmark/benchmark_scoring.py --model-tag "\$tag" \\
        --t-sweep 256,512,1024,2048,4096,8192,16384,32768 --m-sweep 1,2,4,8,16 \\
        --output "$RESULTS/scoring/scoring_\${tag}.json" ;;
esac
EOF
chmod +x "$TASK_SCRIPT"

sbatch_args=(--job-name=h200-bench --partition="$PARTITION" --nodes=1 --ntasks=1 --gres=gpu:1
             --cpus-per-task=16 --mem=128G --time=3:00:00
             --array="0-$((NTASKS - 1))%${MAX_CONCURRENT}"
             --output="$LOG_DIR/h200-bench_%A_%a.log")
[[ -n "$QOS" ]] && sbatch_args+=(--qos="$QOS")
[[ -n "$ACCOUNT" ]] && sbatch_args+=(--account="$ACCOUNT")

echo "=== manifest ($NTASKS tasks): $MANIFEST"
cat "$MANIFEST"
echo
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY] sbatch ${sbatch_args[*]} $TASK_SCRIPT"
    echo "Dry run — no jobs submitted."
else
    sbatch "${sbatch_args[@]}" "$TASK_SCRIPT"
    echo "Monitor with: squeue -u \$USER -n h200-bench"
fi
