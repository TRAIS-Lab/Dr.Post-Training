#!/bin/bash
# Submit all 3 model benchmark jobs to Slurm.
# Uses sbatch --wrap with srun to ensure actual execution.
#
# Usage:
#   bash SFT/benchmark/slurm/launch_all.sh
#   bash SFT/benchmark/slurm/launch_all.sh --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIGS_DIR="SFT/benchmark/slurm/configs"
RESULTS_DIR="SFT/benchmark/results/h200"
mkdir -p "$CONFIGS_DIR" "$RESULTS_DIR" logs

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

echo "========================================"
echo "  Dr.Post-Training H200 Benchmark Suite"
echo "  3 models x 9 configs x 13 combos = 351 runs"
echo "========================================"
echo ""

# ── Write config files ───────────────────────────────────────────────────────
cat > "$CONFIGS_DIR/qwen3-0.6b.json" << 'EOF'
[
  {"n":32,"T":512,"m":1}, {"n":32,"T":512,"m":2}, {"n":32,"T":512,"m":4},
  {"n":16,"T":1024,"m":1},{"n":16,"T":1024,"m":2},{"n":16,"T":1024,"m":4},
  {"n":8,"T":2048,"m":1}, {"n":8,"T":2048,"m":2}, {"n":8,"T":2048,"m":4}
]
EOF

cat > "$CONFIGS_DIR/qwen3-1.7b.json" << 'EOF'
[
  {"n":16,"T":512,"m":1}, {"n":16,"T":512,"m":2}, {"n":16,"T":512,"m":4},
  {"n":8,"T":1024,"m":1}, {"n":8,"T":1024,"m":2}, {"n":8,"T":1024,"m":4},
  {"n":4,"T":2048,"m":1}, {"n":4,"T":2048,"m":2}, {"n":4,"T":2048,"m":4}
]
EOF

cat > "$CONFIGS_DIR/qwen3-8b.json" << 'EOF'
[
  {"n":16,"T":512,"m":1}, {"n":16,"T":512,"m":2}, {"n":16,"T":512,"m":4},
  {"n":8,"T":1024,"m":1}, {"n":8,"T":1024,"m":2}, {"n":8,"T":1024,"m":4},
  {"n":4,"T":2048,"m":1}, {"n":4,"T":2048,"m":2}, {"n":4,"T":2048,"m":4}
]
EOF

# ── Write the runner Python script ───────────────────────────────────────────
cat > "$RESULTS_DIR/run_benchmark.py" << 'PYEOF'
"""Run all benchmark configs for a single model. Called from Slurm."""
import json, subprocess, sys, os, time

model = os.environ["MODEL"]
tag = os.environ["TAG"]
configs_file = os.environ["CONFIGS_FILE"]
results_dir = os.environ["RESULTS_DIR"]
warmup = int(os.environ.get("NUM_WARMUP", "10"))
iters = int(os.environ.get("NUM_ITERS", "20"))

with open(configs_file) as f:
    configs = json.load(f)

print(f"Benchmark: {model} ({tag})", flush=True)
print(f"Configs: {len(configs)}, Warmup: {warmup}, Iters: {iters}", flush=True)

total = len(configs)
for i, cfg in enumerate(configs):
    n, T, m = cfg["n"], cfg["T"], cfg["m"]
    out = os.path.join(results_dir, f"{tag}_n{n}_T{T}_m{m}.json")

    if os.path.exists(out):
        print(f"[{i+1}/{total}] SKIP n={n} T={T} m={m} (exists)", flush=True)
        continue

    print(f"[{i+1}/{total}] Running n={n} T={T} m={m} ...", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable, "SFT/benchmark/benchmark_run.py",
        "--model", model,
        "--batch-size", str(n),
        "--seq-length", str(T),
        "--val-batch-size", str(m),
        "--num-warmup", str(warmup),
        "--num-iterations", str(iters),
        "--output", out,
    ]
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    status = "Done" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  {status} ({elapsed:.0f}s)", flush=True)

print("\nAll configs complete.", flush=True)
PYEOF

# ── Submit function ──────────────────────────────────────────────────────────
submit_job() {
    local model="$1" tag="$2" mem="$3" time="$4"
    local configs_file="$REPO_ROOT/$CONFIGS_DIR/${tag}.json"
    local logfile="$REPO_ROOT/$RESULTS_DIR/${tag}.log"

    echo "  Model:       $model"
    echo "  Tag:         $tag"
    echo "  Resources:   1 GPU, $mem RAM, $time"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would submit"
        echo ""
        return 0
    fi

    sbatch \
        --job-name="$tag" \
        --account=research \
        --partition=general,overflow \
        --qos=high \
        --nodes=1 \
        --ntasks=1 \
        --gres=gpu:1 \
        --mem="$mem" \
        --cpus-per-task=16 \
        --time="$time" \
        --output="$logfile" \
        --error="$logfile" \
        --wrap="srun bash -c '
. /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate drpt
cd $REPO_ROOT
export PYTHONPATH=$REPO_ROOT
export HF_HOME=/workspace-vast/pretrained_ckpts
export MODEL=\"$model\"
export TAG=\"$tag\"
export CONFIGS_FILE=\"$configs_file\"
export RESULTS_DIR=\"$REPO_ROOT/$RESULTS_DIR\"
python $REPO_ROOT/$RESULTS_DIR/run_benchmark.py
'"
    echo ""
}

# ── Submit all 3 jobs ────────────────────────────────────────────────────────
echo "--- Qwen3-0.6B (small) ---"
submit_job "Qwen/Qwen3-0.6B"  "qwen3-0.6b"  "128G"  "3:00:00"

echo "--- Qwen3-1.7B (medium) ---"
submit_job "Qwen/Qwen3-1.7B"  "qwen3-1.7b"  "128G"  "4:00:00"

echo "--- Qwen3-8B (large) ---"
submit_job "Qwen/Qwen3-8B"    "qwen3-8b"    "256G"  "6:00:00"

echo "========================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "  Dry run complete. No jobs submitted."
else
    echo "  Submitted 3 jobs (1 GPU each, running in parallel)"
fi
echo "  Results dir: $RESULTS_DIR/"
echo "  Monitor:     squeue -u \$(whoami)"
echo "========================================"
