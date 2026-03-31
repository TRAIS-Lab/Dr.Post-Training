#!/bin/bash
# Submit benchmark jobs to Slurm.
#
# Models: Qwen3-0.6B, Qwen3-1.7B, Llama-3.2-3B
# Configs per model: n=8 T=512 m=1, n=2 T=1024 m=1
# 3 models x 2 configs x 13 combos = 78 runs
#
# Usage:
#   bash SFT/benchmark/slurm/launch_all.sh
#   bash SFT/benchmark/slurm/launch_all.sh --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIGS_DIR="SFT/benchmark/slurm/configs"
RESULTS_DIR="SFT/benchmark/results/breakdown"
mkdir -p "$CONFIGS_DIR" "$RESULTS_DIR" logs

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

echo "========================================"
echo "  Dr.Post-Training Benchmark Suite"
echo "  3 models x 2 configs x 13 combos = 78 runs"
echo "========================================"
echo ""

submit_job() {
    local model="$1" tag="$2" mem="$3" time="$4"
    local configs_file="$REPO_ROOT/$CONFIGS_DIR/${tag}.json"

    echo "  Model:       $model"
    echo "  Tag:         $tag"
    echo "  Resources:   1 GPU, $mem RAM, $time"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would submit"
        echo ""
        return 0
    fi

    sbatch \
        --job-name="bench-$tag" \
        --partition=standard \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --mem="$mem" --cpus-per-task=16 \
        --time="$time" \
        --output="logs/bench-${tag}_%j.log" \
        --error="logs/bench-${tag}_%j.log" \
        --wrap="
cd $REPO_ROOT
export PYTHONPATH=$REPO_ROOT
export PYTHON=python

python3 -c '
import json, subprocess, sys, os, time
model, tag = \"$model\", \"$tag\"
with open(\"$configs_file\") as f:
    configs = json.load(f)
for i, cfg in enumerate(configs):
    n, T, m = cfg[\"n\"], cfg[\"T\"], cfg[\"m\"]
    out = os.path.join(\"$REPO_ROOT/$RESULTS_DIR\", f\"{tag}_n{n}_T{T}_m{m}.json\")
    if os.path.exists(out):
        print(f\"SKIP n={n} T={T} m={m}\", flush=True); continue
    print(f\"Running n={n} T={T} m={m}...\", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, \"SFT/benchmark/benchmark_run.py\",
        \"--gpu\", \"0\", \"--model\", model,
        \"--batch-size\", str(n), \"--seq-length\", str(T), \"--val-batch-size\", str(m),
        \"--direct-batch-size\", \"1\", \"--output\", out])
    print(f\"  {\"Done\" if r.returncode==0 else \"FAILED\"} ({time.time()-t0:.0f}s)\", flush=True)
print(\"Complete.\", flush=True)
'
"
    echo ""
}

echo "--- Qwen3-0.6B ---"
submit_job "Qwen/Qwen3-0.6B"         "qwen3-0.6b"    "64G"  "1:00:00"

echo "--- Qwen3-1.7B ---"
submit_job "Qwen/Qwen3-1.7B"         "qwen3-1.7b"    "64G"  "2:00:00"

echo "--- Llama-3.2-3B ---"
submit_job "meta-llama/Llama-3.2-3B"  "llama-3.2-3b"  "64G"  "2:00:00"

echo "========================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "  Dry run complete. No jobs submitted."
else
    echo "  Submitted 3 jobs."
fi
echo "  Results: $RESULTS_DIR/"
echo "========================================"
