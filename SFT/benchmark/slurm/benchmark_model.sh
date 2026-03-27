#!/bin/bash
# Benchmark all method x scoring combos for one model across a set of configs.
# Reads MODEL, TAG, CONFIGS_FILE from environment (set by launch_all.sh).

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RESULTS_DIR="$REPO_ROOT/SFT/benchmark/results/h200"
mkdir -p "$RESULTS_DIR"

# Write our own log (Slurm output files have NFS visibility delays)
LOGFILE="$RESULTS_DIR/${TAG:-unknown}_${SLURM_JOB_ID:-$$}.log"
exec > "$LOGFILE" 2>&1
set -x

echo "=== Job start: $(date) on $(hostname) ==="

# Activate drpt conda env
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate drpt

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/workspace-vast/pretrained_ckpts"

: "${MODEL:?ERROR: MODEL env var required}"
: "${TAG:?ERROR: TAG env var required}"
: "${CONFIGS_FILE:?ERROR: CONFIGS_FILE env var required}"
NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_ITERS="${NUM_ITERS:-20}"

echo "Model:   $MODEL"
echo "Tag:     $TAG"
echo "Configs: $CONFIGS_FILE"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-0}"

python3 << PYEOF
import json, subprocess, sys, os, time

with open("$CONFIGS_FILE") as f:
    configs = json.load(f)

total = len(configs)
for i, cfg in enumerate(configs):
    n, T, m = cfg["n"], cfg["T"], cfg["m"]
    out = os.path.join("$RESULTS_DIR", f"$TAG" + f"_n{n}_T{T}_m{m}.json")

    if os.path.exists(out):
        print(f"[{i+1}/{total}] SKIP n={n} T={T} m={m} (exists)", flush=True)
        continue

    print(f"[{i+1}/{total}] Running n={n} T={T} m={m} ...", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable, "SFT/benchmark/benchmark_run.py",
        "--gpu", "0",
        "--model", "$MODEL",
        "--batch-size", str(n),
        "--seq-length", str(T),
        "--val-batch-size", str(m),
        "--num-warmup", str($NUM_WARMUP),
        "--num-iterations", str($NUM_ITERS),
        "--output", out,
    ]
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  WARNING: failed (exit {result.returncode}, {elapsed:.0f}s)", flush=True)
    else:
        print(f"  Done ({elapsed:.0f}s)", flush=True)

print("All configs complete.", flush=True)
PYEOF

echo "=== Job done: $(date) ==="
