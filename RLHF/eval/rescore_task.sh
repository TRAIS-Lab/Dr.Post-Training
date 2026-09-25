#!/bin/bash
# rescore_task.sh <BASE | run dir name under $RLHF_ROOT>   (one Slurm array task; manifest = rescore_manifest.txt)
# Re-scores one final RLHF checkpoint with the DaNLP judge of the paper plus English judges (RLHF/eval/rescore_final.py).
set -u
run=$1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"; activate_env
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
RLHF_ROOT=${RLHF_ROOT:-$SCRATCH_DIR/Dr.Post-Training/RLHF}
if [ "$run" = "BASE" ]; then out=$RLHF_ROOT/rescore_BASE.json; else run=$RLHF_ROOT/$run; out=$run/rescore_final.json; fi
[ -f "$out" ] && { echo "[rescore] exists: $out"; exit 0; }
python -m RLHF.eval.rescore_final --run_dir "$run" --out "$out"
