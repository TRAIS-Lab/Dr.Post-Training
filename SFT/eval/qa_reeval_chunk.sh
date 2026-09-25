#!/bin/bash
# qa_reeval_chunk.sh <manifest> <first_line> <count> <parallel>   (one Slurm array task; wrapper SFT/train/slurm/qa_array.sbatch)
# Runs qa_reeval_task.sh on <count> run dirs of <manifest> starting at 1-based <first_line>, <parallel> at a time on the task's GPU
# (an evaluation of Llama-3.2-1B needs ~5 GB, so several fit on one A40). Chunk manifests: SFT/train/manifests/qa_reeval_chunks.txt.
set -u
manifest=$1; first=$2; count=$3; par=${4:-4}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"
mkdir -p logs/qa/reeval
sed -n "${first},$((first + count - 1))p" "$manifest" | xargs -P "$par" -I{} bash -c 'd="{}"; bash SFT/eval/qa_reeval_task.sh "$d" > "logs/qa/reeval/$(basename "$d").log" 2>&1; echo "[chunk] exit $? $(basename "$d")"'
