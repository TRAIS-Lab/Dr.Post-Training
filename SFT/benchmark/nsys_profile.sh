#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT="${OUT:-${SCRIPT_DIR}/nsys_report_stream_na}"
PYTHON="${PYTHON:-python}"
METHOD="${METHOD:-Streaming_LoGra_Full}"
VAL_STRATEGY="${VAL_STRATEGY:-merged}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
VAL_SEQ_LENGTH="${VAL_SEQ_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-5}"

nsys profile -o "${OUT}" \
  -t cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -- "${PYTHON}" "${SCRIPT_DIR}/benchmark.py" \
  --methods "${METHOD}" \
  --val-strategy "${VAL_STRATEGY}" \
  --batch-size "${BATCH_SIZE}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --num-iterations "${ITERATIONS}" \
  --num-warmup "${WARMUP}" \
  "$@"
