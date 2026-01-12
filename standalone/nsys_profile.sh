#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python}"
VAL_STRATEGY="${VAL_STRATEGY:-merged}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
VAL_SEQ_LENGTH="${VAL_SEQ_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-5}"

# nsys profile -o "${SCRIPT_DIR}/nsys_report_na" \
#   -t cuda,nvtx,osrt \
#   --sample=none \
#   --cuda-memory-usage=true \
#   --force-overwrite=true \
#   -- "${PYTHON}" "${SCRIPT_DIR}/streaming_minimal.py" \
#   --method "NA_NA_Full" \
#   --val-strategy "${VAL_STRATEGY}" \
#   --seq-length "${SEQ_LENGTH}" \
#   --val-seq-length "${VAL_SEQ_LENGTH}" \
#   --batch-size "${BATCH_SIZE}" \
#   --val-batch-size "${VAL_BATCH_SIZE}" \
#   --iterations "${ITERATIONS}" \
#   --warmup "${WARMUP}" \
#   "$@"

nsys profile -o "${SCRIPT_DIR}/nsys_report_stream" \
  -t cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -- "${PYTHON}" "${SCRIPT_DIR}/streaming_minimal.py" \
  --method "Streaming_NA_Full" \
  --val-strategy "${VAL_STRATEGY}" \
  --seq-length "${SEQ_LENGTH}" \
  --val-seq-length "${VAL_SEQ_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --iterations "${ITERATIONS}" \
  --warmup "${WARMUP}" \
  "$@"
