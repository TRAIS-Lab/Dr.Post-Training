#!/bin/bash
# A40 Slurm cluster (one node, 4x A40): array wrapper for the question-answering task scripts. One manifest line per task, passed as the arguments of $TASK
# (manifests: python SFT/train/qa_matrix_manifest.py -> SFT/train/manifests/). Slurm reads the manifest line at task start, so never
# rewrite a manifest an array is still using.
#   mkdir -p logs/qa; N=$(wc -l < SFT/train/manifests/qa_matrix_A1.txt)
#   sbatch --array=0-$((N-1))%4 SFT/train/slurm/qa_array.sbatch SFT/train/manifests/qa_matrix_A1.txt SFT/train/qa_matrix_task.sh
#   task script SFT/train/qa_target_only_task.sh with qa_target_only.txt (Target-Only rows); SFT/case_study/run.sh with case_study.txt (lines = run.sh flags)
# Full-parameter / MeSO arms load a 1.24B-parameter fp32 model + AdamW states, hence 48 GB host memory.
#SBATCH --job-name=qa --partition=standard --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=8:00:00
#SBATCH --output=logs/qa/%x_%A_%a.out
cd "$SLURM_SUBMIT_DIR"
MANIFEST=${1:?usage: sbatch --array=... qa_array.sbatch <manifest> [task script]}; TASK=${2:-SFT/train/qa_matrix_task.sh}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
[ -n "$line" ] || { echo "[sbatch] empty manifest line $((SLURM_ARRAY_TASK_ID+1)) in $MANIFEST"; exit 1; }
echo "[sbatch] task $SLURM_ARRAY_TASK_ID of $MANIFEST: $TASK $line  (GPU $CUDA_VISIBLE_DEVICES, $(date))"
# Shared-node guard: processes started outside Slurm can occupy the allocated card (the array then dies at step 1 with CUDA out of
# memory). Wait until the card holds at most GPU_FREE_MIB MiB (default 2000) for up to GPU_WAIT_MIN minutes (default 720), polling
# every 5 minutes; GPU_WAIT_MIN=0 disables the guard.
GPU_WAIT_MIN=${GPU_WAIT_MIN:-720}; GPU_FREE_MIB=${GPU_FREE_MIB:-2000}; waited=0
while [ "$GPU_WAIT_MIN" -gt 0 ]; do
  used=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$used" ] || [ "$used" -le "$GPU_FREE_MIB" ] && break
  [ "$waited" -ge "$GPU_WAIT_MIN" ] && { echo "[sbatch] GPU $CUDA_VISIBLE_DEVICES still holds ${used} MiB after ${waited} min; giving up"; exit 1; }
  [ "$waited" -eq 0 ] && echo "[sbatch] GPU $CUDA_VISIBLE_DEVICES holds ${used} MiB of foreign memory; waiting (up to ${GPU_WAIT_MIN} min)"
  sleep 300; waited=$((waited + 5))
done
[ "$waited" -gt 0 ] && echo "[sbatch] GPU $CUDA_VISIBLE_DEVICES free after ${waited} min ($(date))"
bash "$TASK" $line
