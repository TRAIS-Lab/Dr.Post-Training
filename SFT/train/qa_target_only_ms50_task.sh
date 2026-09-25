#!/bin/bash
# qa_target_only_ms50_task.sh <setting> <FullTraining-Full|FullTraining-LoRA|FullTraining-MeSO> <seed> 50   (one Slurm array task;
# manifests SFT/train/manifests/qa_target_only_ms50.txt (LoRA, Full) and qa_target_only_ms50_meso.txt, wrapper slurm/qa_array.sbatch)
# Target-Only 50-step probe: qa_target_only_task.sh with the step budget of the manifest line, under a runs root the paper table
# scanner does not read (SFT/tables/qa_downstream.py --scan treats every ms<steps> dir directly under runs_v2 as a Target-Only row).
# Results: SFT/tables/qa_target_only_probe.py.
#   N=$(wc -l < SFT/train/manifests/qa_target_only_ms50.txt)
#   sbatch --array=0-$((N-1))%4 SFT/train/slurm/qa_array.sbatch SFT/train/manifests/qa_target_only_ms50.txt SFT/train/qa_target_only_ms50_task.sh
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"
export RUNS_ROOT=${RUNS_ROOT:-$SCRATCH_DIR/Dr.Post-Training/SFT/runs_v2/target_only_ms50}
mkdir -p "$RUNS_ROOT"
exec bash "$REPO_ROOT/SFT/train/qa_target_only_task.sh" "$@"
