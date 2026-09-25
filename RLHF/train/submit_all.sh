#!/bin/bash
# Submit the RLHF toxicity sweep as ONE Slurm job array (see submit.sh / run.slurm).
#
# Grid: methods x scenarios x seeds, where a scenario is "<self-ref|held-out>/<val_loss_type>"
# with val_loss_type in {reward, token-pg, train-loss}. FullTraining does not use a target
# and is run once per seed (train.sh files it under v0-rew).
#
# Usage:
#   bash RLHF/train/submit_all.sh [--seeds "2 22 42 62 82"] [--methods IIF-LoRA,LayerWiseSubset-LoRA,GlobalSubset-LoRA]
#                                 [--scenarios "self-ref/reward,..."] [--n_val 1024] [--concurrency 8]
#                                 [--time 1-00:00:00] [--smoke] [--dry-run]
# Env: QOS (default high), EXCLUDE=<nodes>, DEPEND=afterany:<jobid>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/cluster_env.sh"

SEEDS="2 22 42 62 82"
METHODS="IIF-LoRA,LayerWiseSubset-LoRA,GlobalSubset-LoRA"
SCENARIOS="self-ref/reward,self-ref/token-pg,self-ref/train-loss,held-out/reward,held-out/token-pg,held-out/train-loss"
N_VAL_HELDOUT=1024
CONCURRENCY=8
TIME="1-00:00:00"
CONFIG_DIR="configs/toxicity"
SMOKE=0; DRY=0; WITH_BASELINE=1; WITH_TARGET_ONLY=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)        SEEDS="$2"; shift 2 ;;
        --methods)      METHODS="$2"; shift 2 ;;
        --scenarios)    SCENARIOS="$2"; shift 2 ;;
        --n_val)        N_VAL_HELDOUT="$2"; shift 2 ;;
        --concurrency)  CONCURRENCY="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --config_dir)   CONFIG_DIR="$2"; shift 2 ;;
        --no-baseline)  WITH_BASELINE=0; shift ;;
        --no-target-only) WITH_TARGET_ONLY=0; shift ;;
        --smoke)        SMOKE=1; shift ;;
        --dry-run)      DRY=1; shift ;;
        -h|--help)      sed -n 2,13p "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
export QOS="${QOS:-high}"

TAG="$(date +%Y%m%d-%H%M%S)"
MANIFEST_DIR="$SCRATCH_DIR/Dr.Post-Training/RLHF/manifests"
mkdir -p "$MANIFEST_DIR"
M="$MANIFEST_DIR/rlhf-${TAG}.txt"; : > "$M"

scenario_args() {   # "<mode>/<vlt>" -> "--n_val N --val_loss_type vlt"
    local mode="${1%%/*}" vlt="${1##*/}"
    case "$vlt" in reward|token-pg|train-loss) ;; *) echo "ERROR: bad val_loss_type '$vlt'" >&2; exit 1 ;; esac
    case "$mode" in
        self-ref) echo "--n_val 0 --val_loss_type $vlt" ;;
        held-out) echo "--n_val $N_VAL_HELDOUT --val_loss_type $vlt" ;;
        *) echo "ERROR: bad scenario mode '$mode' (self-ref|held-out)" >&2; exit 1 ;;
    esac
}

if [[ "$SMOKE" == "1" ]]; then
    # One short held-out job per target type (exercises generation, capture, curation, eval).
    for vlt in reward token-pg train-loss; do
        echo "-c $CONFIG_DIR -m LayerWiseSubset-LoRA --seed 42 --n_val 64 --val_batch_size 64 --val_loss_type $vlt --max_steps 2" >> "$M"
    done
    echo "-c $CONFIG_DIR -m FullTraining-LoRA --seed 42 --max_steps 2" >> "$M"
    TIME="02:00:00"; JOB="rlhf-smoke-${TAG}"
else
    IFS=',' read -ra SC <<< "$SCENARIOS"; IFS=',' read -ra ME <<< "$METHODS"
    for seed in $SEEDS; do
        [[ "$WITH_BASELINE" == "1" ]] && echo "-c $CONFIG_DIR -m FullTraining-LoRA --seed $seed" >> "$M"
        # TargetOnly: once per seed, only if some held-out scenario is requested
        if [[ "$WITH_TARGET_ONLY" == "1" && "$SCENARIOS" == *held-out* ]]; then
            echo "-c $CONFIG_DIR -m TargetOnly-LoRA --seed $seed --n_val $N_VAL_HELDOUT" >> "$M"
        fi
        for sc in "${SC[@]}"; do
            sargs="$(scenario_args "$sc")"
            for m in "${ME[@]}"; do
                echo "-c $CONFIG_DIR -m $m --seed $seed $sargs" >> "$M"
            done
        done
    done
    JOB="rlhf-tox-${TAG}"
fi

N=$(wc -l < "$M"); ARRAY="0-$((N-1))%${CONCURRENCY}"
echo "Manifest: $M ($N tasks)  array=$ARRAY  time=$TIME  qos=$QOS  job=$JOB"
cd "$REPO_ROOT"
if [[ "$DRY" == "1" ]]; then
    cat "$M"
    DRY_RUN=1 ARRAY="$ARRAY" MANIFEST="$M" GPUS=1 MEM=128G TIME="$TIME" JOB_NAME="$JOB" \
        DEPEND="${DEPEND:-}" EXCLUDE="${EXCLUDE:-}" ./submit.sh RLHF/train/train.sh
else
    ARRAY="$ARRAY" MANIFEST="$M" GPUS=1 MEM=128G TIME="$TIME" JOB_NAME="$JOB" \
        DEPEND="${DEPEND:-}" EXCLUDE="${EXCLUDE:-}" ./submit.sh RLHF/train/train.sh
fi
