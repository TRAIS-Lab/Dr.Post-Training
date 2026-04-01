#!/bin/bash
#
# Rerun pipeline: collect LR sweep → train 5 seeds → eval
#
# Run this AFTER all LR sweep jobs complete:
#   bash SFT/train/rerun_pipeline.sh
#
# Steps:
#   1. Collect best LR from sweep results → update YAML configs
#   2. Submit training jobs (5 seeds per method)
#   3. Submit eval jobs (dependent on training completion)
#
# Methods: Layerwise-{Full,LoRA,MeSO}, Subset-{Full,LoRA,MeSO}
# Datasets: alpaca_samsum, tulu3_tydiqa
# Seeds: 2, 22, 42, 62, 82

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source cluster_env.sh || { echo "ERROR: cluster_env.sh not found."; exit 1; }
activate_env

METHODS="layerwise,subset"
SEEDS=(2 22 42 62 82)

DATASETS=(
    "alpaca_samsum:alpaca"
    "tulu3_tydiqa:tulu3"
)

DRY_RUN=false
SKIP_COLLECT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --skip-collect) SKIP_COLLECT=true; shift ;;
        --help|-h)
            echo "Usage: bash SFT/train/rerun_pipeline.sh [--dry-run] [--skip-collect]"
            echo ""
            echo "  --dry-run       Print commands without executing"
            echo "  --skip-collect  Skip LR collection (use existing YAML LRs)"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo ""
echo "========================================================"
echo "  Rerun Pipeline"
echo "========================================================"
echo "Methods:  $METHODS"
echo "Seeds:    ${SEEDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Dry run:  $DRY_RUN"
echo "========================================================"

# =============================================================================
# Step 1: Collect best LR from sweep results
# =============================================================================
if [[ "$SKIP_COLLECT" != "true" ]]; then
    echo ""
    echo "=== Step 1: Collecting LR sweep results ==="
    for ds_entry in "${DATASETS[@]}"; do
        config="${ds_entry%%:*}"
        echo ""
        echo "--- $config ---"
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY-RUN] bash SFT/train/lr/lr_sweep_collect.sh -c configs/$config -m $METHODS"
        else
            bash SFT/train/lr/lr_sweep_collect.sh -c "configs/$config" -m "$METHODS"
        fi
    done
else
    echo ""
    echo "=== Step 1: SKIPPED (--skip-collect) ==="
fi

# =============================================================================
# Step 2: Submit training jobs (5 seeds each)
# =============================================================================
echo ""
echo "=== Step 2: Submitting training jobs ==="

# Resolve method names
RESOLVED_METHODS="Layerwise-Full,Layerwise-LoRA,Layerwise-MeSO,Subset-Full,Subset-LoRA,Subset-MeSO"

declare -A TRAIN_IDS  # key="config:method" value="id1:id2:id3:id4:id5"

for ds_entry in "${DATASETS[@]}"; do
    config="${ds_entry%%:*}"
    train_filter="${ds_entry##*:}"

    IFS=',' read -ra method_list <<< "$RESOLVED_METHODS"
    for method in "${method_list[@]}"; do
        ids=""
        for seed in "${SEEDS[@]}"; do
            job_name="rerun-${config}-${method}-s${seed}"
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "[DRY-RUN] GPUS=1 MEM=64g TIME=4:00:00 JOB_NAME=$job_name ./submit.sh SFT/train/train.sh -c configs/$config -m $method --seed $seed"
                ids="${ids:+$ids:}DRY"
            else
                out=$(GPUS=1 MEM=64g TIME=4:00:00 JOB_NAME="$job_name" \
                    ./submit.sh SFT/train/train.sh -c "configs/$config" -m "$method" --seed "$seed")
                id=$(echo "$out" | grep -oP '\d+')
                ids="${ids:+$ids:}$id"
                echo "  $out"
            fi
        done
        key="${train_filter}:${method}"
        TRAIN_IDS[$key]="$ids"
        echo "  → $config / $method: ${TRAIN_IDS[$key]}"
    done
done

# =============================================================================
# Step 3: Submit eval jobs (dependent on training)
# =============================================================================
echo ""
echo "=== Step 3: Submitting eval jobs ==="

for ds_entry in "${DATASETS[@]}"; do
    config="${ds_entry%%:*}"
    train_filter="${ds_entry##*:}"

    IFS=',' read -ra method_list <<< "$RESOLVED_METHODS"
    for method in "${method_list[@]}"; do
        key="${train_filter}:${method}"
        dep_ids="${TRAIN_IDS[$key]}"
        job_name="eval-${config}-${method}"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY-RUN] DEPEND=afterok:$dep_ids GPUS=1 MEM=64g TIME=1:00:00 JOB_NAME=$job_name ./submit.sh SFT/eval/eval.sh --train $train_filter --method $method --batch_size 64 --n_test 500"
        else
            out=$(DEPEND="afterok:$dep_ids" GPUS=1 MEM=64g TIME=1:00:00 JOB_NAME="$job_name" \
                ./submit.sh SFT/eval/eval.sh --train "$train_filter" --method "$method" --batch_size 64 --n_test 500)
            echo "  $out  ($job_name, depends on $dep_ids)"
        fi
    done
done

echo ""
echo "========================================================"
echo "  Pipeline submitted!"
echo "========================================================"
echo ""
echo "Monitor: squeue -u \$USER"
echo "After completion, run result.ipynb to generate figures."
echo "========================================================"
