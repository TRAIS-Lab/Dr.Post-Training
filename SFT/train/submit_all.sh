#!/bin/bash
#
# 5-seed train + target-only + eval pipeline across all 5 SFT settings.
# Uses fixed LRs from configs (no LR sweep). Includes target-only baselines.
#
# Note: target-only is per-target-task (not per-setting). E.g. tulu3_tydiqa
# and less_tydiqa share the same target-only baseline (tydiqa).
#
# Usage:
#   bash SFT/train/submit_all.sh             # submit
#   bash SFT/train/submit_all.sh --dry-run   # print commands only

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source cluster_env.sh || { echo "ERROR: cluster_env.sh not found."; exit 1; }
activate_env

SEEDS=(2 22 42 62 82)
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Main training: config_dir : train_filter : task_name : main methods
SETTINGS=(
    "alpaca_samsum:alpaca:samsum:FullTraining-Full,FullTraining-LoRA,FullTraining-MeSO,LayerWiseSubset-Full,LayerWiseSubset-LoRA,LayerWiseSubset-MeSO,GlobalSubset-Full,GlobalSubset-LoRA,GlobalSubset-MeSO"
    "tulu3_tydiqa:tulu3:tydiqa:FullTraining-Full,FullTraining-LoRA,FullTraining-MeSO,LayerWiseSubset-Full,LayerWiseSubset-LoRA,LayerWiseSubset-MeSO,GlobalSubset-Full,GlobalSubset-LoRA,GlobalSubset-MeSO"
    "less_tydiqa:less:tydiqa:FullTraining-Full,FullTraining-LoRA,FullTraining-MeSO,LayerWiseSubset-Full,LayerWiseSubset-LoRA,LayerWiseSubset-MeSO,GlobalSubset-Full,GlobalSubset-LoRA,GlobalSubset-MeSO"
    "triviaqa_nq:triviaqa:nq_open:FullTraining-Full,LayerWiseSubset-Full,GlobalSubset-Full"
    "nq_squad:nq_open_squad:squad:FullTraining-Full,LayerWiseSubset-Full,GlobalSubset-Full"
)

# Target-only: task : config_dir_for_yaml : FT types
# (target-only output dir is `${task}_val_${task}-...`, identical regardless of source dataset)
TARGET_TASKS=(
    "samsum:alpaca_samsum:FullTraining-Full,FullTraining-LoRA,FullTraining-MeSO"
    "tydiqa:tulu3_tydiqa:FullTraining-Full,FullTraining-LoRA,FullTraining-MeSO"
    "nq_open:triviaqa_nq:FullTraining-Full"
    "squad:nq_squad:FullTraining-Full"
)

declare -A MAIN_IDS
declare -A TARGET_IDS

submit() {
    local job_name="$1"; shift
    local time="$1"; shift
    local depend="$1"; shift
    local script="$1"; shift
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY] $job_name | TIME=$time DEPEND=${depend:-none} -- $script $*" >&2
        echo "DRY"
    else
        local out
        if [[ -n "$depend" ]]; then
            out=$(DEPEND="$depend" GPUS=1 MEM=64g TIME="$time" JOB_NAME="$job_name" \
                ./submit.sh "$script" "$@")
        else
            out=$(GPUS=1 MEM=64g TIME="$time" JOB_NAME="$job_name" \
                ./submit.sh "$script" "$@")
        fi
        echo "$out" | grep -oP '\d+' | tail -1
    fi
}

echo "========================================================"
echo "  Step 1: Main training (5 seeds × all methods × 5 settings)"
echo "========================================================"
for entry in "${SETTINGS[@]}"; do
    IFS=':' read -r config train_filter task main_methods <<< "$entry"
    IFS=',' read -ra method_list <<< "$main_methods"

    for method in "${method_list[@]}"; do
        ids=""
        for seed in "${SEEDS[@]}"; do
            jid=$(submit "main-${config}-${method}-s${seed}" "4:00:00" "" \
                SFT/train/train.sh -c "configs/$config" -m "$method" --seed "$seed")
            ids="${ids:+$ids:}$jid"
        done
        MAIN_IDS["${config}:${method}"]="$ids"
        echo "  ${config} / ${method}: ${ids}"
    done
done

echo ""
echo "========================================================"
echo "  Step 2: Target-only training (5 seeds × FT types × per-task)"
echo "========================================================"
for entry in "${TARGET_TASKS[@]}"; do
    IFS=':' read -r task config methods <<< "$entry"
    IFS=',' read -ra t_list <<< "$methods"

    for method in "${t_list[@]}"; do
        ids=""
        for seed in "${SEEDS[@]}"; do
            jid=$(submit "target-${task}-${method}-s${seed}" "2:00:00" "" \
                SFT/train/train_val_ablation.sh --task "$task" --config_dir "$config" \
                --methods "$method" --seed "$seed" --eval_steps 12)
            ids="${ids:+$ids:}$jid"
        done
        TARGET_IDS["${task}:${method}"]="$ids"
        echo "  target-${task} / ${method}: ${ids}"
    done
done

echo ""
echo "========================================================"
echo "  Step 3: Eval main training (depends on Step 1)"
echo "========================================================"
for entry in "${SETTINGS[@]}"; do
    IFS=':' read -r config train_filter task main_methods <<< "$entry"
    IFS=',' read -ra method_list <<< "$main_methods"

    for method in "${method_list[@]}"; do
        dep="${MAIN_IDS["${config}:${method}"]}"
        depend_arg=""
        [[ "$DRY_RUN" != "true" ]] && depend_arg="afterok:$dep"
        submit "eval-${config}-${method}" "1:00:00" "$depend_arg" \
            SFT/eval/eval.sh --train "$train_filter" --method "$method" --batch_size 64 --n_test 500 > /dev/null
        echo "  eval-${config}-${method} (dep=$dep)"
    done
done

echo ""
echo "========================================================"
echo "  Step 4: Eval target-only (depends on Step 2)"
echo "========================================================"
for entry in "${TARGET_TASKS[@]}"; do
    IFS=':' read -r task config methods <<< "$entry"
    IFS=',' read -ra t_list <<< "$methods"

    for method in "${t_list[@]}"; do
        dep="${TARGET_IDS["${task}:${method}"]}"
        depend_arg=""
        [[ "$DRY_RUN" != "true" ]] && depend_arg="afterok:$dep"
        # target-only models: train_filter is "${task}_val"
        submit "eval-target-${task}-${method}" "1:00:00" "$depend_arg" \
            SFT/eval/eval.sh --train "${task}_val" --method "$method" --batch_size 64 --n_test 500 > /dev/null
        echo "  eval-target-${task}-${method} (dep=$dep)"
    done
done

echo ""
echo "========================================================"
echo "  Done"
echo "========================================================"
