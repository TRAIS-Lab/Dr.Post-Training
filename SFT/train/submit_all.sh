#!/bin/bash
#
# Sweep launcher: main training + target-only + eval, submitted as ONE Slurm job
# array per stage (never a loop of sbatch calls — that trips the controller's
# per-user RPC limit and has crashed slurmd on shared clusters).
#
# Suites
#   paper : the 4 Llama-3.2-1B settings (alpaca_samsum Full/LoRA/MeSO, the rest
#           LoRA-only) + per-task target-only baselines + their evals.
#   dolci : the Qwen3-1.7B-Base Dolci capability settings (Full only:
#           FullTraining / LayerWiseSubset / GlobalSubset / BlockWiseSubset /
#           SublayerWiseSubset; benchmark eval via eval.sh --target). No target-only stage.
#
# Stages (each one sbatch --array):
#   1 main training            2 target-only training (paper suite only)
#   3 eval main  (afterok: 1)  4 eval target-only     (afterok: 2)
#
# Usage:
#   bash SFT/train/submit_all.sh --suite paper                       # 5 seeds, all 4 stages
#   bash SFT/train/submit_all.sh --suite dolci --seeds 42            # one seed
#   bash SFT/train/submit_all.sh --suite dolci --settings dolci_inst_if --stages 1
#   bash SFT/train/submit_all.sh --suite paper --dry-run             # manifests + sbatch lines only
#
# Options: --suite paper|dolci   --seeds "2 22 42 62 82"   --stages 1,2,3,4
#          --settings a,b,c (filter config dirs)   --n-vals "16 64" (one main run per D* size)
#          --methods m1,m2 (replace every selected setting's method list; main + eval stages only)
#          --lrs "5e-06 2e-05" (one main run per learning rate)
#          --max-concurrent K (array %K, default 8)   --dry-run
# Env:     STAGE1_DEPEND=afterany:<jobid> (queue the training array behind another job), EXCLUDE=<nodes>,
#          QOS (default high: runs don't checkpoint, so preemptible `low` restarts
#          them from scratch), PARTITION, TIME_MAIN/TIME_TARGET/TIME_EVAL overrides.
#
# Requires submit.sh to honour ARRAY/MANIFEST/DEPEND/JOB_NAME/TIME/GPUS/MEM/QOS
# (see the cluster-setup section of the top-level README).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source cluster_env.sh || { echo "ERROR: cluster_env.sh not found."; exit 1; }

# ---------------------------------------------------------------- CLI
SUITE="paper"
SEEDS="2 22 42 62 82"
STAGES="1,2,3,4"
SETTINGS_FILTER=""
N_VALS=""                 # e.g. "16 64": one main run per n_val (adds --n_val to train.sh)
METHODS_OVERRIDE=""       # e.g. "LayerWiseSubset-Full-f75,GlobalSubset-Full-f75": replaces each setting's method list
LRS=""                    # e.g. "5e-06 2e-05": one main run per learning rate (adds --lr to train.sh)
MAX_CONCURRENT=8
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --suite)          SUITE="$2"; shift 2 ;;
        --seeds)          SEEDS="$2"; shift 2 ;;
        --stages)         STAGES="$2"; shift 2 ;;
        --settings)       SETTINGS_FILTER="$2"; shift 2 ;;
        --n-vals)         N_VALS="$2"; shift 2 ;;
        --methods)        METHODS_OVERRIDE="$2"; shift 2 ;;
        --lrs)            LRS="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        -h|--help)        sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1 (use --help)"; exit 1 ;;
    esac
done
read -ra SEED_LIST <<< "$SEEDS"
read -ra N_VAL_LIST <<< "$N_VALS"
read -ra LR_LIST <<< "$LRS"
export QOS="${QOS:-high}"

# ---------------------------------------------------------------- Suites
# Main training: config_dir : train_filter : target : methods
# Target-only:   task : config_dir : methods : eval_steps  (paper suite only)
case "$SUITE" in
    paper)
        SETTINGS=(
            "alpaca_samsum:alpaca:samsum:FullTraining-Full,FullTraining-LoRA,FullTraining-MeSO,LayerWiseSubset-Full,LayerWiseSubset-LoRA,LayerWiseSubset-MeSO,GlobalSubset-Full,GlobalSubset-LoRA,GlobalSubset-MeSO"
            "less_tydiqa:less:tydiqa:FullTraining-LoRA,LayerWiseSubset-LoRA,GlobalSubset-LoRA"
            "triviaqa_nq:triviaqa:nq_open:FullTraining-LoRA,LayerWiseSubset-LoRA,GlobalSubset-LoRA"
            "less_squad:less:squad:FullTraining-LoRA,LayerWiseSubset-LoRA,GlobalSubset-LoRA"
        )
        TARGET_TASKS=(
            "samsum:alpaca_samsum:FullTraining-Full,FullTraining-LoRA,FullTraining-MeSO:26"
            "tydiqa:less_tydiqa:FullTraining-LoRA:12"
            "nq_open:triviaqa_nq:FullTraining-LoRA:11"
            "squad:less_squad:FullTraining-LoRA:11"
        )
        TARGET_EXTRA=""
        EVAL_MODE="task"          # eval.sh --train F --method M --batch_size 64 --n_test 500
        TIME_MAIN="${TIME_MAIN:-3:00:00}"; TIME_TARGET="${TIME_TARGET:-2:00:00}"; TIME_EVAL="${TIME_EVAL:-2:00:00}"
        ;;
    dolci)
        # The Qwen3-1.7B-Base settings of the paper (Sec. 4.1.2 + appendix): pool : target : arms (k = 4 / 2 / 6 of 8).
        SETTINGS=(
            "dolci_inst_if:dolci_instruction:precise_if:FullTraining-Full-eot,GlobalSubset-Full-eot,GlobalSubset-Full-f25-eot,GlobalSubset-Full-f75-eot,LayerWiseSubset-Full-eot,LayerWiseSubset-Full-f25-eot,LayerWiseSubset-Full-f75-eot"
            "dolci_mixed_if:dolci_mixed:precise_if:FullTraining-Full-eot,GlobalSubset-Full-eot,GlobalSubset-Full-f25-eot,GlobalSubset-Full-f75-eot,LayerWiseSubset-Full-eot,LayerWiseSubset-Full-f25-eot,LayerWiseSubset-Full-f75-eot"
            "tulu3_if:tulu3_general:precise_if:FullTraining-Full-eot,GlobalSubset-Full-eot,GlobalSubset-Full-f25-eot,GlobalSubset-Full-f75-eot,LayerWiseSubset-Full-eot,LayerWiseSubset-Full-f25-eot,LayerWiseSubset-Full-f75-eot"
            "dolci_reason_mathpersona:dolci_reasoning:math_persona:FullTraining-Full-eot,GlobalSubset-Full-eot,GlobalSubset-Full-f25-eot,GlobalSubset-Full-f75-eot,LayerWiseSubset-Full-eot,LayerWiseSubset-Full-f25-eot,LayerWiseSubset-Full-f75-eot"
            "dolci_mixed_mathpersona:dolci_mixed:math_persona:FullTraining-Full-eot,GlobalSubset-Full-eot,GlobalSubset-Full-f25-eot,GlobalSubset-Full-f75-eot,LayerWiseSubset-Full-eot,LayerWiseSubset-Full-f25-eot,LayerWiseSubset-Full-f75-eot"
            "tulu3_mathpersona:tulu3_general:math_persona:FullTraining-Full-eot,GlobalSubset-Full-eot,GlobalSubset-Full-f25-eot,GlobalSubset-Full-f75-eot,LayerWiseSubset-Full-eot,LayerWiseSubset-Full-f25-eot,LayerWiseSubset-Full-f75-eot"
            "dolci_reason_mathref128_gen32b:dolci_reasoning:math_ref128_gen32b:FullTraining-Full-eot,GlobalSubset-Full-eot,GlobalSubset-Full-f25-eot,GlobalSubset-Full-f75-eot,LayerWiseSubset-Full-eot,LayerWiseSubset-Full-f25-eot,LayerWiseSubset-Full-f75-eot"
        )
        # Target-Only arms are not part of the paper; none are submitted (see train_val_ablation.sh).
        TARGET_TASKS=()
        TARGET_EXTRA="--model Qwen/Qwen3-1.7B-Base --max_seq_length 2048 --gradient_checkpointing false --n_val 128 --val_seq_length_multiplier 0"
        EVAL_MODE="target"        # eval.sh --train POOL --target T --method M --batch_size 64 --max_new_tokens 2048
        TIME_MAIN="${TIME_MAIN:-16:00:00}"; TIME_TARGET="${TIME_TARGET:-8:00:00}"; TIME_EVAL="${TIME_EVAL:-8:00:00}"
        ;;
    *) echo "ERROR: unknown suite '$SUITE' (paper|dolci)"; exit 1 ;;
esac

if [[ -n "$SETTINGS_FILTER" ]]; then
    keep=(); IFS=',' read -ra want <<< "$SETTINGS_FILTER"
    for entry in "${SETTINGS[@]}"; do
        for w in "${want[@]}"; do [[ "${entry%%:*}" == "$w" ]] && keep+=("$entry"); done
    done
    SETTINGS=("${keep[@]}")
    # target-only rows are keyed by task; keep those whose config dir survived
    tkeep=()
    for entry in "${TARGET_TASKS[@]}"; do
        IFS=':' read -r _ cfg _ _ <<< "$entry"
        for s in "${SETTINGS[@]}"; do [[ "${s%%:*}" == "$cfg" ]] && tkeep+=("$entry"); done
    done
    TARGET_TASKS=("${tkeep[@]}")
fi
[[ ${#SETTINGS[@]} -gt 0 ]] || { echo "ERROR: no settings selected"; exit 1; }
if [[ -n "$METHODS_OVERRIDE" ]]; then
    tmp=()
    for entry in "${SETTINGS[@]}"; do
        IFS=':' read -r config train_filter target _ <<< "$entry"
        for m in ${METHODS_OVERRIDE//,/ }; do
            [[ -f "SFT/train/configs/$config/$m.yaml" ]] || { echo "ERROR: no config SFT/train/configs/$config/$m.yaml"; exit 1; }
        done
        tmp+=("$config:$train_filter:$target:$METHODS_OVERRIDE")
    done
    SETTINGS=("${tmp[@]}")
fi

want_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

TAG="$(date +%Y%m%d-%H%M%S)"
MANIFEST_DIR="$SCRATCH_DIR/Dr.Post-Training/SFT/manifests/${SUITE}-${TAG}"
mkdir -p "$MANIFEST_DIR"

# submit_array <stage-name> <manifest> <time> <depend-or-empty> <script> [fixed args...]
# Prints the job id (or DRY). One sbatch call per stage.
submit_array() {
    local name="$1" manifest="$2" time="$3" depend="$4" script="$5"; shift 5
    local n; n=$(wc -l < "$manifest")
    [[ "$n" -gt 0 ]] || { echo "SKIP"; return; }
    local array="0-$((n - 1))%${MAX_CONCURRENT}"
    local job_name="drpt-${SUITE}-${name}-${TAG}"
    echo "  [$name] $n tasks  array=$array  time=$time  depend=${depend:-none}  manifest=$manifest" >&2
    if [[ "$DRY_RUN" == "true" ]]; then
        DRY_RUN=1 ARRAY="$array" MANIFEST="$manifest" DEPEND="$depend" GPUS=1 MEM=128G TIME="$time" \
            JOB_NAME="$job_name" ./submit.sh "$script" "$@" >&2
        echo "DRY"
    else
        local out
        out=$(ARRAY="$array" MANIFEST="$manifest" DEPEND="$depend" GPUS=1 MEM=128G TIME="$time" \
              JOB_NAME="$job_name" ./submit.sh "$script" "$@") || { echo "ERROR: sbatch failed: $out" >&2; echo "FAIL"; return; }
        echo "$out" >&2
        echo "$out" | grep -oP '\d+' | tail -1
    fi
}

echo "========================================================"
echo "  Suite: $SUITE | seeds: ${SEED_LIST[*]} | n_vals: ${N_VALS:-config} | stages: $STAGES | qos: $QOS"
echo "  Manifests: $MANIFEST_DIR"
echo "========================================================"

# ---------------------------------------------------------------- Stage 1: main training
M1="$MANIFEST_DIR/stage1_main.txt"; : > "$M1"
for entry in "${SETTINGS[@]}"; do
    IFS=':' read -r config _ _ methods <<< "$entry"
    IFS=',' read -ra mlist <<< "$methods"
    for method in "${mlist[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            nv_opts=(""); [[ ${#N_VAL_LIST[@]} -gt 0 ]] && { nv_opts=(); for nv in "${N_VAL_LIST[@]}"; do nv_opts+=("--n_val $nv"); done; }
            lr_opts=(""); [[ ${#LR_LIST[@]} -gt 0 ]] && { lr_opts=(); for lr in "${LR_LIST[@]}"; do lr_opts+=("--lr $lr"); done; }
            for nvo in "${nv_opts[@]}"; do for lro in "${lr_opts[@]}"; do
                echo "-c configs/$config -m $method --seed $seed $nvo $lro" | sed 's/  */ /g; s/ *$//' >> "$M1"
            done; done
        done
    done
done
MAIN_JID=""
if want_stage 1; then
    MAIN_JID=$(submit_array main "$M1" "$TIME_MAIN" "${STAGE1_DEPEND:-}" SFT/train/train.sh)   # STAGE1_DEPEND=afterany:<jobid> queues the sweep behind other work
fi

# ---------------------------------------------------------------- Stage 2: target-only training
M2="$MANIFEST_DIR/stage2_target.txt"; : > "$M2"
for entry in "${TARGET_TASKS[@]}"; do
    IFS=':' read -r task config methods eval_steps <<< "$entry"
    IFS=',' read -ra tlist <<< "$methods"
    for method in "${tlist[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            echo "--task $task --config_dir $config --methods $method --seed $seed --eval_steps $eval_steps ${TARGET_EXTRA}" >> "$M2"
        done
    done
done
TARGET_JID=""
if want_stage 2 && [[ -s "$M2" ]]; then
    TARGET_JID=$(submit_array target "$M2" "$TIME_TARGET" "" SFT/train/train_val_ablation.sh)
fi

# ---------------------------------------------------------------- Stage 3: eval main
M3="$MANIFEST_DIR/stage3_eval_main.txt"; : > "$M3"
for entry in "${SETTINGS[@]}"; do
    IFS=':' read -r config train_filter target methods <<< "$entry"
    IFS=',' read -ra mlist <<< "$methods"
    for method in "${mlist[@]}"; do
        if [[ "$EVAL_MODE" == "target" ]]; then
            echo "--train $train_filter --target $target --method $method --batch_size 64 --max_new_tokens 2048" >> "$M3"
        else
            echo "--train $train_filter --method $method --batch_size 64 --n_test 500" >> "$M3"
        fi
    done
done
if want_stage 3; then
    dep=""; [[ -n "$MAIN_JID" && "$MAIN_JID" =~ ^[0-9]+$ ]] && dep="afterok:$MAIN_JID"
    submit_array eval-main "$M3" "$TIME_EVAL" "$dep" SFT/eval/eval.sh > /dev/null
fi

# ---------------------------------------------------------------- Stage 4: eval target-only
M4="$MANIFEST_DIR/stage4_eval_target.txt"; : > "$M4"
for entry in "${TARGET_TASKS[@]}"; do
    IFS=':' read -r task _ methods _ <<< "$entry"
    IFS=',' read -ra tlist <<< "$methods"
    for method in "${tlist[@]}"; do
        if [[ "$EVAL_MODE" == "target" ]]; then
            echo "--train ${task}_val --target $task --method $method --batch_size 64 --max_new_tokens 2048" >> "$M4"
        else
            echo "--train ${task}_val --method $method --batch_size 64 --n_test 500" >> "$M4"
        fi
    done
done
if want_stage 4 && [[ -s "$M4" ]]; then
    dep=""; [[ -n "$TARGET_JID" && "$TARGET_JID" =~ ^[0-9]+$ ]] && dep="afterok:$TARGET_JID"
    submit_array eval-target "$M4" "$TIME_EVAL" "$dep" SFT/eval/eval.sh > /dev/null
fi

echo ""
echo "Done. Monitor with:  squeue --me -o '%.10i %.30j %.8T %.10M %.6D %R'"
echo "Cancel a stage with: scancel -u \$USER --name=drpt-${SUITE}-<stage>-${TAG}"
