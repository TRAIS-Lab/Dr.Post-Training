#!/bin/bash

# =============================================================================
# LR Sweep Collect — Target-Only Ablation
# =============================================================================
# Scans output dirs from lr_sweep_submit_val.sh, extracts val_loss
# (last entry of evaluation_results.json), picks the smallest LR within
# --lr_margin of the best loss, and writes it to a marker file.
#
# Output dir pattern:
#   {task}_val_{task}-{model}-FullTraining-Full-ms*-lr*-b{bs}-v{nval}-s{seed}
#
# Usage:
#   bash lr_sweep_collect_val.sh --task triviaqa --model meta-llama/Llama-3.2-1B
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Dr.Post-Training

# CLI defaults
task=""
subject=""
finetuning="Full"  # Full | LoRA | MeSO
model="meta-llama/Llama-3.2-1B"
seed=42
n_val=16
batch_size=8
max_steps=""  # if set, restrict to ms{N} dirs (disambiguates same-task setups)
lr_margin=0.01
dry_run=false
cleanup=true
out_file=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)        task="$2"; shift 2 ;;
        --subject)     subject="$2"; shift 2 ;;
        --finetuning)  finetuning="$2"; shift 2 ;;
        --model)       model="$2"; shift 2 ;;
        --seed)        seed="$2"; shift 2 ;;
        --n_val)       n_val="$2"; shift 2 ;;
        --batch_size)  batch_size="$2"; shift 2 ;;
        --max_steps)   max_steps="$2"; shift 2 ;;
        --lr_margin)   lr_margin="$2"; shift 2 ;;
        --out_file)    out_file="$2"; shift 2 ;;
        --no-cleanup)  cleanup=false; shift ;;
        --dry-run)     dry_run=true; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash lr_sweep_collect_val.sh --task <task> [options]

Required:
  --task <task>          Task name (triviaqa, tydiqa, ...)

Options:
  --model <model>        HF model name (default: meta-llama/Llama-3.2-1B)
  --finetuning <ft>      Full | LoRA | MeSO (default: Full)
  --seed <seed>          Seed used in sweep (default: 42)
  --n_val <n>            Val samples (default: 16)
  --batch_size <bs>      Batch size (default: 8)
  --max_steps <n>        Restrict pattern to ms{N} (default: any). Use this to
                         disambiguate when the same task has multiple matched-
                         budget settings (e.g. triviaqa under nq_open vs tulu3).
  --lr_margin <pct>      Stability margin (default: 0.01 = 1%)
  --out_file <path>      Where to write best LR (default:
                         SFT/train/configs/<config_dir>/Target-only.lr.txt)
  --no-cleanup           Keep model weights after collection
  --dry-run              Don't write file or clean up
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1 (use --help)"; exit 1 ;;
    esac
done

if [[ -z "$task" ]]; then
    echo "ERROR: --task is required"
    exit 1
fi

# Map task -> config dir for default out_file path
declare -A CONFIG_DIR_FOR_TASK=(
    ["triviaqa"]="nq_triviaqa"
    ["tydiqa"]="tulu3_tydiqa"
    ["samsum"]="alpaca_samsum"
    ["mmlu"]="less_mmlu"
)

config_dir_name="${CONFIG_DIR_FOR_TASK[$task]:-}"

# Per-subject suffix for tasks like MMLU where target-only is per-subject.
subj_tag=""
subj_filename=""
if [[ -n "$subject" ]]; then
    subj_tag="_${subject}"
    subj_filename=".${subject}"
fi

# Per-finetuning suffix on out_file (Full → no suffix to keep backward compat).
ft_filename=""
[[ "$finetuning" != "Full" ]] && ft_filename=".${finetuning}"

if [[ -z "$out_file" ]]; then
    if [[ -n "$config_dir_name" ]]; then
        out_file="$CODE_DIR/Dr.Post-Training/SFT/train/configs/${config_dir_name}/Target-only${subj_filename}${ft_filename}.lr.txt"
    else
        out_file="$CODE_DIR/Dr.Post-Training/SFT/train/lr/results/val_${task}${subj_tag}_${finetuning}_$(basename "$model").lr.txt"
        mkdir -p "$(dirname "$out_file")"
    fi
fi

model_name=$(basename "$model")

# Output pattern from train_val_ablation.sh:
#   {task}{subj_tag}_val_{task}{subj_tag}-{model_name}-FullTraining-{Full|LoRA|MeSO}-ms*-lr*-b{bs}-v{nval}-s{seed}
ms_part="ms*"
[[ -n "$max_steps" ]] && ms_part="ms${max_steps}"
pattern="${task}${subj_tag}_val_${task}${subj_tag}-${model_name}-FullTraining-${finetuning}-${ms_part}-lr*-b${batch_size}-v${n_val}-s${seed}"

echo ""
echo "========================================================"
echo "  Target-Only LR Sweep Collect"
echo "========================================================"
echo "Task:        $task${subject:+ (subject=$subject)}"
echo "Finetuning:  $finetuning"
echo "Model:       $model"
echo "Pattern:     $pattern"
echo "Out file:    $out_file"
echo "lr_margin:   $lr_margin"
echo "========================================================"

cleanup_model_weights() {
    local d="$1"
    [[ ! -d "$d" ]] && return
    rm -f "$d"/model*.safetensors "$d"/pytorch_model.bin \
          "$d"/adapter_model.safetensors "$d"/adapter_model.bin "$d"/adapter_config.json \
          "$d"/tokenizer.json "$d"/tokenizer_config.json "$d"/special_tokens_map.json \
          "$d"/config.json "$d"/generation_config.json "$d"/README.md 2>/dev/null
    rm -rf "$d"/runs 2>/dev/null
}

sweep_dirs=$(find "$SCRATCH_DIR/Dr.Post-Training/SFT/" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | sort)

if [[ -z "$sweep_dirs" ]]; then
    echo "ERROR: No sweep dirs found matching pattern."
    echo "Looked in: $SCRATCH_DIR/Dr.Post-Training/SFT/"
    exit 1
fi

declare -A lr_losses=()
n_found=0
n_missing=0

for dir in $sweep_dirs; do
    dir_name=$(basename "$dir")
    lr=$(echo "$dir_name" | grep -oP '(?<=-lr).*?(?=-b)')
    if [[ -z "$lr" ]]; then continue; fi

    eval_json="$dir/evaluation_results.json"
    if [[ ! -f "$eval_json" ]]; then
        echo "  WARNING: No evaluation_results.json in $dir_name"
        n_missing=$((n_missing + 1))
        continue
    fi

    # For target-only training, val_dataset == train_dataset, so val_loss is just
    # training loss. Use eval_loss (computed on --eval_split lr held-out split).
    loss=$(python3 -c "
import json, sys
try:
    with open('$eval_json') as f:
        results = json.load(f)
    if results:
        v = results[-1].get('eval_loss')
        if v is not None:
            print(f'{v:.10e}')
            sys.exit(0)
except Exception: pass
sys.exit(1)
" 2>/dev/null)

    if [[ $? -eq 0 && -n "$loss" ]]; then
        lr_losses[$lr]="$loss"
        n_found=$((n_found + 1))
        echo "  LR=$lr  loss=$loss"
    else
        echo "  WARNING: Could not extract loss from $dir_name"
        n_missing=$((n_missing + 1))
    fi
done

echo ""
echo "Found: $n_found results, Missing: $n_missing"
if [[ $n_found -eq 0 ]]; then
    echo "ERROR: No valid results."
    exit 1
fi

# Pick smallest LR within margin of best
best_result=$(python3 -c "
import sys
margin = $lr_margin
pairs = []
$(for lr in "${!lr_losses[@]}"; do echo "pairs.append(($lr, ${lr_losses[$lr]}))"; done)
pairs.sort(key=lambda x: x[0])
best_loss = min(p[1] for p in pairs)
for lr, loss in pairs:
    if loss <= best_loss * (1 + margin):
        print(f'{lr:.2e} {loss:.10e}')
        sys.exit(0)
" 2>/dev/null)

best_lr=$(echo "$best_result" | awk '{print $1}')
best_loss=$(echo "$best_result" | awk '{print $2}')

echo ""
echo ">>> Best Target-Only LR: $best_lr (loss: $best_loss, margin: $lr_margin)"

if [[ "$dry_run" == "true" ]]; then
    echo "[DRY-RUN] Would write: $out_file <- $best_lr"
else
    mkdir -p "$(dirname "$out_file")"
    echo "$best_lr" > "$out_file"
    echo "Wrote $out_file"
fi

if [[ "$cleanup" == "true" && "$dry_run" != "true" ]]; then
    for dir in $sweep_dirs; do
        cleanup_model_weights "$dir"
    done
    echo "Cleaned up model weights from sweep dirs."
fi

echo ""
echo "========================================================"
echo "Use this LR in stage 2:"
if [[ -n "$subject" ]]; then
    echo "  bash SFT/train/train_val_ablation.sh --task $task --subject $subject --methods FullTraining-Full \\"
    echo "    --model $model --lr \$(cat $out_file) --seed <S>"
else
    echo "  bash SFT/train/train_val_ablation.sh --task $task --methods FullTraining-Full \\"
    echo "    --model $model --lr \$(cat $out_file) --seed <S>"
fi
echo "========================================================"
