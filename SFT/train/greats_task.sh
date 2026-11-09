#!/bin/bash
# greats_task.sh <setting> <method_yaml> <seed>   (one Slurm array task; wrapper slurm/qa_array_deltaai.sbatch)
# One run of a GREATS-grounded setting (configs/less_tydiqa_mistral7b_greats, configs/alpaca_samsum_llama3_8b_greats): trains the
# arm with SFT/train/train.sh, then evaluates the final adapter and every checkpoint-<step> it wrote (save_at_steps) on the target's
# GREATS-protocol task (SFT/eval/tasks/greats.py): tydiqa_less on GREATS's 500 test rows, samsum_tt on the whole test split.
# Adapters are evaluated unmerged (--lora_merge false): merged into bf16 weights, the SAMSum adapter (alpha/r = 1/80) is rounded
# away (held-out perplexity 6.9 unmerged vs 9.2 merged, and the merged model echoes the dialogue instead of summarizing).
# Finished stages are skipped, so a task can be re-run.
# Run dir: $RUNS_ROOT/<pool>_<target>-<model>-<method>[-2nd]-p<pct>-lr<lr>-b<bs>-v<n_val>-s<seed>.
set -u
cfg=$1; method=$2; seed=$3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"; activate_env
RUNS_ROOT=${RUNS_ROOT:-$SCRATCH_DIR/Dr.Post-Training/SFT/runs}
cdir=SFT/train/configs/$cfg
yval() { grep -E "^$2:" "$1" | head -1 | cut -d: -f2- | xargs; }
pool=$(yval $cdir/defaults.yaml train_dataset); target=$(yval $cdir/defaults.yaml target_task)
model=$(basename "$(yval $cdir/defaults.yaml model)"); pct=$(yval $cdir/defaults.yaml percentage)
bs=$(yval $cdir/defaults.yaml batch_size); nval=$(yval $cdir/defaults.yaml n_val)
lr=$(yval $cdir/$method.yaml learning_rate); [ -z "$lr" ] && lr=$(yval $cdir/defaults.yaml learning_rate)
msuf=""; [ "$(yval $cdir/$method.yaml use_second_order)" = true ] && msuf=-2nd   # train.sh names second-order arms <method>-2nd
d=$RUNS_ROOT/${pool}_${target}-${model}-${method}${msuf}-p${pct}-lr${lr}-b${bs}-v${nval}-s${seed}
case "$target" in tydiqa_less) n_test=500 ;; *) n_test=-1 ;; esac
echo "[greats_task] $cfg / $method / seed $seed -> $d"
if ls "$d"/*.safetensors >/dev/null 2>&1; then echo "[greats_task] model exists, skipping training"; else
  bash SFT/train/train.sh -c configs/$cfg -m "$method" --seed "$seed" --runs_root "$RUNS_ROOT"; fi
ls "$d"/*.safetensors >/dev/null 2>&1 || { echo "[greats_task] ERROR: no model saved in $d"; exit 1; }
for m in "$d" "$d"/checkpoint-*; do
  [ -d "$m" ] || continue
  if [ -f "$m/${target}_results.json" ]; then echo "[greats_task] eval exists: $m"; continue; fi
  bash SFT/eval/eval.sh --model_path "$m" --task "$target" --batch_size "${EVAL_BATCH:-64}" --n_test "$n_test" --lora_merge false
done
