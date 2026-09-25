#!/bin/bash
# qa_target_only_task.sh <setting> <FullTraining-Full|FullTraining-LoRA|FullTraining-MeSO> <seed> [steps]   (one Slurm array task; manifest
# qa_target_only.txt from SFT/train/qa_matrix_manifest.py, wrapper slurm/qa_array.sbatch)
# Optional 4th argument <steps>: a short step budget instead of the main run's (e.g. 50 = the oracle early-stopping probe,
# evaluation perplexity every 5 steps). Launch those with RUNS_ROOT set to a directory the table scanner does not read
# (SFT/tables/qa_downstream.py --scan matches every ms<steps> dir directly under runs_v2), e.g. runs_v2/target_only_ms50.
# N_VAL=<k> in the environment trains on k target examples instead of 16 (the first k rows of the seed's shuffled D*, the same rows a
# curated run with N_VAL=k scores against; run dir -v<k>-, not read by the paper scanner): the target-set size sweep, SFT/tables/qa_nval_sweep.py.
# Target-Only Update for the question-answering tables (adxtab:sft-prelim-ppl): standard training on the 16 held-out target examples
# (the D* subset a curated run with the same seed uses) for the main run's step budget, with the held-out evaluation perplexity logged
# ~26 times (the blow-up curve), then the paper-protocol downstream eval in the same allocation. Launcher: SFT/train/train_val_ablation.sh
# (yaml learning rate: Full 1e-5, LoRA 1e-4, MeSO 5e-5). Run dir: $RUNS_ROOT/<task>_val_<task>-Llama-3.2-1B-<method>-ms<steps>-lr<lr>-b8-v16-s<seed>.
set -u
cfg=$1; method=$2; seed=$3; steps=${4:-}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"; activate_env
RUNS_ROOT=${RUNS_ROOT:-$SCRATCH_DIR/Dr.Post-Training/SFT/runs_v2}
declare -A TASK=( [alpaca_samsum]=samsum [less_tydiqa]=tydiqa [triviaqa_nq]=nq_open [less_squad]=squad )
declare -A EVERY=( [samsum]=100 [tydiqa]=45 [nq_open]=44 [squad]=44 )   # eval every ~1/26 of the step budget
task=${TASK[$cfg]}
ms="ms*"; extra=""; every=${EVERY[$task]}; nval=${N_VAL:-16}; [ -n "${N_VAL:-}" ] && extra="--n_val $N_VAL"   # N_VAL: target-set size sweep
[ -n "$steps" ] && { ms="ms${steps}"; extra="$extra --max_steps $steps"; every=$(( steps >= 50 ? steps / 10 : 1 )); }
echo "[target_only] $cfg / $method / seed $seed (task $task${steps:+, $steps steps})"
d=$(ls -d "$RUNS_ROOT"/${task}_val_${task}-Llama-3.2-1B-${method}-${ms}-lr*-b8-v${nval}-s${seed} 2>/dev/null | head -1)
if [ -n "$d" ] && ls "$d"/*.safetensors >/dev/null 2>&1; then echo "[target_only] model exists, skipping training: $d"; else
  bash SFT/train/train_val_ablation.sh --task "$task" --methods "$method" --config_dir "configs/$cfg" --seed "$seed" --runs_root "$RUNS_ROOT" --eval_steps "$every" $extra
  d=$(ls -d "$RUNS_ROOT"/${task}_val_${task}-Llama-3.2-1B-${method}-${ms}-lr*-b8-v${nval}-s${seed} 2>/dev/null | head -1); fi
[ -n "$d" ] && ls "$d"/*.safetensors >/dev/null 2>&1 || { echo "[target_only] ERROR: no model saved for $cfg $method $seed"; exit 1; }
[ -f "$d/${task}_results.json" ] && { echo "[target_only] eval exists"; exit 0; }
bash SFT/eval/eval.sh --model_path "$d" --task "$task" --batch_size "${EVAL_BATCH:-64}" --n_test "${EVAL_N_TEST:-500}"
