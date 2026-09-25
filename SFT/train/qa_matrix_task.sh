#!/bin/bash
# qa_matrix_task.sh <setting> <method_yaml> <seed>   (one Slurm array task; manifests from SFT/train/qa_matrix_manifest.py, wrapper slurm/qa_array.sbatch)
# One cell of the question-answering matrix (Llama-3.2-1B; 4 settings x {Full, LoRA, MeSO} x {Full-Training, Global, Block-Wise, Sublayer-Wise,
# Layer-Wise Subset} x {exact GIP, compressed 64x64} plus the Random Subset / layer-normalized controls; 5 seeds): trains the arm with
# SFT/train/train.sh and evaluates it on the setting's target task in the same allocation with the paper protocol
# (SFT/eval/eval.sh --task <target> --batch_size 64 --n_test 500). Finished stages are skipped, so a task can be re-run.
# Run dir: $RUNS_ROOT/<train>_<target>-<model>-<method>-p<pct>-lr<lr>-b<bs>-v<n_val>-s<seed>; the paper tables are read from these
# run dirs by SFT/tables/qa_downstream.py. Full-parameter / MeSO arms load a 1.24B-parameter fp32 model + AdamW states (48 GB host memory).
# N_VAL=<k> in the environment trains with k target examples instead of the config's 16 (run dir -v<k>-; the paper scanner only reads v16):
#   N_VAL=4 sbatch --export=ALL,N_VAL=4 --array=... slurm/qa_array.sbatch manifests/qa_nval_sweep.txt SFT/train/qa_matrix_task.sh
set -u
cfg=$1; method=$2; seed=$3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"; activate_env
RUNS_ROOT=${RUNS_ROOT:-$SCRATCH_DIR/Dr.Post-Training/SFT/runs_v2}
cdir=SFT/train/configs/$cfg
yval() { grep -E "^$2:" "$1" | head -1 | cut -d: -f2- | xargs; }
train=$(yval $cdir/defaults.yaml train_dataset); target=$(yval $cdir/defaults.yaml target_task)
model=$(basename "$(yval $cdir/defaults.yaml model)"); pct=$(yval $cdir/defaults.yaml percentage)
bs=$(yval $cdir/defaults.yaml batch_size); nval=${N_VAL:-$(yval $cdir/defaults.yaml n_val)}   # N_VAL: target-set size sweep (SFT/tables/qa_nval_sweep.py)
lr=$(yval $cdir/$method.yaml learning_rate); [ -z "$lr" ] && lr=$(yval $cdir/defaults.yaml learning_rate)
d=$RUNS_ROOT/${train}_${target}-${model}-${method}-p${pct}-lr${lr}-b${bs}-v${nval}-s${seed}
echo "[qa_matrix] $cfg / $method / seed $seed -> $d"
if ls "$d"/*.safetensors >/dev/null 2>&1; then echo "[qa_matrix] model exists, skipping training"; else
  # Periodic evaluation (500 held-out rows) every `eval_steps` costs ~50 min per task and does not affect training; the matrix evaluates
  # EVAL_STEPS_MULT (default 5) times less often -> ~20 curve points, identical final models and metrics. EVAL_STEPS_MULT=1
  # keeps the dense evaluation cadence of the config.
  es=$(yval $cdir/defaults.yaml eval_steps); es=$(( ${es:-12} * ${EVAL_STEPS_MULT:-5} ))
  bash SFT/train/train.sh -c configs/$cfg -m "$method" --seed "$seed" --runs_root "$RUNS_ROOT" --eval_steps "$es" ${N_VAL:+--n_val $N_VAL}; fi
ls "$d"/*.safetensors >/dev/null 2>&1 || { echo "[qa_matrix] ERROR: no model saved in $d"; exit 1; }
[ -f "$d/${target}_results.json" ] && { echo "[qa_matrix] eval exists"; exit 0; }
bash SFT/eval/eval.sh --model_path "$d" --task "$target" --batch_size "${EVAL_BATCH:-64}" --n_test "${EVAL_N_TEST:-500}"
