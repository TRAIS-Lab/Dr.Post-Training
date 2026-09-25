#!/bin/bash
# qa_reeval_task.sh <run_dir>   (one Slurm array task; wrapper SFT/train/slurm/qa_array.sbatch, manifests SFT/train/manifests/qa_reeval_*.txt)
# Re-evaluates one Llama-3.2-1B question-answering run with the paper protocol (SFT/eval/eval.sh --batch_size 64 --n_test 500) after the
# evaluation fixes: evaluation prompts no longer get a <|begin_of_text|> the training sequences never had (SFT/eval/utils.py), and samsum /
# tydiqa decode greedily like the closed-book tasks (SFT/eval/tasks/{samsum,tydiqa}.py).
# The previous result file is kept as <task>_results.bos.json; a result file that already carries "prompt_encoding" is left alone.
set -u
d=${1:?usage: qa_reeval_task.sh <run_dir>}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"; activate_env
name=$(basename "$d")
case "$name" in
  alpaca_samsum-*|samsum_val_samsum-*)   target=samsum ;;
  less_tydiqa-*|tydiqa_val_tydiqa-*)     target=tydiqa ;;
  triviaqa_nq_open-*|nq_open_val_*)      target=nq_open ;;
  less_squad-*|squad_val_squad-*)        target=squad ;;
  *) echo "[reeval] ERROR: cannot infer the target task of $name"; exit 1 ;;
esac
res="$d/${target}_results.json"
ls "$d"/*.safetensors >/dev/null 2>&1 || { echo "[reeval] ERROR: no model in $d"; exit 1; }
done_check() { grep -q '"prompt_encoding": "chat_template_no_bos+greedy"' "$1" && { [ "$target" != tydiqa ] || grep -q '"generation_batch_size": 16' "$1"; }; }
if [ -f "$res" ] && done_check "$res"; then echo "[reeval] $name already evaluated with the fixed protocol"; exit 0; fi
if [ -f "$res" ]; then
  [ -f "$d/${target}_results.bos.json" ] || mv "$res" "$d/${target}_results.bos.json"   # keep the pre-fix result once
  rm -f "$res"
fi
echo "[reeval] $name -> $target ($(date))"
bash SFT/eval/eval.sh --model_path "$d" --task "$target" --batch_size "${EVAL_BATCH:-64}" --n_test "${EVAL_N_TEST:-500}"
[ -f "$res" ] && done_check "$res" || { echo "[reeval] ERROR: no fixed-protocol result in $d"; exit 1; }
