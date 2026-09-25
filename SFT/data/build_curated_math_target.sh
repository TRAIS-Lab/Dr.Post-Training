#!/bin/bash
# build_curated_math_target.sh: the "Math (curated)" target math_ref128_gen32b = math_ref128 re-solved by Qwen3-32B and verified.
#   stage 1  SFT/data/gen_target_candidates.py  8 candidate solutions per D* row (2 per held-out row), non-thinking,
#            T 0.7 / top-p 0.8 / top-k 20, seed 42, MATH500 boxed template; vLLM in $VLLM_ENV, HF generate fallback in the drpt env
#   stage 2  SFT/data/build_rewrite_target.py    shortest candidate whose final answer Math-Verify matches the reference;
#            D* rows without a correct candidate dropped, held-out rows keep the reference (manifest.json records every decision)
# Needs eval/math_ref128 (SFT/data/prepare_qwen_data.sh). One GPU, ~1-2 h for Qwen3-32B:
#   GPUS=1 TIME=6:00:00 ./submit.sh SFT/data/build_curated_math_target.sh
# Env: GENERATOR (hub id or local snapshot; default Qwen/Qwen3-32B), TAG (default 32b), JOBS (default "math_ref128:solve"),
#      VLLM_ENV (conda env with vLLM; default drpt_rlvr), DATA_DIR. Existing candidate files are reused (requeue-safe).
# Stage 1 is sampling and is not bit-reproducible across vLLM/GPU versions; stage 2 is deterministic given the candidates,
# which is why the candidate files are kept next to the data ($DATA_DIR/candidates/).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"; activate_env
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
GEN="${GENERATOR:-Qwen/Qwen3-32B}"; TAG="${TAG:-32b}"; VLLM_ENV="${VLLM_ENV:-drpt_rlvr}"
read -r -a JOBS <<< "${JOBS:-math_ref128:solve}"
GEN_NAME=$(basename "$GEN"); [[ "$GEN" == */snapshots/* ]] && GEN_NAME=$(basename "$(dirname "$(dirname "$GEN")")" | sed 's/^models--Qwen--//')
DATA="${DATA_DIR:-$SCRATCH_DIR/Dr.Post-Training/SFT/data}"; CAND="$DATA/candidates"; mkdir -p "$CAND"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "[curated-math] generator=$GEN ($GEN_NAME) tag=$TAG jobs=${JOBS[*]} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none}"
common=(--data_dir "$DATA" --generator "$GEN" --generator_name "$GEN_NAME" --jobs "${JOBS[@]}" --out_dir "$CAND" --tag "$TAG"
        --num_samples 8 --num_samples_test 2 --temperature 0.7 --top_p 0.8 --top_k 20 --seed 42)
if conda activate "$VLLM_ENV" 2>/dev/null; then
  python SFT/data/gen_target_candidates.py "${common[@]}" --backend vllm --max_model_len 8192 --gpu_memory_utilization 0.90; status=$?
else
  echo "[curated-math] no conda env $VLLM_ENV; using HF generate"; status=1
fi
activate_env
if [ "$status" -ne 0 ]; then
  python SFT/data/gen_target_candidates.py "${common[@]}" --backend hf --batch_size 32 || { echo "[curated-math] stage 1 failed"; exit 1; }
fi
for job in "${JOBS[@]}"; do
  t=${job%%:*}; m=${job##*:}
  case $m in solve|gen) mt=gen ;; rewrite|rw) mt=rw ;; *) echo "bad mode $m"; exit 1 ;; esac
  name=${t}_${mt}${TAG}
  python SFT/data/build_rewrite_target.py --data_dir "$DATA" --source_target "$t" --name "$name" --candidates "$CAND/$name.jsonl" --overwrite || exit 1
done
echo "[curated-math] done $(date -Is)"
