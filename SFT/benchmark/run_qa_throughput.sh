#!/bin/bash
# Real-job step throughput for the paper's QA setting: alpaca -> samsum on Llama-3.2-1B
# (n=8, T<=512 with dynamic padding, m=1, k=4), every curated method x scoring method x
# fine-tuning mode, with the fused kernels and with the PyTorch reference ops.
#
# Each job runs MAX_STEPS optimizer steps with the production launcher's exact command
# (SFT/train/train.sh --dry-run), evaluating every EVAL_STEPS steps.  The trainer logs
# train_wall_time (evaluation time excluded, CUDA-event timed) at every evaluation, so
# the per-step time is the slope between the first and the last evaluation; the first
# interval absorbs kernel compilation and allocator warm-up.  Jobs are spread over GPUS
# (one worker per GPU, round-robin); finished jobs (evaluation_results.json present) are
# skipped, so the script can be re-run to fill gaps.
#
#   bash SFT/benchmark/run_qa_throughput.sh                       # 3 baselines + 2 x 3 x 3 x 2 = 39 jobs on 4 GPUs
#   GPUS=1 FINETUNINGS=Full SCORINGS=pip bash SFT/benchmark/run_qa_throughput.sh
#   PROFILE=1 BACKENDS=cute bash SFT/benchmark/run_qa_throughput.sh   # 60 steps, kernel attribution of steps 30-39
#                                                                     # -> results/paper/qa_profile/<job>/profile.json
#
# The scoring method is overridden on the command line (--scoring_method); for
# 'compress' the compressor is the one of the LayerWiseSubset-<ft> config (64x64 for
# Full / LoRA, 512x512 for MeSO — the values of the actual experiments).
#
# Results: SFT/benchmark/results/paper/qa_throughput/<config>_<scoring>_<backend>/evaluation_results.json
# Summary: python SFT/benchmark/qa_throughput_table.py [--tex out.tex]
set -uo pipefail
set -f   # commands contain 'normal-64*64'; never glob
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-$(command -v python)}"
TORCHRUN="$(dirname "$PYTHON")/torchrun"
GPUS="${GPUS:-0,1,2,3}"
PROFILE="${PROFILE:-0}"            # 1 = kernel-level profile of steps PROFILE_STEPS -> <run_dir>/profile.json
PROFILE_STEPS="${PROFILE_STEPS:-30:40}"
if [[ "$PROFILE" == 1 ]]; then
    MAX_STEPS="${MAX_STEPS:-60}"; EVAL_STEPS="${EVAL_STEPS:-20}"
else
    MAX_STEPS="${MAX_STEPS:-130}"; EVAL_STEPS="${EVAL_STEPS:-26}"
fi
N_EVAL="${N_EVAL:-64}"          # held-out loss set; evaluation time is excluded from the metric anyway
CONFIG="${CONFIG:-configs/alpaca_samsum}"
FINETUNINGS="${FINETUNINGS:-Full,LoRA,MeSO}"
METHODS="${METHODS:-FullTraining,LayerWiseSubset,GlobalSubset}"
SCORINGS="${SCORINGS:-compress,gip,pip}"
BACKENDS="${BACKENDS:-cute,off}"   # off = reference PyTorch ops (DRPT_KERNEL_BACKEND=off)
if [[ "$PROFILE" == 1 ]]; then
    OUT="${OUT:-$REPO_ROOT/SFT/benchmark/results/paper/qa_profile}"
else
    OUT="${OUT:-$REPO_ROOT/SFT/benchmark/results/paper/qa_throughput}"
fi
mkdir -p "$OUT"
CONFIG_ABS="$CONFIG"; [[ "$CONFIG_ABS" != /* ]] && CONFIG_ABS="$REPO_ROOT/SFT/train/$CONFIG"

IFS=',' read -ra fts <<< "$FINETUNINGS"
IFS=',' read -ra methods <<< "$METHODS"
IFS=',' read -ra scorings <<< "$SCORINGS"
IFS=',' read -ra backends <<< "$BACKENDS"
IFS=',' read -ra gpus <<< "$GPUS"

# ---- job list: "config|scoring|backend" -------------------------------------------
jobs=()
for ft in "${fts[@]}"; do
    for method in "${methods[@]}"; do
        cfg="${method}-${ft}"
        [[ -f "$CONFIG_ABS/$cfg.yaml" ]] || { echo "[warn] no config $cfg.yaml"; continue; }
        if [[ "$method" == FullTraining ]]; then
            jobs+=("$cfg|na|${backends[0]}")         # no custom backward: backend irrelevant, run once
            continue
        fi
        for scoring in "${scorings[@]}"; do
            for backend in "${backends[@]}"; do
                jobs+=("$cfg|$scoring|$backend")
            done
        done
    done
done

run_job() {   # gpu config scoring backend
    local gpu="$1" cfg="$2" scoring="$3" backend="$4"
    local ft="${cfg##*-}"
    local run_dir="$OUT/${cfg}_${scoring}_${backend}"
    local marker="$run_dir/evaluation_results.json"
    [[ "$PROFILE" == 1 ]] && marker="$run_dir/profile.json"
    if [[ -f "$marker" ]]; then
        echo "[skip] $cfg $scoring [$backend] (exists)"; return
    fi
    local cmd
    cmd="$(bash SFT/train/train.sh -c "$CONFIG" -m "$cfg" --dry-run 2>/dev/null | grep '^\[DRY-RUN\]' | head -1 | sed 's/^\[DRY-RUN\] //')"
    [[ -n "$cmd" ]] || { echo "[error] no command for $cfg"; return; }
    cmd="$(echo "$cmd" | sed -E "s#--output_dir [^ ]+#--output_dir $run_dir#; s#--eval_steps [0-9]+#--eval_steps $EVAL_STEPS#; s#--n_eval [0-9]+#--n_eval $N_EVAL#")"
    if [[ "$scoring" != na ]]; then
        cmd="$(echo "$cmd" | sed -E "s#--scoring_method [a-z]+#--scoring_method $scoring#; s# --score_compression [^ ]+##")"
        if [[ "$scoring" == compress ]]; then
            local comp
            comp="$(grep -A3 '^scoring:' "$CONFIG_ABS/LayerWiseSubset-$ft.yaml" | grep 'compression:' | head -1 | sed 's/.*compression: *//; s/[\"'"'"' ]//g')"
            cmd="$cmd --score_compression ${comp:-normal-64*64}"
        fi
    fi
    cmd="${cmd/torchrun/$TORCHRUN} --max_steps $MAX_STEPS"
    mkdir -p "$run_dir"
    echo "[$(date +%H:%M) gpu$gpu] $cfg $scoring [$backend] -> $(basename "$run_dir")"
    echo "$cmd" > "$run_dir/command.txt"
    local penv=()
    [[ "$PROFILE" == 1 ]] && penv=(DRPT_PROFILE_STEPS="$PROFILE_STEPS" DRPT_PROFILE_TRACE="$run_dir/trace.json")
    env "${penv[@]}" DRPT_KERNEL_BACKEND="$backend" CUDA_VISIBLE_DEVICES="$gpu" bash -fc "$cmd" > "$run_dir/train.log" 2>&1
    local rc=$?
    # train.py saves the fine-tuned weights / adapters (GBs) into output_dir; keep only the timings
    find "$run_dir" -mindepth 1 ! -name evaluation_results.json ! -name profile.json ! -name command.txt ! -name train.log \
         ! -name trace.json -delete 2>/dev/null
    local n_eval
    n_eval=$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$run_dir/evaluation_results.json" 2>/dev/null || echo 0)
    echo "[$(date +%H:%M) gpu$gpu]   $cfg $scoring [$backend] exit $rc ($n_eval evaluations)"
    [[ "$PROFILE" == 1 ]] && grep -h "\[drpt profile\]" "$run_dir/train.log" | sed 's/^/    /'
}

echo "${#jobs[@]} jobs on GPUs $GPUS"
ngpu=${#gpus[@]}
for ((w = 0; w < ngpu; w++)); do
    (
        for ((j = w; j < ${#jobs[@]}; j += ngpu)); do
            IFS='|' read -r cfg scoring backend <<< "${jobs[$j]}"
            run_job "${gpus[$w]}" "$cfg" "$scoring" "$backend"
        done
    ) &
done
wait
if [[ "$PROFILE" == 1 ]]; then
    "$PYTHON" SFT/benchmark/qa_decomposition_table.py --profile-dir "$OUT"
else
    "$PYTHON" SFT/benchmark/qa_throughput_table.py --dir "$OUT"
fi
