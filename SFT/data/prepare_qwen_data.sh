#!/bin/bash
# prepare_qwen_data.sh: every CPU-side data artefact of the Qwen3-1.7B-Base settings, in dependency order
# (benchmarks -> math_ref/mbpp targets -> Dolci pools + precise_if -> math_persona -> Tulu 3 pool -> math_ref128 -> audit).
# Launch as a CPU Slurm job, e.g. GPUS=0 CPUS=32 MEM=128G TIME=6:00:00 ./submit.sh SFT/data/prepare_qwen_data.sh
# The GPU stage (Qwen3-32B re-solving of math_ref128 -> math_ref128_gen32b, "Math (curated)") is build_curated_math_target.sh.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"; activate_env
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/Dr.Post-Training/SFT/data}"; mkdir -p "$DATA_DIR"
NP="${SLURM_CPUS_PER_TASK:-16}"
prep() { python SFT/data/prepare_datasets.py --output_dir "$DATA_DIR" "$@" || exit 1; }
echo "== [1/7] benchmark files (IFEval, IFBench, MATH500, GSM8K, MBPP+)";  prep --datasets ifeval ifbench math500 gsm8k mbpp_plus
echo "== [2/7] targets the pools are decontaminated against (math_ref, mbpp)"; prep --datasets math_ref mbpp
echo "== [3/7] Dolci pools (3 x 32K) + precise_if target";                  prep --datasets dolci_pools --num_proc "$NP"
echo "== [4/7] math_persona target (Math (Dolci))";                          python SFT/data/build_math_persona_target.py --data_dir "$DATA_DIR" --num_proc "$NP" || exit 1
echo "== [5/7] Tulu 3 general pool (32K)";                                   prep --datasets tulu3_pool --num_proc "$NP"
echo "== [6/7] math_ref128 (base of Math (curated))";                        prep --datasets math_ref128
echo "== [7/7] leakage audit (must print CLEAN)";                            prep --datasets dolci_audit
echo "== done $(date -Is)"
