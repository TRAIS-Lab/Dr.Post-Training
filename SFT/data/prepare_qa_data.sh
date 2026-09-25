#!/bin/bash
# prepare_qa_data.sh: create the alpaca pool and the samsum target/eval splits under $SCRATCH_DIR/Dr.Post-Training/SFT/data
# (prepare_datasets.py writes RELATIVE to the repo root; the files are moved afterwards). CPU only, downloads from the Hub.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"; source "$REPO_ROOT/cluster_env.sh"; activate_env
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
D=$SCRATCH_DIR/Dr.Post-Training/SFT/data
[ -f "$D/train/alpaca/alpaca_data.jsonl" ] && [ -f "$D/eval/samsum/samsum_test_data.jsonl" ] && { echo "[prep_qa] data exists"; exit 0; }
python SFT/data/prepare_datasets.py --datasets alpaca samsum
mkdir -p "$D/train" "$D/eval"
for p in train/alpaca train/samsum eval/samsum; do [ -d "SFT/data/$p" ] && [ ! -e "$D/$p" ] && mv "SFT/data/$p" "$D/$p"; done
ls "$D/train/alpaca" "$D/eval/samsum"
