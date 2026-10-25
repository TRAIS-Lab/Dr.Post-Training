# Per-machine settings for this checkout. Copy to cluster_env.sh (gitignored) at the repo root and edit. Every script
# sources cluster_env.sh for paths and conda activation; submit.sh reads the SLURM_* values. Nothing tracked in the
# repository names a machine.

# Directory paths
export SCRATCH_DIR="$HOME/Project"   # results root: runs/checkpoints land under $SCRATCH_DIR/Dr.Post-Training/<setting>/runs (may equal CODE_DIR)
export CODE_DIR="$HOME/Project"      # the directory this repo is cloned into

# Conda environment activation (RLVR uses its own environment, see README "Environment")
activate_env() { source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate drpt; }
# If conda activate does not work in non-interactive shells:
# activate_env() { export PATH="$HOME/miniconda3/envs/drpt/bin:$PATH"; }

# SLURM defaults read by submit.sh
export SLURM_ACCOUNT="my-account"
export SLURM_PARTITION="gpu"
export SLURM_MAIL_USER="user@example.edu"

# Hugging Face cache (the defense task scripts expect HF_HOME to be set)
export HF_HOME="$HOME/.cache/huggingface"

# IFBench verifier for SFT/eval: defaults to the third_party/IFBench submodule; set only to use another checkout
# export DRPT_IFBENCH_REPO="$CODE_DIR/Dr.Post-Training/third_party/IFBench"
# NLTK punkt models for the IFEval/IFBench verifiers (downloaded here once if missing)
export NLTK_DATA="$CODE_DIR/Dr.Post-Training/data/nltk_data"
