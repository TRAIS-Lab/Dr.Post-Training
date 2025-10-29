#!/bin/bash
#SBATCH --job-name=MMLU
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4 # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdzy-delta-gpu
#SBATCH --time=2:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Efficient-Fine-Tuning/experiment/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

cd ~/Project/Efficient-Fine-Tuning/experiment

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load cuda/12.4.0
module list  # job documentation and metadata

#¬ This script runs GREATS with the hook-based gradient compression
#
# Usage:
#   sh online_batch_select_mmlu.sh GREATS 2 0.05 5 mmlu llama2 1 2e-05 11 1 sociology "" "Gaussian-64*64"
#

DATA_DIR=data
MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
DATA_SEED=3
JOB_NAME=llama3-1b-p${PERCENTAGE}-lora-seed${DATA_SEED}-hooks

method=$1
batch_size=$2
PERCENTAGE=$3
NVAL=$4
task=$5
model=$6
lora_alpha=$7
lr=$8
seed=${9:-"42"}
gradient_accumulation_steps=${10:-"1"}
subject=${11:-"world_religions"}
sparsification=${12:-""}
projection=${13:-""}

# Set combined_modules based on the task
if [ "$task" = "mmlu" ]; then
    combined_modules="q_proj k_proj v_proj o_proj"
else
    combined_modules="q_proj k_proj"
fi

# Call the hook-based training script
./core/scripts/train/warmup_lora_train_hooks.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$method" "$batch_size" "$subject" "$NVAL" "$task" "$combined_modules" "$lora_alpha" "$lr" "$gradient_accumulation_steps" "$seed" "$sparsification" "$projection"
