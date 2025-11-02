#!/bin/bash
#
# Hook-based version of warmup_lora.sh
#
# This version uses train.py instead of train.py
#

source core/train/scripts/base_args.sh

data_dir=$1
model_path=$2
percentage=$3
data_seed=$4
job_name=$5

method=$6
batch_size=$7
subject=$8
nval=$9
n_eval=${10}          # Number of evaluation examples
task=${11}
combined_modules=${12}
lora_alpha=${13}
lr=${14}
gradient_accumulation_steps=${15}
seed=${16}
compression=${17}     # Optional: GraSS, LoGra, or Vanilla (default)

echo "Training with combined modules: $combined_modules"

output_dir=/scratch/pbb/Project/Efficient-Fine-Tuning/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

# Select training files based on task
if [[ $task == "samsum" ]]; then
    # ALPACA training data for SAMSUM evaluation
    train_files=(
        "$data_dir/train/alpaca/alpaca_data.jsonl"
    )
else
    # LESS (FLAN-v2+CoT+Dolly+OASST1) for testing with MMLU/TydiQA/BBH
    train_files=(
        "$data_dir/train/flan_v2/flan_v2_data.jsonl"
        "$data_dir/train/cot/cot_data.jsonl"
        "$data_dir/train/dolly/dolly_data.jsonl"
        "$data_dir/train/oasst1/oasst1_data.jsonl"
    )
fi

# Use fsdp for large models
if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
    elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
fi

ID=$RANDOM
# Generate a unique port for this job to avoid conflicts with other jobs
PORT=$((29400 + RANDOM % 10000))
export header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d --rdzv-endpoint=localhost:$PORT \
-m core.train.train"

training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--per_device_train_batch_size $batch_size \
--method $method \
--subject $subject \
--n_val $nval \
--n_eval $n_eval \
--analysis_dataset $task \
--lora_target_modules $combined_modules \
--lora_alpha $lora_alpha \
--learning_rate $lr \
--gradient_accumulation_steps $gradient_accumulation_steps \
--seed $seed"

# Process compression argument and set sparsification/projection accordingly
if [[ -n "$compression" ]]; then
    if [ "$compression" = "GraSS" ]; then
        sparsification="RandomMask-128*128"
        projection="SJLT-4096"
    elif [ "$compression" = "LoGra" ]; then
        sparsification=""
        projection="Gaussian-64*64"
    else
        # Vanilla or any other value: no compression
        sparsification=""
        projection=""
    fi

    # Add gradient compression arguments if set
    if [[ -n "$sparsification" ]]; then
        training_args="$training_args --sparsification $sparsification"
    fi
    if [[ -n "$projection" ]]; then
        training_args="$training_args --projection $projection"
    fi
fi

training_args="$training_args --train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args"
