#!/bin/bash
#
# Hook-based version of warmup_lora_train.sh
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
task=${10}
combined_modules=${11}
lora_alpha=${12}
lr=${13}
gradient_accumulation_steps=${14}
seed=${15}
sparsification=${16}  # Optional
projection=${17}      # Optional

echo "Training with combined modules: $combined_modules"

output_dir=./out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

train_files=(
    "$data_dir/train/processed/flan_v2/flan_v2_data.jsonl"
    "$data_dir/train/processed/cot/cot_data.jsonl"
    "$data_dir/train/processed/dolly/dolly_data.jsonl"
    "$data_dir/train/processed/oasst1/oasst1_data.jsonl"
    )

# Use fsdp for large models
if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
    elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
fi

ID=$RANDOM
export header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d \
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
--analysis_dataset $task \
--lora_target_modules $combined_modules \
--lora_alpha $lora_alpha \
--learning_rate $lr \
--gradient_accumulation_steps $gradient_accumulation_steps \
--seed $seed"

# Add gradient compression arguments if provided
if [[ -n "$sparsification" ]]; then
    training_args="$training_args --sparsification $sparsification"
fi
if [[ -n "$projection" ]]; then
    training_args="$training_args --projection $projection"
fi

training_args="$training_args --train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args"
