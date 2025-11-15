#!/bin/bash

#SBATCH --job-name=Train
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdzy-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Efficient-Fine-Tuning/experiment/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

cd ~/Project/Efficient-Fine-Tuning

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="/home/pbb/Project/Efficient-Fine-Tuning:$PYTHONPATH"

# Base training arguments
export base_training_args="--do_train=True \
--do_eval=True \
--max_seq_length=512 \
--use_fast_tokenizer=True \
--lr_scheduler_type=linear \
--warmup_ratio=0.03 \
--weight_decay=0.0 \
--logging_steps=1 \
--eval_steps=100 \
--eval_strategy=steps \
--save_strategy=no \
--num_train_epochs=1 \
--bf16=True \
--tf32=False \
--fp16=False \
--overwrite_output_dir=True \
--report_to=none \
--seed=0 \
--percentage=1.0 \
--selection_frac=0.5"

# Default values
data_selection="NA"  # NA, GREATS, or GradNorm
MeSO=false  # Use MeSO compressed optimizer for memory efficiency
optim="adamw_torch"  # Standard HF optimizer (adamw_torch, adamw_hf, etc.)
data_dir="experiment/data"
percentage=0.05
n_val=5
n_eval=500
model="llama3-1b"
batch_size=4
lr=5e-05
seed=42
gradient_accumulation_steps=1
task="mmlu"
subject="world_religions"
compression=""  # Will be auto-set based on data_selection/use_compressed_optimizer
use_lora=false

# LoRA-specific defaults (only used if --lora is passed)
lora_alpha=1
lora_r=8
lora_dropout=0.1

update_compressor_freq=200

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_selection)
            data_selection="$2"
            shift 2
            ;;
        --optim)
            optim="$2"
            shift 2
            ;;
        --MeSO)
            MeSO=true
            shift 1
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --percentage)
            percentage="$2"
            shift 2
            ;;
        --n_val)
            n_val="$2"
            shift 2
            ;;
        --n_eval)
            n_eval="$2"
            shift 2
            ;;
        --model)
            model="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            gradient_accumulation_steps="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --subject)
            subject="$2"
            shift 2
            ;;
        --compression)
            compression="$2"
            shift 2
            ;;
        --update_compressor_freq)
            update_compressor_freq="$2"
            shift 2
            ;;
        --lora)
            use_lora=true
            shift 1
            ;;
        --lora_alpha)
            lora_alpha="$2"
            shift 2
            ;;
        --lora_r)
            lora_r="$2"
            shift 2
            ;;
        --lora_dropout)
            lora_dropout="$2"
            shift 2
            ;;
        --data_dir)
            data_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --data_selection <method>              Data selection method: NA, GREATS, GradNorm (default: NA)"
            echo "  --optim <optimizer>                    Optimizer: adamw_torch, adamw_hf, etc. (default: adamw_torch)"
            echo "  --MeSO                                 Use MeSO compressed optimizer for memory efficiency"
            echo "  --batch_size <size>                    Batch size (default: 2)"
            echo "  --percentage <pct>                     Data sampling percentage (default: 0.05)"
            echo "  --n_val <n>                            Number of validation examples (default: 5)"
            echo "  --n_eval <n>                           Number of evaluation examples (default: 500)"
            echo "  --model <model>                        Model name (default: llama3-1b)"
            echo "  --lr <lr>                              Learning rate (default: 5e-05)"
            echo "  --seed <seed>                          Random seed (default: 42)"
            echo "  --gradient_accumulation_steps <steps>  Gradient accumulation (default: 1)"
            echo "  --task <task>                          Task name: mmlu, samsum, tydiqa, bbh (default: mmlu)"
            echo "  --subject <subject>                    MMLU/BBH subject (default: world_religions)"
            echo "  --compression <method>                 Compression: GraSS, LoGra (auto-set if data_selection!=NA or optimizer=MeSO)"
            echo "  --update_compressor_freq <steps>              Projector refresh interval (default: 200)"
            echo "  --lora                                 Use LoRA fine-tuning (default: off - full fine-tuning)"
            echo "  --lora_alpha <alpha>                   LoRA alpha (default: 1, only used if --lora)"
            echo "  --lora_r <r>                           LoRA rank (default: 256, only used if --lora)"
            echo "  --lora_dropout <dropout>               LoRA dropout (default: 0.1, only used if --lora)"
            echo "  --data_dir <dir>                       Data directory (default: data)"
            exit 1
            ;;
    esac
done

# ========================================
# Compression Configuration
# ========================================
# Auto-enable compression if data_selection or MeSO is used
if [[ "$data_selection" != "NA" ]] || [[ "$MeSO" == "true" ]]; then
    if [[ -z "$compression" ]]; then
        compression="GraSS"
        echo "INFO: Automatically setting compression to GraSS (required for data_selection or MeSO)"
    fi
fi

# Validate and configure compression method
sparsification=""
projection=""
if [[ -n "$compression" ]]; then
    case "$compression" in
        GraSS)
			# if lora is used, use smaller sketch sizes
			if [ "$use_lora" = true ]; then
				sparsification="random_mask-128*128"
				projection="sjlt-4096"
			else
				sparsification="random_mask-1024*1024"
				projection="sjlt-16384"
			fi
			;;
        LoGra)
			if [ "$use_lora" = true ]; then
				sparsification="normal-64*64"
				projection=""
			else
				sparsification="normal-128*128"
				projection=""
			fi
            ;;
        *)
            echo "ERROR: Invalid compression method: $compression"
            echo "Valid options: GraSS, LoGra"
            exit 1
            ;;
    esac
fi


# Build job name
if [ "$use_lora" = true ]; then
    job_type="lora"
else
    job_type="full"
fi

# Build optimizer string for job name
if [ "$MeSO" = true ]; then
    optimizer_str="MeSO"
else
    optimizer_str="Regular"
fi

# Build method string for job name (data_selection-optimizer-compression)
method_str="${data_selection}-${optimizer_str}-${compression:-NA}"

# For MMLU/BBH, include subject in job name
if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
    JOB_NAME="${task}_${subject}-${method_str}-${model}-${job_type}-p${percentage}-lr${lr}-b${batch_size}-v${n_val}-s${seed}"
else
    JOB_NAME="${task}-${method_str}-${model}-${job_type}-p${percentage}-lr${lr}-b${batch_size}-v${n_val}-s${seed}"
fi

output_dir=/scratch/pbb/Project/Efficient-Fine-Tuning/${JOB_NAME}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

echo "=== Unified Training Script ==="
echo "Training mode: $([ "$use_lora" = true ] && echo "LoRA (LoRA alpha: $lora_alpha, LoRA rank: $lora_r)" || echo "Full Model")"
echo "Model: $model"
echo "Task: $task"
echo "Optimizer: $optim"
echo "Learning rate: $lr"
echo "Data selection: $data_selection"
echo "MeSO: $MeSO"
echo "Compression: ${compression:-None}"
echo "Batch size: $batch_size"
echo "Data percentage: $percentage"
echo "Validation examples: $n_val"
echo "Evaluation examples: $n_eval"
echo "Model output dir: $output_dir"
echo "==============================="


DATA_SEED=$((seed + 1))

# Map model shorthand to full path
case $model in
    llama3-1b)
        MODEL_PATH="meta-llama/Llama-3.2-1B"
        ;;
    llama2-7b)
        MODEL_PATH="meta-llama/Llama-2-7b-hf"
        ;;
    llama2-13b)
        MODEL_PATH="meta-llama/Llama-2-13b-hf"
    	base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
        ;;
    mistral-7b)
        MODEL_PATH="mistralai/Mistral-7B-v0.1"
    	base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
        ;;
    *)
        # Assume it's a full path
        MODEL_PATH="$model"
        ;;
esac

ID=$RANDOM
# Generate a unique port for this job to avoid conflicts with other jobs
PORT=$((29400 + RANDOM % 10000))
export header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d --rdzv-endpoint=localhost:$PORT \
-m experiment.train.train"

# Build training arguments
training_args="$base_training_args \
--model_name_or_path $MODEL_PATH \
--output_dir $output_dir \
--data_dir $data_dir \
--percentage $percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $batch_size \
--method $data_selection \
--subject $subject \
--n_val $n_val \
--n_eval $n_eval \
--analysis_dataset $task \
--learning_rate $lr \
--gradient_accumulation_steps $gradient_accumulation_steps \
--seed $seed \
--optim $optim"

# Add LoRA arguments if using LoRA
if [ "$use_lora" = true ]; then
    training_args="$training_args --lora True --lora_alpha $lora_alpha --lora_r $lora_r --lora_dropout $lora_dropout --lora_target_modules q_proj k_proj v_proj o_proj"
else
    training_args="$training_args --lora False"
fi

# Add gradient compression arguments (configured earlier)
if [[ -n "$sparsification" ]]; then
    training_args="$training_args --sparsification $sparsification"
fi
if [[ -n "$projection" ]]; then
    training_args="$training_args --projection $projection"
fi
if [[ -n "$compression" ]]; then
    training_args="$training_args --update_compressor_freq $update_compressor_freq"
fi

# Add MeSO compressed optimizer flag if enabled
if [ "$MeSO" = true ]; then
    training_args="$training_args --use_compressed_optimizer True"
fi

training_args="$training_args 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args"
