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
#SBATCH --output=/u/%u/Project/Gradient-Streaming/SFT/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

cd $HOME/Project/Gradient-Streaming

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="$HOME/Project/Gradient-Streaming:$PYTHONPATH"

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
--selection_frac=0.5 \
--use_flash_attention=True"

# Default values
data_selection="NA"  # NA or Streaming
optim="adamw_torch"  # Standard HF optimizer (adamw_torch, adamw_hf, etc.)
data_dir="SFT/data"
percentage=0.05
n_val=5
n_eval=500
model="llama3-1b"
batch_size=4
val_batch_size=""  # Defaults to batch_size if not specified
lr=5e-05
seed=42
gradient_accumulation_steps=1
task="mmlu"
train_dataset=""  # Training dataset (if empty, uses task-based default)
subject="world_religions"
compression=""  # NA, GraSS, or LoGra. Compression implies MeSO optimizer.
selection_mode="per_layer"  # per_layer (Streaming) or global (GREATS)
use_second_order=false  # If true, use greedy selection with second-order interactions
use_lora=false
use_flash_attention=true

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
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --val_batch_size)
            val_batch_size="$2"
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
        --train)
            train_dataset="$2"
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
        --selection_mode)
            selection_mode="$2"
            shift 2
            ;;
        --use_second_order)
            use_second_order=true
            shift 1
            ;;
        --update_compressor_freq)
            update_compressor_freq="$2"
            shift 2
            ;;
        --lora)
            use_lora=true
            shift 1
            ;;
        --flash_attention)
            use_flash_attention=true
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
            echo ""
            echo "Options:"
            echo "  --data_selection <method>              Data selection: NA, Streaming (default: NA)"
            echo "  --selection_mode <mode>                Selection mode: per_layer (Streaming), global (GREATS) (default: per_layer)"
            echo "  --compression <method>                 Compression: GraSS, LoGra (implies MeSO optimizer) (default: none)"
            echo "  --use_second_order                     Use greedy selection with second-order interactions"
            echo ""
            echo "  --lora                                 Use LoRA fine-tuning (default: full fine-tuning)"
            echo "  --lora_alpha <alpha>                   LoRA alpha (default: 1)"
            echo "  --lora_r <r>                           LoRA rank (default: 8)"
            echo "  --lora_dropout <dropout>               LoRA dropout (default: 0.1)"
            echo ""
            echo "  --model <model>                        Model: llama3-1b, llama2-7b, etc. (default: llama3-1b)"
            echo "  --task <task>                          Task: mmlu, bbh, tydiqa, samsum, gsm8k (default: mmlu)"
            echo "  --train <dataset>                      Training dataset (default: task-based)"
            echo "  --subject <subject>                    MMLU/BBH subject (default: world_religions)"
            echo ""
            echo "  --batch_size <size>                    Training batch size (default: 4)"
            echo "  --val_batch_size <size>                Validation batch size for selection (default: same as batch_size)"
            echo "  --lr <lr>                              Learning rate (default: 5e-05)"
            echo "  --percentage <pct>                     Data sampling percentage (default: 0.05)"
            echo "  --n_val <n>                            Validation examples (default: 5)"
            echo "  --n_eval <n>                           Evaluation examples (default: 500)"
            echo "  --seed <seed>                          Random seed (default: 42)"
            echo ""
            echo "  --update_compressor_freq <steps>       Projector refresh interval (default: 200)"
            echo "  --flash_attention                      Enable Flash Attention 2"
            echo "  --data_dir <dir>                       Data directory (default: SFT/data)"
            echo ""
            echo "Naming convention: {selection}-{compression}-{training_type}"
            echo "  Examples: Streaming-NA-full, GREATS-GraSS-lora, NA-LoGra-full"
            exit 1
            ;;
    esac
done

# ========================================
# Update base_training_args with user-specified Flash Attention setting
# ========================================
if [ "$use_flash_attention" = true ]; then
    base_training_args="${base_training_args/--use_flash_attention=False/--use_flash_attention=True}"
fi

# ========================================
# Compression Configuration
# ========================================
# Compression implies MeSO optimizer. No compression = standard optimizer.

# Validate and configure compression method
sparsification=""
projection=""
if [[ -n "$compression" ]]; then
    case "$compression" in
        GraSS)
            # if lora is used, use smaller sketch sizes
            if [ "$use_lora" = true ]; then
                sparsification="random_mask-256*256"
                projection="sjlt-16384"
            else
                sparsification="random_mask-1024*1024"
                projection="sjlt-262144"
            fi
            ;;
        LoGra)
            if [ "$use_lora" = true ]; then
                sparsification="normal-128*128"
                projection=""
            else
                sparsification="normal-512*512"
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


# ========================================
# Build Job Name
# ========================================
# Naming convention: {selection}-{compression}-{training_type}
# Examples: Streaming-NA-full, GREATS-GraSS-lora, NA-LoGra-full

if [ "$use_lora" = true ]; then
    job_type="lora"
else
    job_type="full"
fi

# Build selection string
# - NA: no data selection
# - Streaming: per-layer selection (selection_mode=per_layer)
# - GREATS: global selection (selection_mode=global)
if [[ "$data_selection" == "Streaming" ]]; then
    if [[ "$selection_mode" == "global" ]]; then
        selection_str="GREATS"
    else
        selection_str="Streaming"
    fi
else
    selection_str="NA"
fi

# Build method string: {selection}-{compression}
method_str="${selection_str}-${compression:-NA}"

# Add second-order suffix for data selection experiments
if [[ "$data_selection" == "Streaming" ]] && [ "$use_second_order" = true ]; then
    method_str="${method_str}-2nd"
fi

# Build train string for job name
train_str="${train_dataset:-default}"

# For MMLU/BBH, include subject in job name
if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
    JOB_NAME="${train_str}_${task}_${subject}-${method_str}-${model}-${job_type}-p${percentage}-lr${lr}-b${batch_size}-v${n_val}-s${seed}"
else
    JOB_NAME="${train_str}_${task}-${method_str}-${model}-${job_type}-p${percentage}-lr${lr}-b${batch_size}-v${n_val}-s${seed}"
fi

output_dir=/scratch/pbb/Project/Gradient-Streaming/SFT/${JOB_NAME}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

echo "=== Training Configuration ==="
echo "Job name: $JOB_NAME"
echo "Method: ${method_str}-${job_type}"
echo ""
echo "Model: $model"
echo "Training type: $([ "$use_lora" = true ] && echo "LoRA (alpha=$lora_alpha, r=$lora_r)" || echo "Full")"
echo "Task: $task"
echo "Training data: ${train_dataset:-default for $task}"
echo ""
echo "Selection: $selection_str"
if [[ "$data_selection" == "Streaming" ]]; then
    echo "  Selection mode: $selection_mode"
    echo "  Second-order: $([ "$use_second_order" = true ] && echo "yes" || echo "no")"
fi
echo "Compression: ${compression:-None}"
if [[ -n "$compression" ]]; then
    echo "  Optimizer: MeSO (compressed)"
else
    echo "  Optimizer: AdamW (standard)"
fi
echo ""
echo "Batch size: $batch_size (val: ${val_batch_size:-$batch_size})"
echo "Learning rate: $lr"
echo "Data percentage: $percentage"
echo "Validation examples: $n_val"
echo "Evaluation examples: $n_eval"
echo ""
echo "Output: $output_dir"
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
-m SFT.train.train"

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

# Add train_dataset_names if specified
if [[ -n "$train_dataset" ]]; then
    training_args="$training_args --train_dataset_names $train_dataset"
fi

# Add val_batch_size if specified (defaults to train batch size if not specified)
if [[ -n "$val_batch_size" ]]; then
    training_args="$training_args --val_batch_size_for_selection $val_batch_size"
fi

# Add LoRA arguments if using LoRA
if [ "$use_lora" = true ]; then
    training_args="$training_args --lora True --lora_alpha $lora_alpha --lora_r $lora_r --lora_dropout $lora_dropout --lora_target_modules q_proj k_proj v_proj o_proj"
else
    training_args="$training_args --lora False"
fi

# Add gradient compression arguments (compression implies MeSO optimizer)
if [[ -n "$sparsification" ]]; then
    training_args="$training_args --sparsification $sparsification"
fi
if [[ -n "$projection" ]]; then
    training_args="$training_args --projection $projection"
fi
if [[ -n "$compression" ]]; then
    training_args="$training_args --update_compressor_freq $update_compressor_freq"
fi

# Add selection_mode for data selection (per_layer=Streaming, global=GREATS)
if [[ "$data_selection" == "Streaming" ]]; then
    training_args="$training_args --selection_mode $selection_mode"
fi

# Add use_second_order for data selection
if [[ "$data_selection" == "Streaming" ]] && [ "$use_second_order" = true ]; then
    training_args="$training_args --use_second_order True"
fi

training_args="$training_args 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args"
