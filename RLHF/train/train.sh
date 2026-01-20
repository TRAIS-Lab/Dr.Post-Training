#!/bin/bash

#SBATCH --job-name=RLHF-Train
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfwm-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Gradient-Streaming/RLHF/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

SCRATCH_DIR=/work/hdd/bfwm/phu1/Project
CODE_DIR=/u/phu1/Project

cd $CODE_DIR/Gradient-Streaming

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="$CODE_DIR/Gradient-Streaming:$PYTHONPATH"

# Base training arguments (static, never change)
export base_training_args="--bf16=True \
--lr_scheduler_type=linear \
--warmup_ratio=0.03 \
--weight_decay=0.0 \
--logging_steps=1 \
--report_to=none \
--enable_eval=True \
--eval_on_step_generations=True"

# Default values
model="EleutherAI/gpt-neo-2.7B"
reward_model=""  # Will be set based on task if not specified
task="toxicity"
seed=42

# Training settings
epochs=1  # Number of training epochs (used when max_steps <= 0)
batch_size=256
ppo_epochs=4
mini_batch_size=4
kl_estimator="k1"  # Options: k1, k2, k3 (k3 recommended - unbiased, low variance, always positive)
adap_kl_ctrl=true
target=70.0  # Target KL for AdaptiveKLController
target_kl=0.3  # Early stopping threshold in PPO epochs: skip step if KL > 1.5 * target_kl
early_stopping=true  # Enable early stopping based on target_kl
max_new_tokens="30"  # Will be set based on task if not specified
min_new_tokens=0
max_steps=-1  # -1 means no step limit (use epochs instead)
use_flash_attention=true

# data selection
data_selection="NA"  # NA, IIF, Streaming, or GREATS
filter_frac=1.0  # Fraction of negative samples to drop (1.0 = no filtering)
use_second_order=false # If true, use greedy selection with second-order interactions
n_val=1024  # Number of validation samples for data selection (0 = self-referencing)
val_batch_size=256  # Batch size for validation gradient computation
val_generation_interval=1  # Steps between regenerating val completions (0 = once at start)
val_loss_type="seqloss-reward"  # Options: seqloss-lastadv (GAE last-token), seqloss-reward (normalized reward)
# Config file (contains lr, lr_vhead, init_kl_coef per method)
config_file="RLHF/train/config.json"
lr_override=""  # Set when --lr is explicitly passed
lr_vhead_override=""  # Set when --lr_vhead is explicitly passed
init_kl_coef_override=""  # Set when --init_kl_coef is explicitly passed
# Default values (used when config.json has no entry for the method)
default_lr_full="1e-5"
default_lr_lora="1e-5"
default_lr_vhead="5e-4"
default_init_kl_coef="0.02"

# Evaluation settings
eval_interval=1  # 0 = end of epoch only, N > 0 = every N steps
n_eval=500
eval_batch_size=256

# LoRA settings
use_lora=false
lora_r=16
lora_alpha=32
lora_target_modules=""  # Empty = PEFT auto-detects (out_proj for GPT-Neo, o_proj for LLaMA)

# Compression settings
compression=""  # NA, GraSS, or LoGra. Compression implies MeSO optimizer and compressed gradient for data selection.
update_compressor_freq=200
score_compression_dim=0  # Auto score compression (LoGra factorized) dimension. Set to 0 to disable.

# Multi-method mode
methods="" # Comma-separated list of methods or categories
dry_run=false

# ========================================
# Experiment Definitions
# ========================================
# Format: "data_selection:compression:use_lora"
# Note: use_second_order is controlled by the --use_second_order flag, not per-method
declare -A METHOD_DEFS=(
    ["NA-NA-Full"]="NA::false"
    ["NA-NA-LoRA"]="NA::true"
    ["IIF-NA-Full"]="IIF::false"
    ["IIF-NA-LoRA"]="IIF::true"
    ["Streaming-NA-Full"]="Streaming::false"
    ["Streaming-NA-LoRA"]="Streaming::true"
    ["GREATS-NA-Full"]="GREATS::false"
    ["GREATS-NA-LoRA"]="GREATS::true"
    ["Streaming-LoGra-Full"]="Streaming:LoGra:false"
    ["GREATS-LoGra-Full"]="GREATS:LoGra:false"
)

# Category mappings
declare -A CATEGORY_METHODS=(
    ["all"]="NA-NA-Full,NA-NA-LoRA,IIF-NA-Full,IIF-NA-LoRA,Streaming-NA-Full,Streaming-NA-LoRA,GREATS-NA-Full,GREATS-NA-LoRA,Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["baseline"]="NA-NA-Full,NA-NA-LoRA"
    ["iif"]="IIF-NA-Full,IIF-NA-LoRA"
    ["streaming"]="Streaming-NA-Full,Streaming-NA-LoRA,Streaming-LoGra-Full"
    ["greats"]="GREATS-NA-Full,GREATS-NA-LoRA,GREATS-LoGra-Full"
    ["full"]="NA-NA-Full,IIF-NA-Full,Streaming-NA-Full,GREATS-NA-Full,Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["lora"]="NA-NA-LoRA,IIF-NA-LoRA,Streaming-NA-LoRA,GREATS-NA-LoRA"
    ["compression"]="Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["no-compression"]="NA-NA-Full,NA-NA-LoRA,IIF-NA-Full,IIF-NA-LoRA,Streaming-NA-Full,Streaming-NA-LoRA,GREATS-NA-Full,GREATS-NA-LoRA"
)

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            task="$2"
            shift 2
            ;;
        --data_selection)
            data_selection="$2"
            shift 2
            ;;
        --model)
            model="$2"
            shift 2
            ;;
        --reward_model)
            reward_model="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --lr)
            lr_override="$2"
            shift 2
            ;;
        --config)
            config_file="$2"
            shift 2
            ;;
        --filter_frac)
            filter_frac="$2"
            shift 2
            ;;
        --max_steps)
            max_steps="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --lora)
            use_lora=true
            shift 1
            ;;
        --lora_r)
            lora_r="$2"
            shift 2
            ;;
        --lora_alpha)
            lora_alpha="$2"
            shift 2
            ;;
        --lora_target_modules)
            lora_target_modules="$2"
            shift 2
            ;;
        --flash_attention)
            use_flash_attention=true
            shift 1
            ;;
        --no_flash_attention)
            use_flash_attention=false
            shift 1
            ;;
        --compression)
            compression="$2"
            shift 2
            ;;
        --update_compressor_freq)
            update_compressor_freq="$2"
            shift 2
            ;;
        --score_compression_dim)
            score_compression_dim="$2"
            shift 2
            ;;
        --ppo_epochs)
            ppo_epochs="$2"
            shift 2
            ;;
        --mini_batch_size)
            mini_batch_size="$2"
            shift 2
            ;;
        --kl_coef|--init_kl_coef)
            init_kl_coef_override="$2"
            shift 2
            ;;
        --lr_vhead)
            lr_vhead_override="$2"
            shift 2
            ;;
        --kl_estimator)
            kl_estimator="$2"
            shift 2
            ;;
        --target)
            target="$2"
            shift 2
            ;;
        --target_kl)
            target_kl="$2"
            shift 2
            ;;
        --early_stopping)
            early_stopping=true
            shift 1
            ;;
        --max_new_tokens)
            max_new_tokens="$2"
            shift 2
            ;;
        --use_second_order)
            use_second_order=true
            shift 1
            ;;
        --n_val)
            n_val="$2"
            shift 2
            ;;
        --val_batch_size)
            val_batch_size="$2"
            shift 2
            ;;
        --val_generation_interval)
            val_generation_interval="$2"
            shift 2
            ;;
        --val_loss_type)
            val_loss_type="$2"
            shift 2
            ;;
        --eval_interval)
            eval_interval="$2"
            shift 2
            ;;
        --n_eval)
            n_eval="$2"
            shift 2
            ;;
        --eval_batch_size)
            eval_batch_size="$2"
            shift 2
            ;;
        --methods)
            methods="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run single or multiple RLHF training methods."
            echo ""
            echo "Multi-Method Mode:"
            echo "  --methods <list>       Run multiple methods (see below)"
            echo "  --dry-run                  Print commands without executing"
            echo ""
            echo "  Method names:"
            echo "    NA-NA-Full, NA-NA-LoRA, IIF-NA-Full, IIF-NA-LoRA,"
            echo "    Streaming-NA-Full, Streaming-NA-LoRA, GREATS-NA-Full, GREATS-NA-LoRA,"
            echo "    Streaming-LoGra-Full, GREATS-LoGra-Full"
            echo ""
            echo "  Categories: all, baseline, iif, streaming, greats, Full, LoRA, compression, no-compression"
            echo ""
            echo "Single Method Options:"
            echo "  --task <task>              Task: toxicity (default: toxicity)"
            echo "  --data_selection <method>  Data selection: NA, IIF, Streaming, GREATS (default: NA)"
            echo "  --model <model>            Policy model"
            echo "  --reward_model <model>     Reward model (auto-selected if not specified)"
            echo ""
            echo "  --lora                     Enable LoRA fine-tuning (default: full fine-tuning)"
            echo "  --lora_r <r>               LoRA rank (default: 16)"
            echo "  --lora_alpha <alpha>       LoRA alpha (default: 32)"
            echo "  --lora_target_modules <m>  Target modules (default: auto-detect)"
            echo ""
            echo "  --compression <method>     Compression: LoGra, GraSS (implies MeSO optimizer)"
            echo "  --score_compression_dim <dim>  Score compression dimension (default: 64, set to 0 to disable)"
            echo "  --use_second_order         Enable second-order selection"
            echo ""
            echo "Shared Options:"
            echo "  --batch_size <size>        Training batch size (default: 256)"
            echo "  --lr <lr>                  Learning rate override (ignores config file)"
            echo "  --lr_vhead <lr>            Value head learning rate override (default: 1e-4)"
            echo "  --config <path>            Config file (default: RLHF/train/config.json)"
            echo "  --filter_frac <frac>       Fraction of negative samples to drop (default: 1.0)"
            echo "  --max_steps <steps>        Maximum training steps (default: -1, meaning no limit)"
            echo "  --epochs <n>               Number of training epochs (default: 1, used when max_steps <= 0)"
            echo "  --seed <seed>              Random seed (default: 42)"
            echo ""
            echo "PPO Options:"
            echo "  --ppo_epochs <n>           PPO epochs per batch (default: 4)"
            echo "  --mini_batch_size <n>      Mini-batch size for PPO updates (default: 8)"
            echo "  --kl_coef <coef>           Initial KL penalty coefficient (default: 0.1)"
            echo "  --kl_estimator <mode>      KL estimator: k1, k2, k3 (default: k1)"
            echo "  --target <val>             Target KL for AdaptiveKLController (default: 1.0)"
            echo "  --target_kl <val>          Early stopping threshold (default: 0.1)"
            echo "  --early_stopping           Enable early stopping when KL > 1.5 * target_kl"
            echo "  --max_new_tokens <n>       Maximum new tokens to generate (default: 30)"
            echo ""
            echo "Note: 'target' controls adaptive KL coefficient adjustment."
            echo "      'target_kl' controls early stopping (skip step if KL too high)."
            echo ""
            echo "Validation Options (for data selection):"
            echo "  --n_val <n>                Number of validation samples (default: 1024, 0=self-ref)"
            echo "  --val_batch_size <n>       Batch size for validation gradients (default: 64)"
            echo "  --val_generation_interval <n>  Steps between regenerating val completions (default: 1)"
            echo "  --val_loss_type <type>     Validation loss type: seqloss-lastadv (default), seqloss-reward"
            echo ""
            echo "Evaluation Options:"
            echo "  --eval_interval <n>        Evaluate every N steps (0=epoch end only, default: 1)"
            echo "  --n_eval <n>               Samples for full evaluation (default: 500)"
            echo "  --eval_batch_size <n>      Batch size for generation (default: 256)"
            echo ""
            echo "  Toxicity: Uses DaNLP classifier (independent of reward model)"
            echo ""
            echo "Learning Rate Resolution:"
            echo "  1. If --lr is specified, use that value"
            echo "  2. Otherwise, look up from config.json based on task + method"
            echo "  3. If not found, use fallback (1e-5)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Extract model name from path for use in output directories
# e.g., "EleutherAI/gpt-neo-2.7B" -> "gpt-neo-2.7B"
model_name=$(basename "$model")

# ========================================
# Resolve method names from categories
# ========================================
resolve_methods() {
    local input="$1"
    local resolved=""

    IFS=',' read -ra items <<< "$input"
    for item in "${items[@]}"; do
        item=$(echo "$item" | xargs)

        if [[ -n "${CATEGORY_METHODS[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,${CATEGORY_METHODS[$item]}"
            else
                resolved="${CATEGORY_METHODS[$item]}"
            fi
        elif [[ -n "${METHOD_DEFS[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,$item"
            else
                resolved="$item"
            fi
        else
            echo "ERROR: Unknown method or category: $item"
            echo "Valid methods: ${!METHOD_DEFS[*]}"
            echo "Valid categories: ${!CATEGORY_METHODS[*]}"
            exit 1
        fi
    done

    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# ========================================
# Look up LR from config file
# ========================================
lookup_lr() {
    local config_key="$1"
    local exp_name="$2"
    local is_lora="$3"

    # If lr_override is set, use it
    if [[ -n "$lr_override" ]]; then
        echo "$lr_override"
        return
    fi

    # Try to look up from config file
    if [[ -f "$config_file" ]]; then
        local looked_up_lr
        looked_up_lr=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    lr_val = config.get('$config_key', {}).get('$exp_name', {}).get('lr')
    if lr_val is not None:
        print(f'{lr_val:.0e}' if lr_val < 0.001 else f'{lr_val}')
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)

        if [[ $? -eq 0 ]] && [[ -n "$looked_up_lr" ]]; then
            echo "$looked_up_lr"
            return
        fi
    fi

    # Fallback to defaults
    if [ "$is_lora" = "true" ]; then
        echo "$default_lr_lora"
    else
        echo "$default_lr_full"
    fi
}

# ========================================
# Look up lr_vhead from config file
# ========================================
lookup_lr_vhead() {
    local config_key="$1"
    local exp_name="$2"

    # If lr_vhead_override is set, use it
    if [[ -n "$lr_vhead_override" ]]; then
        echo "$lr_vhead_override"
        return
    fi

    # Try to look up from config file
    if [[ -f "$config_file" ]]; then
        local looked_up_lr_vhead
        looked_up_lr_vhead=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    lr_val = config.get('$config_key', {}).get('$exp_name', {}).get('lr_vhead')
    if lr_val is not None:
        print(f'{lr_val:.0e}' if lr_val < 0.001 else f'{lr_val}')
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)

        if [[ $? -eq 0 ]] && [[ -n "$looked_up_lr_vhead" ]]; then
            echo "$looked_up_lr_vhead"
            return
        fi
    fi

    # Fallback to default
    echo "$default_lr_vhead"
}

# ========================================
# Look up init_kl_coef from config file
# ========================================
lookup_init_kl_coef() {
    local config_key="$1"
    local exp_name="$2"

    # If init_kl_coef_override is set, use it
    if [[ -n "$init_kl_coef_override" ]]; then
        echo "$init_kl_coef_override"
        return
    fi

    # Try to look up from config file
    if [[ -f "$config_file" ]]; then
        local looked_up_kl_coef
        looked_up_kl_coef=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    kl_val = config.get('$config_key', {}).get('$exp_name', {}).get('init_kl_coef')
    if kl_val is not None:
        print(kl_val)
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)

        if [[ $? -eq 0 ]] && [[ -n "$looked_up_kl_coef" ]]; then
            echo "$looked_up_kl_coef"
            return
        fi
    fi

    # Fallback to default
    echo "$default_init_kl_coef"
}

# ========================================
# Function to run a single method
# ========================================
run_single_method() {
    local exp_data_selection="$1"
    local exp_compression="$2"
    local exp_use_lora="$3"
    local exp_use_second_order="$4"
    local exp_lr="$5"
    local exp_lr_vhead="$6"
    local exp_init_kl_coef="$7"
    local exp_name="$8"

    # Start with base training args
    local exp_base_training_args="$base_training_args"

    # ========================================
    # Compression Configuration
    # ========================================
    local sparsification=""
    local projection=""
    if [[ -n "$exp_compression" ]]; then
        case "$exp_compression" in
            LoGra)
                # LoGra: Gaussian random projection
                if [[ "$exp_use_lora" == "true" ]]; then
                    sparsification="normal-128*128"
                else
                    sparsification="normal-512*512"
                fi
                ;;
            GraSS)
                # GraSS: Random mask sparsification + SJLT projection
                if [[ "$exp_use_lora" == "true" ]]; then
                    sparsification="random_mask-256*256"
                    projection="sjlt-16384"
                else
                    sparsification="random_mask-1024*1024"
                    projection="sjlt-262144"
                fi
                ;;
            *)
                echo "ERROR: Invalid compression method: $exp_compression"
                exit 1
                ;;
        esac
    fi

    # ========================================
    # Build Job Name
    # ========================================
    local job_type
    if [[ "$exp_use_lora" == "true" ]]; then
        job_type="LoRA"
    else
        job_type="Full"
    fi

    # Build method string (include second-order indicator if enabled)
    local method_str="${exp_data_selection}-${exp_compression:-NA}"
    if [[ "$exp_use_second_order" == "true" ]]; then
        method_str="${method_str}-2nd"
    fi

    # Build validation string
    local val_str="v${n_val}"
    if [[ "$n_val" -gt 0 ]]; then
        val_str="${val_str}b${val_batch_size}"
    fi

    # Job name format: task-model-method-job_type-lr-b-v-ppo-mb-kl-s
    # Includes: validation settings, PPO epochs, mini-batch size, KL coefficient
    local JOB_NAME="${task}-${model_name}-${method_str}-${job_type}-lr${exp_lr}-b${batch_size}-${val_str}-pe${ppo_epochs}-mb${mini_batch_size}-kl${exp_init_kl_coef}-s${seed}"

    local output_dir=$SCRATCH_DIR/Gradient-Streaming/RLHF/${JOB_NAME}
    if [[ ! -d $output_dir ]]; then
        mkdir -p $output_dir
    fi

    # Indicate config source
    local lr_source
    if [[ -n "$lr_override" ]]; then
        lr_source="(--lr override)"
    elif [[ -f "$config_file" ]]; then
        lr_source="(from config)"
    else
        lr_source="(fallback)"
    fi

    local lr_vhead_source
    if [[ -n "$lr_vhead_override" ]]; then
        lr_vhead_source="(--lr_vhead override)"
    elif [[ -f "$config_file" ]]; then
        lr_vhead_source="(from config)"
    else
        lr_vhead_source="(fallback)"
    fi

    local kl_source
    if [[ -n "$init_kl_coef_override" ]]; then
        kl_source="(--kl_coef override)"
    elif [[ -f "$config_file" ]]; then
        kl_source="(from config)"
    else
        kl_source="(fallback)"
    fi

    echo ""
    echo "=============================================="
    if [[ -n "$exp_name" ]]; then
        echo "  Running: $exp_name"
    fi
    echo "=============================================="
    echo "Job name: $JOB_NAME"
    echo "Task: $task"
    echo "Selection: $method_str"
    echo "Model: $model"
    echo "Training type: $job_type"
    echo ""
    echo "Training:"
    echo "  Learning rate: $exp_lr $lr_source"
    echo "  Learning rate (v_head): $exp_lr_vhead $lr_vhead_source"
    echo "  Batch size: $batch_size"
    echo "  PPO epochs: $ppo_epochs"
    echo "  Mini-batch size: $mini_batch_size"
    echo "  KL coefficient: $exp_init_kl_coef $kl_source"
    echo ""
    echo "Validation (for data selection):"
    if [[ "$n_val" -gt 0 ]]; then
        echo "  Mode: fixed (n_val=$n_val, batch_size=$val_batch_size)"
    else
        echo "  Mode: self-reference"
    fi
    if [[ "$exp_data_selection" != "NA" ]]; then
        echo "  Second-order: $([ "$exp_use_second_order" = "true" ] && echo "yes" || echo "no")"
    fi
    echo ""
    echo "Output: $output_dir"
    echo "=============================================="

    # ========================================
    # Build training arguments
    # ========================================
    local training_args="$exp_base_training_args \
--task=$task \
--method=$exp_data_selection \
--model_name_or_path=$model \
--reward_model_name=$reward_model \
--per_device_train_batch_size=$batch_size \
--learning_rate=$exp_lr \
--learning_rate_vhead=$exp_lr_vhead \
--filter_frac=$filter_frac \
--max_steps=$max_steps \
--num_train_epochs=$epochs \
--seed=$seed \
--ppo_epochs=$ppo_epochs \
--mini_batch_size=$mini_batch_size \
--init_kl_coef=$exp_init_kl_coef \
--kl_estimator=$kl_estimator \
--adap_kl_ctrl=$adap_kl_ctrl \
--target=$target \
--target_kl=$target_kl \
--early_stopping=$early_stopping \
--max_new_tokens=$max_new_tokens \
--min_new_tokens=$min_new_tokens \
--output_dir=$output_dir"

    # Add LoRA settings
    if [[ "$exp_use_lora" == "true" ]]; then
        training_args="$training_args --lora=True --lora_r=$lora_r --lora_alpha=$lora_alpha"
        # Only add target_modules if explicitly specified (empty = PEFT auto-detects)
        if [[ -n "$lora_target_modules" ]]; then
            training_args="$training_args --lora_target_modules $lora_target_modules"
        fi
    else
        training_args="$training_args --lora=False"
    fi

    # Add flash attention setting
    if [[ "$use_flash_attention" == "true" ]]; then
        training_args="$training_args --use_flash_attention=True"
    else
        training_args="$training_args --use_flash_attention=False"
    fi

    # Add compression settings
    if [[ -n "$sparsification" ]]; then
        training_args="$training_args --sparsification=$sparsification"
    fi
    if [[ -n "$projection" ]]; then
        training_args="$training_args --projection=$projection"
    fi
    if [[ -n "$sparsification" || -n "$projection" ]]; then
        training_args="$training_args --update_compressor_freq=$update_compressor_freq"
    fi

    # Add score compression dimension for data selection methods without explicit compression
    if [[ "$exp_data_selection" != "NA" ]] && [[ -z "$exp_compression" ]]; then
        training_args="$training_args --score_compression_dim=$score_compression_dim"
    fi

    # Add second-order flag
    if [[ "$exp_use_second_order" == "true" ]]; then
        training_args="$training_args --use_second_order=True"
    fi

    # Add validation settings (for data selection)
    training_args="$training_args --n_val=$n_val"
    training_args="$training_args --val_batch_size=$val_batch_size"
    training_args="$training_args --val_generation_interval=$val_generation_interval"
    training_args="$training_args --val_loss_type=$val_loss_type"

    # Add evaluation settings (enable_eval and eval_on_step_generations are in base_training_args)
    training_args="$training_args --eval_interval=$eval_interval"
    training_args="$training_args --n_eval=$n_eval"
    training_args="$training_args --eval_batch_size=$eval_batch_size"

    # Add log file
    training_args="$training_args 2>&1 | tee $output_dir/train.log"

    local cmd="python RLHF/train/train.py $training_args"

    echo "Running command:"
    echo "$cmd"
    echo ""

    # Execute or print command
    if [ "$dry_run" = true ]; then
        echo "[DRY-RUN] Would execute above command"
    else
        eval "$cmd"
    fi
}

# ========================================
# Set task-specific defaults
# ========================================
# Set reward model based on task if not specified
if [[ -z "$reward_model" ]]; then
    case "$task" in
        toxicity)
            reward_model="facebook/roberta-hate-speech-dynabench-r4-target"
            ;;
        *)
            echo "ERROR: Unknown task: $task. Valid: toxicity"
            exit 1
            ;;
    esac
fi

# Set max_new_tokens based on task if not specified
if [[ -z "$max_new_tokens" ]]; then
    max_new_tokens=30
fi

# ========================================
# Main execution logic
# ========================================
config_key="${task}"

if [[ -n "$methods" ]]; then
    # ========================================
    # Multi-method mode
    # ========================================
    resolved_methods=$(resolve_methods "$methods")

    echo ""
    echo "========================================================"
    echo "  RLHF Multi-Method Mode"
    echo "========================================================"
    echo "Task: $task"
    echo "Model: $model"
    echo "Seed: $seed"
    echo ""
    echo "Training Settings:"
    echo "  Batch size: $batch_size"
    echo "  Epochs: $epochs"
    echo "  Max steps: $max_steps (use -1 for epoch-based training)"
    echo "  Filter fraction: $filter_frac"
    echo ""
    echo "Validation (for data selection):"
    if [[ "$n_val" -gt 0 ]]; then
        echo "  Mode: fixed validation set"
        echo "  n_val: $n_val"
        echo "  val_batch_size: $val_batch_size"
        echo "  val_generation_interval: $val_generation_interval"
        echo "  val_loss_type: $val_loss_type"
    else
        echo "  Mode: self-reference (training buffer)"
        echo "  val_loss_type: $val_loss_type"
    fi
    echo ""
    echo "PPO Settings:"
    echo "  PPO epochs: $ppo_epochs"
    echo "  Mini-batch size: $mini_batch_size"
    echo "  Init KL coefficient: (per-method from config)"
    echo "  Adaptive KL control: $adap_kl_ctrl"
    echo "  Target (adaptive KL): $target"
    echo "  Target KL (early stop): $target_kl"
    echo "  Early stopping: $early_stopping"
    echo ""
    echo "Evaluation:"
    echo "  Eval interval: $eval_interval (0=epoch end only)"
    echo "  Eval samples: $n_eval"
    echo "  Eval batch size: $eval_batch_size"
    echo ""
    echo "Config: $config_file"
    echo "Methods to run: $resolved_methods"
    echo "========================================================"

    method_count=$(echo "$resolved_methods" | tr ',' '\n' | wc -l)
    current=0

    IFS=',' read -ra exp_list <<< "$resolved_methods"
    for exp_name in "${exp_list[@]}"; do
        current=$((current + 1))
        echo ""
        echo "========================================================"
        echo "  [$current/$method_count] $exp_name"
        echo "========================================================"

        IFS=':' read -ra exp_parts <<< "${METHOD_DEFS[$exp_name]}"
        exp_data_selection="${exp_parts[0]}"
        exp_compression="${exp_parts[1]}"
        exp_use_lora="${exp_parts[2]}"
        # use_second_order is controlled globally via --use_second_order flag
        exp_use_second_order="$use_second_order"

        # Look up learning rate, lr_vhead, and init_kl_coef
        exp_lr=$(lookup_lr "$config_key" "$exp_name" "$exp_use_lora")
        exp_lr_vhead=$(lookup_lr_vhead "$config_key" "$exp_name")
        exp_init_kl_coef=$(lookup_init_kl_coef "$config_key" "$exp_name")

        run_single_method "$exp_data_selection" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$exp_lr" "$exp_lr_vhead" "$exp_init_kl_coef" "$exp_name"
    done

    echo ""
    echo "========================================================"
    echo "  All methods completed!"
    echo "========================================================"
    echo "Total: $method_count methods"

else
    # ========================================
    # Single method mode
    # ========================================

    # Validate data_selection
    if [[ ! "$data_selection" =~ ^(NA|IIF|Streaming|GREATS)$ ]]; then
        echo "Error: data_selection must be NA, IIF, Streaming, or GREATS"
        exit 1
    fi

    # ========================================
    # Compression Configuration
    # ========================================
    # Default: LoGra (Gaussian projection). GraSS (random mask + SJLT) also available.
    sparsification=""
    projection=""
    if [[ -n "$compression" ]]; then
        case "$compression" in
            LoGra)
                # LoGra: Gaussian random projection
                if [ "$use_lora" = true ]; then
                    sparsification="normal-128*128"
                else
                    sparsification="normal-512*512"
                fi
                ;;
            GraSS)
                # GraSS: Random mask sparsification + SJLT projection
                if [ "$use_lora" = true ]; then
                    sparsification="random_mask-256*256"
                    projection="sjlt-16384"
                else
                    sparsification="random_mask-1024*1024"
                    projection="sjlt-262144"
                fi
                ;;
            *)
                echo "ERROR: Invalid compression method: $compression"
                echo "Valid options: LoGra, GraSS"
                exit 1
                ;;
        esac
    fi

    # Build method name for LR lookup
    local_compression=""
    if [[ -n "$compression" ]]; then
        local_compression="$compression"
    fi

    if [[ "$use_lora" == "true" ]]; then
        exp_name="${data_selection}-${local_compression:-NA}-LoRA"
    else
        exp_name="${data_selection}-${local_compression:-NA}-Full"
    fi

    # Look up learning rate, lr_vhead, and init_kl_coef
    lr=$(lookup_lr "$config_key" "$exp_name" "$use_lora")
    lr_vhead=$(lookup_lr_vhead "$config_key" "$exp_name")
    init_kl_coef=$(lookup_init_kl_coef "$config_key" "$exp_name")

    # Run single method
    run_single_method "$data_selection" "$local_compression" "$use_lora" "$use_second_order" "$lr" "$lr_vhead" "$init_kl_coef" ""
fi
