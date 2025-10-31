#!/bin/bash
#
# MMLU Evaluation Script
#
# This script evaluates a trained model on MMLU dataset
#

model_path=$1
subject=$2
n_val=$3
n_eval=$4
output_file=$5
batch_size=${6:-1}

if [[ -z "$model_path" || -z "$subject" || -z "$n_val" || -z "$n_eval" ]]; then
    echo "Usage: $0 <model_path> <subject> <n_val> <n_eval> [output_file] [batch_size]"
    echo ""
    echo "Arguments:"
    echo "  model_path   : Path to the trained model checkpoint"
    echo "  subject      : MMLU subject to evaluate (e.g., sociology, world_religions)"
    echo "  n_val        : Number of validation examples for in-context learning"
    echo "  n_eval       : Number of evaluation examples"
    echo "  output_file  : Optional output file path (default: model_path/mmlu_results.json)"
    echo "  batch_size   : Optional batch size (default: 1)"
    exit 1
fi

echo "======================================"
echo "MMLU Evaluation"
echo "======================================"
echo "Model Path:    $model_path"
echo "Subject:       $subject"
echo "N Val:         $n_val"
echo "N Eval:        $n_eval"
echo "Batch Size:    $batch_size"
echo "======================================"

# Build the command
cmd="python -m experiment.core.eval.evaluate \
    --task mmlu \
    --model_path $model_path \
    --subject $subject \
    --n_val $n_val \
    --n_eval $n_eval \
    --batch_size $batch_size"

# Add output file if specified
if [[ -n "$output_file" ]]; then
    cmd="$cmd --output_file $output_file"
fi

# Run evaluation
echo "Running: $cmd"
eval "$cmd"

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "======================================"
    echo "MMLU Evaluation completed successfully"
    echo "======================================"
else
    echo "======================================"
    echo "MMLU Evaluation failed with exit code $exit_code"
    echo "======================================"
fi

exit $exit_code
