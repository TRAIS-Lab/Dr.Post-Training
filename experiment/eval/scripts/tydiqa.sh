#!/bin/bash
#
# TyDiQA Evaluation Script
#
# This script evaluates a trained model on TyDiQA dataset
#

model_path=$1
batch_size=$2
n_test=$3
output_file=$4

if [[ -z "$model_path" || -z "$n_test" ]]; then
    echo "Usage: $0 <model_path> <n_test> [output_file] [batch_size]"
    echo ""
    echo "Arguments:"
    echo "  model_path   : Path to the trained model checkpoint"
    echo "  batch_size   : Optional batch size (default: 1)"
    echo "  n_test       : Number of test examples to evaluate"
    echo "  output_file  : Optional output file path (default: model_path/tydiqa_results.json)"
    exit 1
fi

echo "======================================"
echo "TyDiQA Evaluation"
echo "======================================"
echo "Model Path:    $model_path"
echo "N Test:        $n_test"
echo "Batch Size:    $batch_size"
echo "======================================"

# Build the command
cmd="python -m core.eval.eval \
    --task tydiqa \
    --model_path $model_path \
    --n_test $n_test \
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
    echo "TyDiQA Evaluation completed successfully"
    echo "======================================"
else
    echo "======================================"
    echo "TyDiQA Evaluation failed with exit code $exit_code"
    echo "======================================"
fi

exit $exit_code
