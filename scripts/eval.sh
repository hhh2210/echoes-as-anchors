#!/bin/bash
# Evaluation Script for Train Repeat Project
#
# Usage: ./scripts/eval.sh [exp_dir] [gpu_ids]
# Example: ./scripts/eval.sh /path/to/your/output_dir/exp_w_repeat_... "1,2,3,4"

set -e

# Default values
EXP_DIR=${1:-""}
GPU_IDS=${2:-"1,2,3,4"}
TASKS=${3:-"gsm8k"}
TEMPERATURE=${4:-""}

echo "=================================================="
echo "Starting Model Evaluation - Train Repeat Project"
echo "=================================================="
echo "Experiment directory: $EXP_DIR"
echo "GPU devices: $GPU_IDS"
echo "Evaluation tasks: $TASKS"
if [ -n "$TEMPERATURE" ]; then
    echo "Generation temperature: $TEMPERATURE"
fi
echo "=================================================="

# Check experiment directory
if [ -z "$EXP_DIR" ]; then
    echo "Error: Please provide experiment directory path"
    echo "Usage: $0 [exp_dir] [gpu_ids]"
    exit 1
fi

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Experiment directory does not exist: $EXP_DIR"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Check GPU count and set parallelism
GPU_COUNT=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
echo "Detected $GPU_COUNT GPU(s)"

EVAL_ARGS="--exp_dir $EXP_DIR --tasks $TASKS --use_multi_gpu --tensor_parallel_size $GPU_COUNT"

if [ -n "$TEMPERATURE" ]; then
    EVAL_ARGS="$EVAL_ARGS --temperature $TEMPERATURE"
fi

echo "Starting evaluation..."
python src/evaluation/harness_eval.py $EVAL_ARGS

echo "Evaluation complete! Results saved in $EXP_DIR/evaluations/"