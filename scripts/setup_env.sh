#!/bin/bash
# Environment Setup Script for Train Repeat Project

echo "=================================================="
echo "Setting up Train Repeat Project Environment"
echo "=================================================="

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment: Repeat"
    source activate Repeat
else
    echo "Warning: conda not found, skipping environment activation"
fi

# Set HuggingFace cache paths (uncomment and modify as needed)
# export HF_HOME="$HOME/.cache/huggingface"
# export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
# export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"

echo "HuggingFace cache settings:"
echo "  HF_HOME: ${HF_HOME:-not set}"
echo "  HF_HUB_CACHE: ${HF_HUB_CACHE:-not set}"
echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE:-not set}"

# Set CUDA environment variables
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

echo ""
echo "CUDA environment settings:"
echo "  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "  TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"

# Check GPU status
echo ""
echo "GPU status check:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "Warning: nvidia-smi not available"
fi

# Check key Python packages
echo ""
echo "Checking key dependencies:"
python -c "
import sys
packages = ['torch', 'transformers', 'datasets', 'deepspeed', 'vllm', 'sentence_transformers']

for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✓ {pkg}')
    except ImportError:
        print(f'  ✗ {pkg} (not installed)')
"

echo ""
echo "Environment setup complete!"
echo "=================================================="