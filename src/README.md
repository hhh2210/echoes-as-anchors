# Source Code Repository

## Overview

This directory contains the complete source code implementation for studying the impact of "repetition" behavior in LLM reasoning, covering core functional modules including data processing, model training, evaluation, and visualization.

## Directory Structure

```
src/
├── data_processing/     # Data processing and conversion modules
├── evaluation/          # Model evaluation and experimental analysis
├── training/            # GRPO training implementation
├── visualization/       # Results visualization tools
└── utils/              # General utility functions
```

## Module Details

### 1. data_processing/ - Data Processing Module

Handles various data format conversions and MLP training pipeline.

#### Core Files:

- **convert_to_dpo.py**: Convert AM samples to DPO format
  - Remove verbose reasoning preambles
  - Create preference pairs: rejected (original) vs preferred (cleaned)

- **dataset_downloader.py**: Dataset download utility

- **mlp_pipeline/**: Complete MLP probe pipeline
  - `preprocess.py`: Data preprocessing
  - `labeling.py`: Data labeling logic
  - `trainer.py`: MLP trainer
  - `inference.py`: Runtime repeat detection and pruning
  - `utils.py`: Helper functions

### 2. evaluation/ - Evaluation Module

Contains various evaluation scripts and experimental analysis tools.

#### Core Evaluation Tools:

- **harness_eval.py**: Main evaluation script based on lm-evaluation-harness
  - Supports GSM8K, MMLU and other tasks
  - Configurable thinking mode
  - Generates detailed evaluation reports

- **logp_trim_experiment.py**: Log probability trimming experiment
  - Analyzes log probability changes before/after deduplication
  - Validates impact of repetition on model confidence

- **compare_trimmed_accuracy.py**: Trimmed accuracy comparison
  - Compares correct/incorrect answer groups
  - Generates detailed statistical reports

- **attention_analysis.py**: Attention mechanism analysis
  - Analyzes attention patterns in repetitive content

- **pruning_eval.py**: MLP pruning evaluation
  - Evaluates effectiveness of different pruning strategies

- **quick_check.py**: Quick evaluation checking tool

#### Data Splitting Tools:

- **split_gsm8k_by_accuracy.py**: Split GSM8K by accuracy
- **split_by_exact_match.py**: Split by exact match
- **convert_lm_eval_for_logp.py**: Convert format for logp analysis

### 3. training/ - Training Module

Implements GRPO (Group Relative Policy Optimization) training.

#### Core Components:

- **grpo_trainer.py**: Main GRPO training logic
  - Integrated repeat reward mechanism
  - Uses embedding similarity to compute repeat scores
  - Supports DeepSpeed distributed training
  - Configuration:
    - `w_repeat`: Repeat reward weight (-0.05)
    - `beta`: KL divergence coefficient (0.04)
    - `train_batch_size`: 2 (L20 GPU)

- **ref_server.py**: Reference model server
  - Provides reference log probability computation
  - HTTP service interface
  - Supports tensor serialization transport

### 4. visualization/ - Visualization Module

Generates visualization charts for training and evaluation results.

#### Visualization Tools:

- **plot_results.py**: Plot experiment result comparisons
  - GSM8K accuracy curves
  - Supports flexible-extract and strict-match metrics
  - Multi-experiment comparison charts

- **plot_loss.py**: Training loss curves
  - Tracks loss changes during training
  - Supports multi-step aggregation

- **plot_reward.py**: Reward distribution visualization
  - Shows repeat reward distributions
  - Analyzes reward trends

### 5. utils/ - Utility Module

Common utility functions and auxiliary features.

## Usage Examples

### Data Processing
```bash
# Convert AM data to DPO format
python src/data_processing/convert_to_dpo.py \
    input.jsonl output_dpo.jsonl \
    --min-preamble-words 5
```

### Model Training
```bash
# Start reference server
CUDA_VISIBLE_DEVICES=7 python src/training/ref_server.py

# GRPO training
CUDA_VISIBLE_DEVICES=2,4,5,6 deepspeed src/training/grpo_trainer.py \
    --gen_device 1
```

### Model Evaluation
```bash
# Run complete evaluation
python src/evaluation/harness_eval.py \
    --exp_dir experiments/w_repeat_-0.05 \
    --tasks gsm8k \
    --enable_thinking
```

### Results Visualization
```bash
# Plot accuracy comparison chart
python src/visualization/plot_results.py \
    --exp_dirs experiments/w_repeat_* \
    --output_path results/comparison.png
```

## Key Configuration

### Model Paths
- Main model: `/path/to/your/Qwen2.5-7B-Instruct/`
- Embedding model: `/path/to/your/Qwen3-Embedding-0.6B/`

### Training Parameters
- Repeat penalty weight: -0.05 (negative value penalizes repetition)
- KL coefficient: 0.04
- Batch size: 2 (L20 GPU limitation)
- Save interval: 20 steps

### Evaluation Settings
- Default task: GSM8K
- Evaluation metrics: exact_match (flexible-extract, strict-match)
- Batching: auto

## Environment Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- DeepSpeed
- sentence-transformers
- lm-evaluation-harness
- matplotlib (visualization)

## Important Notes

1. **GPU Allocation**: Ensure different processes use non-overlapping GPUs
2. **Memory Management**: L20 GPU has limited memory, pay attention to batch size settings
3. **API Configuration**: Configure correct API keys when using LLM labeling
4. **Path Settings**: Use absolute paths for all paths to avoid errors