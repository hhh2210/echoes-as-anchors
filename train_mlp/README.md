# MLP Probe Training Pipeline for Repeat Detection

## Overview

This directory contains the complete pipeline for training and using MLP probes to detect "repetition" behavior during LLM reasoning. The system is designed to identify patterns where models repeat user questions within `<think>` tags and provide inference-time pruning functionality.

## File Structure

```
train_mlp/
├── preprocess_data.py          # Data preprocessing: Extract (question, thinking content) pairs from raw conversations
├── label_repeat.py              # Data labeling using LLM (supports semantic truncation)
├── label_repeat_semantic.py    # Pure semantic similarity labeling method
├── train_repeat_mlp.py          # MLP classifier training script
├── prune_inference_with_mlp.py # Inference-time repetition pruning using MLP
├── utils.py                     # Common utility function library
├── models/                      # Directory for trained models
│   └── repeat_mlp.pt           # Trained MLP probe model
└── am_0.9M_sample_1k.jsonl     # Sample dataset (1k samples)
```

## Core Function Modules

### 1. Data Preprocessing (`preprocess_data.py`)

**Function**: Extract (question, thinking content) pairs from raw conversation data

**Main Functions**:
- `extract_think_content(text)`: Extract content within `<think>...</think>` tags
- `process_raw_data(input_path, output_path)`: Process JSONL format conversation data

**Usage Example**:
```bash
python preprocess_data.py raw_conversations.jsonl qt_pairs.jsonl
```

### 2. Data Labeling

#### 2.1 LLM Labeling (`label_repeat.py`)

**Function**: Use GPT-4 or other LLM models for repetition detection labeling

**Main Features**:
- Support concurrent API calls for efficiency
- Support semantic truncation (`--use-semantic-truncation`)
- Hybrid labeling method (LLM + semantic similarity)

**Main Functions**:
- `parse_llm_response(text)`: Parse JSON response from LLM
- `label_with_llm_async(question, think_content)`: Asynchronous LLM labeling
- `process_data(input_path, output_path, ...)`: Batch process dataset

**Usage Examples**:
```bash
# Pure LLM labeling
CUDA_VISIBLE_DEVICES=4 python label_repeat.py qt_pairs.jsonl labels.jsonl

# Hybrid method with semantic truncation (recommended)
CUDA_VISIBLE_DEVICES=4 python label_repeat.py qt_pairs.jsonl labels_hybrid.jsonl --use-semantic-truncation
```

#### 2.2 Semantic Similarity Labeling (`label_repeat_semantic.py`)

**Function**: Pure semantic similarity-based labeling method

**Main Functions**:
- `find_repetition_boundary()`: Find repetition boundary using sentence-level semantic similarity
- `label_dataset()`: Label entire dataset

**Usage Example**:
```bash
CUDA_VISIBLE_DEVICES=0 python label_repeat_semantic.py qt_pairs.jsonl labels_semantic.jsonl
```

### 3. MLP Training (`train_repeat_mlp.py`)

**Function**: Train binary classification MLP probe to detect repetition patterns

**Main Components**:
- `RepeatDataset`: Custom dataset class with embedding caching support
- `RepeatDetector`: Two-layer MLP network (configurable hidden layer size)
- `train_model()`: Complete training loop with validation and early stopping

**Main Features**:
- Support embedding vector caching for accelerated training
- Automatic training curve plotting
- Save best model checkpoints

**Usage Example**:
```bash
CUDA_VISIBLE_DEVICES=0 python train_repeat_mlp.py labels_hybrid.jsonl models/repeat_mlp.pt \
    --hidden_dim 32 \
    --epochs 50 \
    --batch_size 64
```

### 4. Inference-time Pruning (`prune_inference_with_mlp.py`)

**Function**: Real-time detection and removal of repetitive content during generation

**Main Components**:
- `RepeatDetector`: MLP probe model definition (consistent with training)
- `RepetitionPruningLogitsProcessor`: Custom LogitsProcessor implementing pruning logic

**Pruning Strategies**:
- `terminate`: Terminate generation when repetition detected
- `truncate_and_continue`: Remove repetitive part and continue generation

**Usage Example**:
```python
from prune_inference_with_mlp import RepeatDetector, RepetitionPruningLogitsProcessor

# Load model
mlp_probe = RepeatDetector(input_dim=embedding_dim, hidden_dim=32)
mlp_probe.load_state_dict(torch.load("models/repeat_mlp.pt"))

# Create processor
processor = RepetitionPruningLogitsProcessor(
    question=user_question,
    tokenizer=tokenizer,
    embedder=embedding_model,
    mlp_probe=mlp_probe,
    device=device,
    remove_strategy="truncate_and_continue"
)

# Use during generation
outputs = model.generate(
    input_ids,
    logits_processor=LogitsProcessorList([processor]),
    ...
)
```

### 5. Utility Functions (`utils.py`)

**Provided Common Functions**:
- `init_nltk()`: Initialize NLTK sentence splitting tool
- `iter_dataset(path)`: JSONL file iterator
- `load_embedding_model()`: Load sentence embedding model
- `get_truncated_think_content()`: Truncate thinking content using semantic similarity
- `compute_embedding_features()`: Compute embedding feature vectors

## Complete Workflow

```bash
# 1. Data preprocessing
python preprocess_data.py raw_conversations.jsonl qt_pairs.jsonl

# 2. Data labeling (hybrid method recommended)
CUDA_VISIBLE_DEVICES=4 python label_repeat.py qt_pairs.jsonl labels_hybrid.jsonl \
    --use-semantic-truncation \
    --max-concurrent 10

# 3. Train MLP probe
CUDA_VISIBLE_DEVICES=0 python train_repeat_mlp.py labels_hybrid.jsonl models/repeat_mlp.pt \
    --hidden_dim 32 \
    --epochs 50 \
    --batch_size 64

# 4. Evaluate and use (see run_pruning_eval.py in parent directory)
cd ..
CUDA_VISIBLE_DEVICES=7 python run_pruning_eval.py \
    --main_model_path /path/to/llm \
    --embedding_model_path /path/to/embedder \
    --mlp_probe_path train_mlp/models/repeat_mlp.pt
```

## Configuration Parameters

### Embedding Model
- Default path: `/path/to/your/embedding_model/`
- Can be modified via parameters or configuration in each script

### LLM Labeling Configuration (label_repeat.py)
- API configuration defined at top of file
- Support concurrency control: `--max-concurrent`
- Semantic truncation thresholds: `--initial-threshold`, `--drop-threshold`

### MLP Training Hyperparameters
- Hidden layer dimension: `--hidden_dim` (default 32, 0 for linear probe)
- Batch size: `--batch_size` (default 64)
- Learning rate: `--learning_rate` (default 0.001)
- Training epochs: `--epochs` (default 50)
- Early stopping patience: `--patience` (default 10)

## Important Notes

1. **GPU Memory**: Embedding model and LLM inference require sufficient GPU memory
2. **API Limits**: Be aware of API concurrency limits when using LLM labeling
3. **Caching**: Embedding vectors are automatically cached during training to speed up subsequent runs
4. **Data Format**: Input data must be JSONL format with `messages` field in each line

## Dependencies

```python
torch
transformers
sentence-transformers
openai
tqdm
nltk
matplotlib (optional, for plotting)
numpy
```

## Performance Metrics

- Training on 1k samples takes approximately 300 minutes (including embedding computation)
- With cached embeddings, retraining takes only about 10 minutes
- Typical accuracy: 85-90% (depending on data quality and labeling method)