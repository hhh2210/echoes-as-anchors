# Evaluation Module

This module provides comprehensive evaluation tools for analyzing the impact of repetitive content in model reasoning. It includes tools for model evaluation, log-probability analysis, attention pattern analysis, and statistical post-processing.

## 📁 Module Structure

```
evaluation/
├── __init__.py                         # Module initialization
├── attention_from_converted.py         # Attention pattern analysis on converted JSONL
├── compare_attention_ed_sft.py         # ED-SFT vs Baseline attention comparison (Qwen3)
├── compare_models_with_repeat.py       # Model comparison with repeat frequency analysis
├── compare_trimmed_accuracy.py         # Compare accuracy before/after trimming repetition
├── convert_lm_eval_for_logp.py        # Format converter for lm-eval outputs
├── harness_eval.py                     # lm-evaluation-harness integration
├── logp_trim_experiment.py            # Log-probability analysis after trimming
├── offline_stats.py                   # Statistical post-processing and reporting
├── pruning_eval.py                    # MLP-based pruning evaluation
├── split_by_exact_match.py            # Split samples by correctness
├── split_gsm8k_by_accuracy.py         # GSM8K-specific sample splitting
└── test_evaluation_pipeline.py        # Unit tests for evaluation pipeline
```

## 🔧 Core Components

### 1. Model Evaluation (`harness_eval.py`)

**Purpose**: Integration with lm-evaluation-harness for standardized model evaluation.

**Key Functions**:
- `evaluate_model()`: Run single model evaluation
- `parse_harness_results()`: Parse harness output files
- `evaluate_experiment_models()`: Batch evaluation of experiment checkpoints

**Usage**:
```bash
# Evaluate single model
python src/evaluation/harness_eval.py --model_path /path/to/model --tasks gsm8k

# Multi-GPU evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/evaluation/harness_eval.py \
    --model_path /path/to/model --use_multi_gpu --tensor_parallel_size 4
```

### 2. Repeat Frequency Analysis (`compare_models_with_repeat.py`)

**Purpose**: Comprehensive model comparison with MLP-based repeat detection.

**Key Features**:
- GSM8K accuracy evaluation
- MLP probe-based repeat frequency analysis
- Think token extraction and analysis
- Comparative reporting across models

**Usage**:
```bash
CUDA_VISIBLE_DEVICES=4,5 python src/evaluation/compare_models_with_repeat.py \
    --model_paths /path/to/model1 /path/to/model2 \
    --embedding_model_path /path/to/Qwen3-Embedding-0.6B/ \
    --mlp_probe_path train_mlp/models/repeat_mlp.pt \
    --tasks gsm8k \
    --output_dir results/evaluation_results
```

**Outputs**:
- `model_comparison_report.txt`: Detailed comparison report
- `{model_name}_repeat_tokens.jsonl`: Individual model repeat analysis

### 3. Log-Probability Analysis (`logp_trim_experiment.py`)

**Purpose**: Analyze log-probability differences before and after removing repetitive content.

**Key Metrics**:
- `gL_full`: Original answer log-probability
- `gL_trim`: Trimmed answer log-probability  
- `delta`: Difference (original - trimmed)

**Usage**:
```bash
python src/evaluation/logp_trim_experiment.py \
    --input_file converted.jsonl \
    --output_file results.json \
    --model /path/to/model
```

**Output Format**:
```json
{
  "summary": {
    "total_samples": 1000,
    "mean_logp_delta": -0.05,
    "std_logp_delta": 0.02,
    "negative_delta_ratio": 0.6
  },
  "details": [...]
}
```

### 4. Trimmed Accuracy Comparison (`compare_trimmed_accuracy.py`)

**Purpose**: Compare model performance on correct vs incorrect answers before/after repetition removal.

**Pipeline**:
1. Convert lm-eval format → logp format
2. Run log-probability analysis on both groups
3. Generate comparative statistics
4. Optional: Extended metrics and attention analysis

**Usage**:
```bash
python src/evaluation/compare_trimmed_accuracy.py \
    --correct_file samples_correct.jsonl \
    --wrong_file samples_wrong.jsonl \
    --model /path/to/model \
    --output_dir ./comparison_results
```

**Extended Analysis**:
- Length stratification
- Common suffix alignment
- Per-token impact analysis
- Optional attention pattern verification

### 5. Attention Pattern Analysis (`attention_from_converted.py`)

**Purpose**: Analyze attention patterns in model responses, focusing on question-answer relationships.

**Key Metrics**:
- `answer→question`: Attention from answer tokens to question tokens
- `answer→answer_prefix`: Self-attention within answer prefix
- `answer_tail→removed_prefix`: Attention to removed repetitive content

**Usage**:
```bash
python src/evaluation/attention_from_converted.py \
    --correct_converted samples_correct_converted.jsonl \
    --wrong_converted samples_wrong_converted.jsonl \
    --model /path/to/model \
    --output_dir attention_results \
    --answer_prefix_tokens 32
```

**Outputs**:
- `attention_metrics_correct.jsonl`: Per-sample attention metrics for correct answers
- `attention_metrics_wrong.jsonl`: Per-sample attention metrics for wrong answers
- `attention_summary.json`: Aggregated attention comparison

### 5.1 ED-SFT vs Baseline Attention Comparison (Qwen3-8B) (`compare_attention_ed_sft.py`)

**Purpose**: 按论文描述，验证 ED-SFT 使中间层(7-18)的 answer→answer-prefix 注意力与正确性更一致。对比 `Qwen3-8B-Base` 与 `Qwen3-8B-ED-SFT`。

**Inputs**: 需分别使用 lm-eval 评测得到两个模型各自的 `samples.jsonl`，并分割为正确/错误，再转换为 `converted.jsonl`（含 `problem/pred/is_correct`）。

**Steps**:
```bash
# 1) 对 Baseline 与 ED-SFT 分别跑 lm-eval（示例）
python src/evaluation/harness_eval.py --model_path /path/to/Qwen3-8B-Base --tasks gsm8k --output_dir base_eval/
python src/evaluation/harness_eval.py --model_path /path/to/Qwen3-8B-ED-SFT --tasks gsm8k --output_dir edsft_eval/

# 假设得到的样本文件如下（以 lm-eval 默认命名为例）：
# base_eval/evaluations/.../samples_gsm8k_*.jsonl
# edsft_eval/evaluations/.../samples_gsm8k_*.jsonl

# 2) 按正确/错误分割（保持各自模型内部自洽）
python src/evaluation/split_gsm8k_by_accuracy.py \
  --input_file base_eval/evaluations/.../samples_gsm8k_xxx.jsonl \
  --output_dir base_eval/split/

python src/evaluation/split_gsm8k_by_accuracy.py \
  --input_file edsft_eval/evaluations/.../samples_gsm8k_xxx.jsonl \
  --output_dir edsft_eval/split/

# 3) 转换为 converted.jsonl（包含 problem/pred/is_correct）
python src/evaluation/convert_lm_eval_for_logp.py \
  --input base_eval/split/samples_gsm8k_xxx_correct.jsonl \
  --output base_eval/split/correct_converted.jsonl

python src/evaluation/convert_lm_eval_for_logp.py \
  --input base_eval/split/samples_gsm8k_xxx_wrong.jsonl \
  --output base_eval/split/wrong_converted.jsonl

python src/evaluation/convert_lm_eval_for_logp.py \
  --input edsft_eval/split/samples_gsm8k_xxx_correct.jsonl \
  --output edsft_eval/split/correct_converted.jsonl

python src/evaluation/convert_lm_eval_for_logp.py \
  --input edsft_eval/split/samples_gsm8k_xxx_wrong.jsonl \
  --output edsft_eval/split/wrong_converted.jsonl

# 4) 运行对比脚本（聚焦中层 7-18 的 answer→answer-prefix）
CUDA_VISIBLE_DEVICES=0 \
python src/evaluation/compare_attention_ed_sft.py \
  --base_model /path/to/Qwen3-8B-Base \
  --edsft_model /path/to/Qwen3-8B-ED-SFT \
  --base_correct_converted base_eval/split/correct_converted.jsonl \
  --base_wrong_converted base_eval/split/wrong_converted.jsonl \
  --edsft_correct_converted edsft_eval/split/correct_converted.jsonl \
  --edsft_wrong_converted edsft_eval/split/wrong_converted.jsonl \
  --output_dir results/ed_sft_vs_base_attention \
  --answer_prefix_tokens 32 \
  --bucket_def "early:0-6,mid:7-18,late:19-31" \
  # 可选：若提供嵌入模型，并希望用探针估计的前缀长度作为窗口
  --use_mlp_echo_removal \
  --embedding_model_path /data1/public/models/Qwen3-Embedding-0.6B/ \
  --use_probe_prefix_len_for_ans_prefix
```

**Outputs**:
- `results/ed_sft_vs_base_attention/baseline/`：Baseline 的逐样本与 `layer_stats.json`
- `results/ed_sft_vs_base_attention/edsft/`：ED-SFT 的逐样本与 `layer_stats.json`
- `results/ed_sft_vs_base_attention/comparison_summary.json`：包含：
  - `base.mid_means.diff` 与 `edsft.mid_means.diff`：mid 层区间中 `正确-错误` 的 `answer→answer-prefix` 均值差
  - `delta.mid_diff_improvement`：ED-SFT 相对 Baseline 的改进幅度（×100 为百分比点）
  - `mid_auc` / `mid_d`：来自各自 `layer_stats.json` 的 mid 桶 AUC 与 Cohen's d

备注：`bucket_def` 默认 `mid:7-18`，与论文中“中间层(7-18)”一致；如模型层数不同，可按需要调整。

### 6. MLP-Based Pruning Evaluation (`pruning_eval.py`)

**Purpose**: Evaluate models with real-time MLP-based repetition removal.

**Usage**:
```bash
python src/evaluation/pruning_eval.py \
    --main_model_path /path/to/model \
    --embedding_model_path /path/to/embeddings \
    --mlp_probe_path /path/to/mlp.pt \
    --tasks gsm8k
```

**Strategies**:
- `terminate`: Stop generation when repetition detected
- `truncate_and_continue`: Remove repetition and continue generation

### 7. Statistical Post-Processing (`offline_stats.py`)

**Purpose**: Advanced statistical analysis of evaluation results.

**Key Analyses**:
- DeltaL decile analysis vs accuracy
- Removed tokens distribution vs accuracy
- Acceptance rate analysis (Zx≈1[removed_tokens>0])
- Spearman correlation between DeltaL and correctness
- Optional logistic regression

**Usage**:
```bash
python src/evaluation/offline_stats.py \
    --results_dir /path/to/analysis_results_YYYYMMDD_hhmmss
```

**Outputs**:
- `deltaL_deciles_vs_acc.csv`: Accuracy by DeltaL deciles
- `removed_tokens_bin_vs_acc.csv`: Accuracy by removal bins
- `spearman_correlation.txt`: Correlation analysis
- `latex_tables/*.tex`: LaTeX tables for papers

## 🔄 Common Workflows

### Workflow 1: Complete Model Evaluation

```bash
# 1. Generate model predictions with lm-eval
python src/evaluation/harness_eval.py --model_path /path/to/model --tasks gsm8k

# 2. Split samples by correctness
python src/evaluation/split_gsm8k_by_accuracy.py \
    --input_file samples.jsonl --output_dir split_results/

# 3. Compare trimmed accuracy
python src/evaluation/compare_trimmed_accuracy.py \
    --correct_file samples_correct.jsonl \
    --wrong_file samples_wrong.jsonl \
    --model /path/to/model \
    --output_dir comparison_results/
```

### Workflow 2: Multi-Model Comparison

```bash
# Compare multiple models with repeat analysis
python src/evaluation/compare_models_with_repeat.py \
    --model_paths /model1 /model2 /model3 \
    --embedding_model_path /path/to/embeddings \
    --mlp_probe_path /path/to/mlp.pt \
    --output_dir multi_model_results/
```

### Workflow 3: Deep Analysis Pipeline

```bash
# 1. Basic evaluation
python src/evaluation/harness_eval.py --model_path /path/to/model

# 2. Split and analyze
python src/evaluation/split_gsm8k_by_accuracy.py --input_file samples.jsonl --output_dir ./
python src/evaluation/compare_trimmed_accuracy.py \
    --correct_file samples_correct.jsonl \
    --wrong_file samples_wrong.jsonl \
    --run_attention  # Include attention analysis

# 3. Statistical post-processing
python src/evaluation/offline_stats.py --results_dir analysis_results_*/
```

## 📊 Key Metrics

### Accuracy Metrics
- **Exact Match**: Binary correctness from lm-eval
- **Flexible Extract**: Alternative parsing method
- **Strict Match**: Original lm-eval scoring

### Repetition Metrics  
- **Repeat Frequency**: Proportion of samples with detected repetition
- **Average Repeat Score**: Mean MLP confidence for repetition
- **Think Tokens**: Extracted reasoning content

### Log-Probability Metrics
- **DeltaL (Δ)**: `original_logp - trimmed_logp`
  - Negative: Repetition hurts probability
  - Positive: Repetition helps probability
- **Negative Delta Ratio**: Proportion of samples with Δ < 0

### Attention Metrics
- **Answer→Question**: Cross-attention to question tokens
- **Answer→Answer Prefix**: Self-attention within answer
- **Tail→Removed**: Attention to removed repetitive content

## 🔧 Utility Scripts

### Format Conversion
```bash
# Convert lm-eval format for logp analysis
python src/evaluation/convert_lm_eval_for_logp.py \
    --input lm_eval_output.jsonl --output converted.jsonl
```

### Sample Splitting
```bash
# Generic splitting by exact_match
python src/evaluation/split_by_exact_match.py \
    --samples samples.jsonl --output_dir split_results/

# GSM8K-specific splitting
python src/evaluation/split_gsm8k_by_accuracy.py \
    --input_file samples.jsonl --output_dir split_results/
```

## 🧪 Testing

Run unit tests for the evaluation pipeline:

```bash
python src/evaluation/test_evaluation_pipeline.py
```

**Test Coverage**:
- Sample data integrity validation
- Evaluation and grouping logic
- Log-probability calculation logic
- Statistical summary computation

## 📋 Dependencies

### Core Dependencies
- `torch`: Model inference and computation
- `transformers`: HuggingFace model integration  
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `tqdm`: Progress tracking

### Optional Dependencies
- `sentence_transformers`: Embedding models for repeat detection
- `statsmodels`: Advanced statistical analysis
- `scikit-learn`: Machine learning utilities
- `lm_eval`: lm-evaluation-harness integration

### Model Dependencies
- Main models: Any HuggingFace compatible model
- Embedding models: For semantic similarity (e.g., Qwen3-Embedding-0.6B)
- MLP probes: Trained repeat detection models

## ⚙️ Configuration

### Environment Variables
```bash
export HF_HOME=/home/user/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Common Parameters
- `--batch_size auto`: Automatic batch sizing
- `--temperature 0.0`: Deterministic generation  
- `--tensor_parallel_size 4`: Multi-GPU parallelism
- `--answer_prefix_tokens 32`: Analysis window size

## 📈 Output Formats

### CSV Outputs
- Tabular data for statistical analysis
- Suitable for plotting and further processing

### JSONL Outputs  
- Per-sample detailed metrics
- Preserves all intermediate calculations

### JSON Outputs
- Summary statistics and metadata
- Structured analysis results

### LaTeX Outputs
- Publication-ready tables
- Formatted for academic papers

## 🔍 Troubleshooting

### Common Issues
1. **Model loading failures**: Check trust_remote_code=True
2. **CUDA out of memory**: Reduce batch_size or use tensor_parallel_size
3. **Missing files**: Verify file paths and permissions
4. **Format errors**: Ensure JSONL files have one JSON object per line

### Performance Tips
1. Use multi-GPU for large models (`--use_multi_gpu`)
2. Implement proper CUDA device allocation
3. Monitor memory usage with large datasets
4. Use `--limit` parameter for quick testing

## 📚 References

- **lm-evaluation-harness**: https://slyracoon23.github.io/lm-evaluation-harness/
- **resps field**: Contains actual LLM outputs (important for analysis)
- **Project CLAUDE.md**: Contains additional configuration details

## ➕ Direct OpenAI API Inference + MLP Repeat Detection

不依赖 harness，直接通过 OpenAI 兼容接口（例如本地 `http://localhost:8000/v1`）调用多个模型在 GSM8K 上推理，并使用已训练的 MLP 探针进行重复检测与概率统计。

### 脚本

`src/evaluation/direct_gsm8k_openai_mlp.py`

### 依赖

- `openai>=1.0.0`
- `datasets`
- `sentence-transformers`
- `torch`
- `tqdm`

### 使用示例

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.evaluation.direct_gsm8k_openai_mlp \
  --base_url http://localhost:8000/v1 \
  --api_key EMPTY \
  --models qwen3-8B DeepSeek-8B gpt-oss \
  --split test \
  --limit 200 \
  --embedding_model_path /data1/public/models/Qwen3-Embedding-0.6B/ \
  --mlp_probe_path train_mlp/models/repeat_mlp.pt \
  --output_dir results/direct_openai_mlp \
  --mlp_threshold 0.9 \
  --answer_prefix_tokens 32 \
  --max_tokens 512 \
  --temperature 0.0
```

### 输出

- 每个模型一个 JSONL：`{model}_gsm8k_samples.jsonl`，包含：
  - `question`, `response`, `think_content`, `repeat_score`, `is_repeat`
- 汇总 JSON：`repeat_summary.json`，按模型聚合：
  - `repeat_frequency`（重复概率）、`repeat_count`、`samples_with_think` 等

### 说明

- `--models` 需与本地 OpenAI 兼容服务注册的 `model` 名称一致。
- `--base_url` 指向 OpenAI 兼容服务（例如 vLLM/OpenAI-oss 网关）。
- `--mlp_threshold` 越高越保守（默认 0.9）。
